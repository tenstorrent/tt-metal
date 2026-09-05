# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal repro: `prepare_conv_weights` disagrees with `ttnn.conv1d` on Wormhole.

No model and no checkpoint -- a random weight, a random input, one convolution run two
ways. It exists to be pasted into an upstream issue.

**What happens.** For `Conv1d(128 -> 128, k=11, stride 1, padding 5)` over a `[1, L, 128]`
bfloat16 activation, `ttnn.conv1d` gives two different answers depending on whether its
weight was pre-transformed by `ttnn.prepare_conv_weights` or left for the op to prepare
itself. At some `L` the difference is a few percent; at others it is `1e37`. Blackhole
agrees exactly at every `L` tested, on two boards.

The two paths are supposed to be interchangeable -- preparation is hoisted out so that
convolutions can be captured in a trace, which the op's own weight transfer forbids.

**How it was found.** CosyVoice-300M's HiFT vocoder emits a 15x-too-loud waveform for one
streamed chunk on Wormhole and is correct on Blackhole. Bisecting inward: the streaming mel
cache -> the length it changes (110 -> 130 mel frames) -> the NSF source branch -> a Snake
activation returning `inf` -> the convolution feeding it, whose input maxes at 1.46 and
whose output reaches 1.58e38. `sin()` of that is `inf`, the vocoder's magnitude spectrum
rails at its `1e2` clip, and the waveform saturates.

**What it is not**, each ruled out by measurement rather than argument:

  - *not the input's tile padding.* Zeroing it -- by round-tripping the input through the
    host, or by re-tiling it on device -- changes nothing.
  - *not the HiFi4 + fp32-accumulate combination* tt-metal warns about on Wormhole:
    `HiFi3` fails identically.
  - *not fp32 accumulation.* Turning it off does not fix the failure, it **moves** the
    affected lengths.
  - *not arithmetic.* At `HiFi2` one length came back with **two** bad elements out of a
    million. Overflow does not produce two bad elements.
  - *not the input data.* Feeding the same conv a completely different activation returns
    the identical wrong value.
  - *not the weight values.* The prepared weight reads back at `max|w| = 0.1816`, exactly
    the stored weight, with no non-finite elements. It is the layout, not the numbers.

If the two columns below agree at every `L`, this geometry alone is not sufficient and the
model's own weights are needed -- `models/demos/cosyvoice/scripts/probe_prepared_weights.py`
is the version that drives the real module.

    python3 repro_conv1d_wormhole.py [--lengths 8129,8193,...] [--fidelity HiFi4]
"""
from __future__ import annotations

import argparse

import torch

import ttnn

# Around the band found in the vocoder. The lengths are `64 * mel_frames + 1`, which is why
# they are all 1 (mod 32); the sweep is not restricted to that form.
DEFAULT = "8064,8096,8128,8129,8160,8192,8193,8224,8256,8320,8321,8448,8576,8577,8608,8640,8704,8705,8736,9216,9217"

IN_C = OUT_C = 128
# The failing convolution is HiFT's source ResBlock: kernel 11, "same" padding,
# dilation 1. Kernel size matters -- an earlier version of this repro used k=3 and did
# not reproduce at any length.
K, PAD, DIL = 11, 5, 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default=DEFAULT)
    ap.add_argument("--fidelity", default="HiFi4")
    ap.add_argument("--fp32-acc", type=int, default=1)
    ap.add_argument("--l1", type=int, default=131072)
    ap.add_argument("--kernel", type=int, default=K)
    ap.add_argument("--dilation", type=int, default=DIL)
    args = ap.parse_args()
    k, dil = args.kernel, args.dilation
    pad = (k * dil - dil) // 2

    device = ttnn.open_device(device_id=0, l1_small_size=args.l1)
    try:
        compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, args.fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=bool(args.fp32_acc),
            packer_l1_acc=True,
        )
        conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, deallocate_activation=False)

        torch.manual_seed(0)
        w = torch.randn(OUT_C, IN_C, k) * 0.05
        # conv1d is conv2d at H=1, so the weight travels as OIHW with H=1.
        w_dev = ttnn.from_torch(w.reshape(OUT_C, IN_C, 1, k), dtype=ttnn.bfloat16)

        print(f"\n  arch {device.arch()}   grid {device.compute_with_storage_grid_size()}")
        print(
            f"  Conv1d({IN_C} -> {OUT_C}, k={k}, pad={pad}, dil={dil}), bfloat16, "
            f"{args.fidelity}, fp32_acc={bool(args.fp32_acc)}"
        )
        print(f"  input ~ N(0, 1)\n")
        print(f"  {'L':>8}{'L%32':>7}{'prepared':>14}{'raw weight':>14}{'torch':>11}{'  verdict'}")
        print("  " + "-" * 66)

        bad = []
        for L in [int(x) for x in args.lengths.split(",")]:
            torch.manual_seed(1)
            x = torch.randn(1, L, IN_C)
            ref = torch.nn.functional.conv1d(x.permute(0, 2, 1), w, padding=pad, dilation=dil)
            x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            prep = ttnn.prepare_conv_weights(
                weight_tensor=w_dev,
                weights_format="OIHW",
                has_bias=False,
                input_memory_config=x_dev.memory_config(),
                input_layout=x_dev.layout,
                in_channels=IN_C,
                out_channels=OUT_C,
                batch_size=1,
                input_height=1,
                input_width=L,
                kernel_size=(1, k),
                stride=(1, 1),
                padding=(0, pad),
                dilation=(1, dil),
                groups=1,
                device=device,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_config,
            )
            outs = []
            for w_use in (prep, w_dev):
                out, _ = ttnn.conv1d(
                    input_tensor=x_dev,
                    weight_tensor=w_use,
                    bias_tensor=None,
                    device=device,
                    in_channels=IN_C,
                    out_channels=OUT_C,
                    batch_size=1,
                    input_length=L,
                    kernel_size=k,
                    stride=1,
                    padding=pad,
                    dilation=dil,
                    groups=1,
                    conv_config=conv_config,
                    compute_config=compute,
                    dtype=ttnn.bfloat16,
                    return_output_dim=True,
                )
                v = ttnn.to_torch(out).float().reshape(-1)
                fin = v[torch.isfinite(v)]
                outs.append(float(fin.abs().max()) if fin.numel() else float("nan"))
                ttnn.deallocate(out)
            ref_max = float(ref.abs().max())
            # The two paths are meant to be interchangeable, so any disagreement counts.
            broken = not (abs(outs[0] - outs[1]) <= 0.02 * max(outs[1], 1e-9))
            if broken:
                bad.append(L)
            print(
                f"  {L:>8}{L % 32:>7}{outs[0]:>14.4g}{outs[1]:>14.4g}{ref_max:>11.3f}"
                f"{'   <-- DISAGREE' if broken else ''}"
            )
            ttnn.deallocate(x_dev)

        print(f"\n  {len(bad)} of {len(args.lengths.split(','))} lengths disagree: {bad}")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
