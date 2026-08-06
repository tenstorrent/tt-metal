# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Where inside the vocoder does the streamed chunk blow up, and is it length or content?

`probe_streaming_bisect.py` named `hift_mel` -- the 20 prepended mel frames -- as the
carrier. But "the cache is the carrier" has two very different readings, and the streaming
harness cannot tell them apart because prepending 20 frames changes **both**:

    with hift_mel     130 mel frames    RMS 0.929   <- 15x too loud
    without           110 mel frames    RMS 0.036

So is 130 a bad *length*, or are those 20 frames bad *content*? This probe answers that by
vocoding a **real, known-good mel** at a sweep of lengths, with no streaming machinery
anywhere: no fades, no splices, `cache_source=None`. If a length explodes on its own the
content is exonerated and the bug is a shape.

The second question is *where*. `wav` RMS 0.93 against the `+-0.99` clamp is not "loud", it
is **saturated** -- the signal is railed almost everywhere. Working backwards through
`decode`:

    wav = clamp(istft(mag*cos, mag*sin), +-0.99)      saturated
    mag = clamp(exp(conv_post_out), 0, 1e2)           => railed at 100
                                                      => conv_post_out >= ln(100) ~ 4.6

so something upstream of `conv_post` went large. Two things feed it: the mel (measured
normal, 6.67-7.39) and the NSF excitation `s`, which is the interesting one -- it is the
only fp32 path in the vocoder and its phase comes from a **blocked cumsum whose block count
is exactly `mel_frames`** (`source.py:phase_mod1` reshapes `[1, L*256, 9] -> [1, L, 256, 9]`
and scans dim 1 across L tiles). A length-dependent, architecture-dependent failure in a
scan over a non-tiled axis would look precisely like this.

So print RMS and max for every intermediate, at every length, and run it on both parts.

    python3 models/demos/cosyvoice/scripts/probe_hift_isolate.py [--lengths 110,130,172]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
GOLDEN = os.path.join(os.path.dirname(__file__), "..", "tests", "golden")

# 110 = chunk 1 with hift_mel disabled (works), 130 = chunk 1 as shipped (fails),
# 172 = chunk 0 (works), 282 = the whole non-streamed utterance (works). The rest fill in
# the gaps so a boundary, if there is one, is visible rather than inferred from four points.
DEFAULT_LENGTHS = (96, 100, 110, 120, 128, 129, 130, 131, 132, 144, 160, 172, 192, 206, 256, 282)


def stat(t) -> tuple[float, float]:
    """RMS and max-abs of a device tensor, as floats."""
    x = ttnn.to_torch(t).float()
    return float(x.pow(2).mean().sqrt()), float(x.abs().max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default=",".join(str(x) for x in DEFAULT_LENGTHS))
    # 32768 is what every device test uses, and it is not a floor to be raised freely:
    # the L1_SMALL bank is carved out of the same L1 the circular buffers live in, so a
    # generous 262144 makes the iSTFT's `conv_transpose2d` fail with "statically allocated
    # circular buffers clash with L1 buffers" -- more reserved memory, less room to run.
    ap.add_argument("--l1", type=int, default=32768)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    from models.demos.cosyvoice.tt.hifigan.generator import TtHiFTGenerator, shape_trace
    from models.demos.cosyvoice.tt.weights import WeightBag

    device = ttnn.open_device(device_id=0, l1_small_size=args.l1)
    try:
        hift = TtHiFTGenerator(device, WeightBag.load(os.path.join(GOLDEN, "hift_weights.npz")))

        # A real mel, not noise: the conv stack's behaviour at the tails of its dynamic
        # range is the thing under test, and synthetic input would not reach them.
        g = np.load(os.path.join(GOLDEN, "hift.inference.npz"))
        mel_full = torch.from_numpy(g["call0.in_speech_feat"]).float().permute(0, 2, 1).contiguous()  # [1, 282, 80]

        _gen = torch.Generator().manual_seed(1986)
        phase = torch.empty(1, 1, 9).uniform_(-3.14159265, 3.14159265, generator=_gen)
        phase[0, 0, 0] = 0.0

        def dev(v, dtype=ttnn.bfloat16):
            return ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        print(f"\n  arch {device.arch()}   grid {device.compute_with_storage_grid_size()}")
        print(f"  mel source: hift.inference golden, first L of 282 frames, RMS {mel_full.pow(2).mean().sqrt():.4f}")
        print(f"\n  {'L':>5}{'f0':>18}{'excitation s':>20}{'s_stft':>18}{'conv_post':>18}{'wav':>18}")
        print(f"  {'':>5}{'rms / max':>18}{'rms / max':>20}{'rms / max':>18}{'rms / max':>18}{'rms / max':>18}")
        print("  " + "-" * 95)

        for L in lengths:
            if L > mel_full.shape[1]:
                continue
            mel = dev(mel_full[:, :L, :].contiguous())
            trace = shape_trace(L, n_fft=hift.n_fft, hop_len=hift.hop_len)

            noise_unit = torch.randn(1, L * 256, 9, generator=torch.Generator().manual_seed(1986 + L))

            # --- the source branch, step by step -------------------------------
            f0 = hift.f0_predictor(mel, L, 1)
            up, audio_len = hift.upsample_f0(f0, L, 1)
            s_wide, _, _ = hift.m_source(
                up, phase_vec=dev(phase, ttnn.float32), sine_noise_unit=dev(noise_unit, ttnn.float32)
            )
            ttnn.deallocate(up)
            s = ttnn.typecast(s_wide, hift.dtype)
            ttnn.deallocate(s_wide)

            # --- decode, instrumented ------------------------------------------
            s_stft_raw, _ = hift.stft(s, trace["audio_length"], 1)
            s_stft = ttnn.permute(s_stft_raw, (0, 2, 1))
            ttnn.deallocate(s_stft_raw)

            x, _ = hift.conv_pre(mel, L, 1)
            stage_rms = []
            for st in trace["stages"]:
                act = ttnn.leaky_relu(x, hift.lrelu_slope)
                ttnn.deallocate(x)
                x, _ = hift.ups[st.index](act, st.in_length, 1)
                ttnn.deallocate(act)
                if st.index == hift.num_upsamples - 1:
                    head = ttnn.slice(x, [0, 1, 0], [1, 2, st.out_channels])
                    padded = ttnn.concat([head, x], dim=1)
                    ttnn.deallocate(head)
                    ttnn.deallocate(x)
                    x = padded
                si, _ = hift.source_downs[st.index](s_stft, trace["stft_frames"], 1)
                si_res = hift.source_resblocks[st.index](si, st.source_length, 1)
                ttnn.deallocate(si)
                # The source contribution is recorded separately from the main path: if
                # the excitation is the carrier this is the row that moves first.
                stage_rms.append((f"up{st.index}", stat(x)[0], f"src{st.index}", stat(si_res)[0]))
                nx = ttnn.add(x, si_res)
                ttnn.deallocate(si_res)
                ttnn.deallocate(x)
                x = nx
                acc = None
                for j in range(hift.num_kernels):
                    out = hift.resblocks[st.index * hift.num_kernels + j](x, st.padded_length, 1)
                    if acc is None:
                        acc = out
                    else:
                        nacc = ttnn.add(acc, out)
                        ttnn.deallocate(acc)
                        ttnn.deallocate(out)
                        acc = nacc
                ttnn.deallocate(x)
                x = ttnn.multiply(acc, 1.0 / hift.num_kernels)
                ttnn.deallocate(acc)

            act = ttnn.leaky_relu(x, 0.01)
            ttnn.deallocate(x)
            post, _ = hift.conv_post(act, trace["conv_post_length"], 1)
            ttnn.deallocate(act)

            wav = hift.decode(mel, s, L, 1)

            f0_s, s_s, ss_s, po_s, w_s = (stat(t) for t in (f0, s, s_stft, post, wav))
            flag = "   <-- SATURATED" if w_s[1] >= 0.98 and w_s[0] > 0.2 else ""
            print(
                f"  {L:>5}{f'{f0_s[0]:8.2f} /{f0_s[1]:7.1f}':>18}"
                f"{f'{s_s[0]:9.4f} /{s_s[1]:7.3f}':>20}"
                f"{f'{ss_s[0]:8.3f} /{ss_s[1]:7.2f}':>18}"
                f"{f'{po_s[0]:8.3f} /{po_s[1]:7.2f}':>18}"
                f"{f'{w_s[0]:8.4f} /{w_s[1]:7.4f}':>18}{flag}"
            )
            if flag:
                # Only print the per-stage walk for the failures -- on a healthy length it
                # is 4 uninformative rows per L.
                for nm, v, snm, sv in stage_rms:
                    print(f"        {nm:>6} rms {v:10.3f}      {snm:>6} rms {sv:10.3f}")

            for t in (mel, f0, s, s_stft, post, wav):
                ttnn.deallocate(t)

        print("\n  A length that explodes with known-good mel content exonerates the cache's")
        print("  values and makes this a shape bug. Compare the two architectures.")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
