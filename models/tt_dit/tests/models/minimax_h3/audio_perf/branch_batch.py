"""Can the 3 parallel AMP branches run as one grouped conv, and is it actually cheaper?

`vocoder_ltx.py:462-478` runs num_kernels=3 AMP branches sequentially over the same stage input and
averages them. AUDIO_FUSION_PLAN.md flags batching them along the channel axis as "nearly free, and it
also widens rows, so it wins twice" -- but that was reasoning, not measurement, and it glosses over a
correctness requirement: an AMP conv mixes all input channels, so stacking three branches into one
tensor would let them leak into each other. Keeping them separate needs `groups=3` and a
block-diagonal weight.

So this checks the two things that decide whether the idea is worth building:

  correctness  one grouped conv over concat([b0, b1, b2], dim=C) with a block-diagonal weight must
               equal three separate convs, bit-for-bit or near it
  speed        is 1 grouped conv at 3C actually faster than 3 convs at C? Today's compute_intensity
               result says cost tracks rows and op count, not arithmetic, so it should be ~3x -- but
               grouped convs may be dispatched differently, and that is the whole question.

Run standalone; no model needed.
"""

import statistics
import time

import torch

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3 import decoder_minimax_h3_audio  # noqa: F401

# stage shapes the decoder actually presents: (label, C, T)
CASES = [
    ("s3 C128", 128, 20701),
    ("s4 C64", 64, 41401),
    ("s5 C32", 32, 82801),
]
K = 3
ITERS = 5


def timed(fn, iters=ITERS):
    fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def conv1d(x_BTC, weight, *, B, C_in, C_out, T, K, groups, device, cache):
    return ttnn.conv1d(
        input_tensor=x_BTC,
        weight_tensor=weight,
        in_channels=C_in,
        out_channels=C_out,
        device=device,
        bias_tensor=None,
        kernel_size=K,
        stride=1,
        padding=K // 2,
        batch_size=B,
        input_length=T,
        groups=groups,
        dtype=ttnn.float32,
        conv_config=cache["cc"],
        compute_config=cache["mc"],
        return_output_dim=False,
        return_weights_and_bias=False,
    )


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        cache = {
            "cc": ttnn.Conv1dConfig(weights_dtype=ttnn.float32, deallocate_activation=False),
            "mc": ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
            ),
        }
        B = 2
        print(f"{'case':<10} {'3 separate ms':>14} {'1 grouped ms':>13} {'speedup':>8} {'exact':>7} {'maxdiff':>11}")
        print("-" * 70)
        for label, C, T in CASES:
            torch.manual_seed(0)
            xs = [torch.randn(B, T, C) * 0.3 for _ in range(3)]
            ws = [torch.randn(C, C, K) * (1.0 / (C * K) ** 0.5) for _ in range(3)]

            xds = [ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device) for x in xs]
            wds = [
                ttnn.from_torch(
                    w, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
                )
                for w in ws
            ]

            def separate():
                outs = [
                    conv1d(xds[i], wds[i], B=B, C_in=C, C_out=C, T=T, K=K, groups=1, device=device, cache=cache)
                    for i in range(3)
                ]
                return outs

            # grouped: stack inputs along C, weight is block-diagonal by construction of groups=3
            x_cat = torch.cat(xs, dim=2)
            w_cat = torch.cat(ws, dim=0)  # (3C, C, K): groups=3 reads C in-channels per group
            xcd = ttnn.from_torch(x_cat, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            wcd = ttnn.from_torch(
                w_cat, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
            )

            def grouped():
                return conv1d(xcd, wcd, B=B, C_in=3 * C, C_out=3 * C, T=T, K=K, groups=3, device=device, cache=cache)

            try:
                sep_out = separate()
                grp_out = grouped()
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<10}  FAILED {str(exc).splitlines()[0][:52]}")
                continue

            # conv1d returns (1, 1, B*T, C_out): the branches stack on the CHANNEL axis, dim=-1.
            got = ttnn.to_torch(grp_out).float()
            ref = torch.cat([ttnn.to_torch(o).float() for o in sep_out], dim=-1)
            if tuple(got.shape) != tuple(ref.shape):
                print(f"{label:<10}  SHAPE {tuple(got.shape)} != {tuple(ref.shape)}")
                continue
            d = float((got - ref).abs().max())

            t_sep = timed(separate)
            t_grp = timed(grouped)
            print(
                f"{label:<10} {t_sep:>14.3f} {t_grp:>13.3f} {t_sep / t_grp:>7.2f}x "
                f"{str(torch.equal(got, ref)):>7} {d:>11.3e}"
            )
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
