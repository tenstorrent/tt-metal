"""Is ttnn.linear(..., activation="silu") actually FUSED, or is it a second dispatch?

mm_block_ab.py produced an inversion that needs explaining before anything is shipped:
    w2  isolated 2.43x faster  -> in-block  0.00 ms
    wo  isolated 2.33x faster  -> in-block  0.00 ms
    w13 isolated 1.03x faster  -> in-block -3.24 ms
-3.24 ms over 47 layer-passes is 69 us per pass, which is almost exactly the p150's per-op cost
(6.45). And the w13 arm is the ONLY one that also moved silu from linear's `activation=` kwarg
into the program config's `fused_activation`. So the suspicion is that the win has nothing to do
with the grid or in0_block_w at all -- `activation=` is dispatching a second op, and the program
config is what finally fuses it.

If that is right the fix is far simpler and safer than a tuned grid: fuse silu, leave every grid
on the default heuristic. It would also apply to w1 in BOTH blocks, 47 sites per frame.

ARMS on the real w1 shape (K=3072, N=9216), all checked against a float64 reference built from
the device's own quantised weights:
    plain      linear(h, w1)                       -- no activation at all, the floor
    kwarg      linear(h, w1, activation="silu")    -- what ships
    explicit   linear(h, w1) then ttnn.silu(...)   -- deliberately two ops, the ceiling
    fused      linear(h, w1, program_config=<...fused_activation=SILU>)
    cfg-noact  linear(h, w1, program_config=<...fused_activation=None>)
The last one is the control that separates "the program config is good" from "the fusion is good".
"""
import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device

COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
K, N = 3072, 9216
SILU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
REPS = 300


def prg(act):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(12, 6), in0_block_w=2, out_subblock_h=1,
        out_subblock_w=4, per_core_M=1, per_core_N=4, fuse_batch=True,
        fused_activation=act, mcast_in0=True)


def main():
    dev = open_device()
    try:
        torch.manual_seed(0)
        wt = torch.randn(K, N) * 0.02
        w = ttnn.from_torch(wt.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                            device=dev)
        for rows in (1, 6):
            x = ttnn.from_torch((torch.randn(1, rows, K) * 0.02).contiguous(),
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            xd = ttnn.to_torch(x).double().reshape(rows, K)
            lin = xd @ ttnn.to_torch(w).double()
            ref_silu = torch.nn.functional.silu(lin)
            print(f"\n=== rows={rows}  K={K} N={N} ===")
            print(f"  {'arm':<32} {'us':>8} {'vs kwarg':>10} {'PCC vs silu(ref)':>18}")
            res = {}
            for lbl, fn, want_silu in (
                    ("plain linear (no activation)",
                     lambda: ttnn.linear(x, w, compute_kernel_config=COMPUTE), False),
                    ("kwarg activation='silu'  SHIPS",
                     lambda: ttnn.linear(x, w, activation="silu",
                                         compute_kernel_config=COMPUTE), True),
                    ("explicit linear + ttnn.silu",
                     lambda: ttnn.silu(ttnn.linear(x, w, compute_kernel_config=COMPUTE)), True),
                    ("prg cfg, fused_activation=SILU",
                     lambda: ttnn.linear(x, w, program_config=prg(SILU),
                                         compute_kernel_config=COMPUTE), True),
                    ("prg cfg, fused_activation=None",
                     lambda: ttnn.linear(x, w, program_config=prg(None),
                                         compute_kernel_config=COMPUTE), False)):
                try:
                    out = fn()
                    got = ttnn.to_torch(out).double().reshape(-1)[:rows * N].reshape(rows, N)
                    p = pcc(got, ref_silu if want_silu else lin)
                    fn(); ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(REPS):
                        fn()
                    ttnn.synchronize_device(dev)
                    us = (time.perf_counter() - t0) / REPS * 1e6
                    res[lbl] = us
                    base = res.get("kwarg activation='silu'  SHIPS")
                    d = f"{base - us:+9.1f}u" if base else f"{'':>10}"
                    print(f"  {lbl:<32} {us:>8.1f} {d:>10} {p:>18.7f}"
                          f"{'' if want_silu else '   (vs plain matmul)'}")
                except Exception as e:
                    print(f"  {lbl:<32} FAILED: {type(e).__name__}: "
                          f"{str(e).splitlines()[0][:50]}")
            if "plain linear (no activation)" in res and \
                    "kwarg activation='silu'  SHIPS" in res:
                gap = res["kwarg activation='silu'  SHIPS"] - res["plain linear (no activation)"]
                print(f"\n  activation='silu' costs {gap:+.1f} us over a plain matmul.")
                print(f"  If it were fused that gap would be ~0; the p150 per-op floor is ~68 us.")
            del x
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
