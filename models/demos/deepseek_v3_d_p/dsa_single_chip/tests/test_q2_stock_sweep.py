"""Sweep stock ttnn.linear program configs for the Q2 wq_b shape (bf16 x bf8 -> bf16 L1, 11x10 grid, HiFi2)."""
import math

import torch
import ttnn
from models.demos.deepseek_v3_d_p.dsa_single_chip.tests.bench_utils import open_device, close_device, to_dev, time_trace, pcc

MT, KT, NT = 5, 48, 128
GX, GY = 11, 10


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def subblocks(pm, pn, limit=8):
    """(h, w) with h | pm, w | pn, h * w <= limit; the largest few by area."""
    out = [(h, w) for h in divisors(pm) for w in divisors(pn) if h * w <= limit]
    return sorted(out, key=lambda s: -s[0] * s[1])[:2]


def configs():
    out = []
    # 1D, in0 multicast, N split over cores
    for pn in (1, 2, 3, 4, 6, 8):
        if math.ceil(NT / pn) > GX * GY:
            continue
        for ibw in (2, 4, 6, 8, 12, 16, 24):
            for sh, sw in subblocks(MT, pn):
                out.append(("1d", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(GX, GY), in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
                    per_core_M=MT, per_core_N=pn, fuse_batch=True, mcast_in0=True), (ibw, sh, sw, MT, pn)))
    # 2D multicast: grid x splits N, grid y splits M
    for gy in (1, 5):
        pm = MT // gy
        for gx in range(6, GX + 1):
            pn = math.ceil(NT / gx)
            for ibw in (2, 4, 6, 8, 12, 16):
                for sh, sw in subblocks(pm, pn):
                    out.append(("2d", ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
                        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None, fuse_batch=True),
                        (gx, gy, ibw, sh, sw, pm, pn)))
    return out


def test_stock_sweep():
    dev = open_device()
    try:
        torch.manual_seed(0)
        xt = torch.randn(1, 1, 160, 1536)
        wt = torch.randn(1536, 4096) * 0.02
        x = to_dev(xt, dev)
        w = to_dev(wt, dev, dtype=ttnn.bfloat8_b)
        ref = (ttnn.to_torch(x).double() @ ttnn.to_torch(w).double()).float()
        cc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False,
            packer_l1_acc=True)

        def run(pc):
            return ttnn.linear(x, w, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=pc,
                               compute_kernel_config=cc)

        results = []
        for fam, pc, desc in [("auto", None, ())] + configs():
            try:
                o = run(pc)
                p = pcc(ttnn.to_torch(o), ref)
                ttnn.deallocate(o)
                us = time_trace(dev, lambda: run(pc), iters=20, repeats=3)
                results.append((us, fam, desc, p))
                print(f"SWEEP {fam} {desc} {us:.2f}us pcc={p:.6f}", flush=True)
            except Exception as e:
                print(f"SWEEP {fam} {desc} FAIL {str(e).splitlines()[0][:100]}", flush=True)
        results.sort()
        print("SWEEP TOP 15:")
        for us, fam, desc, p in results[:15]:
            print(f"SWEEPTOP {us:.2f}us {fam} {desc} pcc={p:.6f}")
    finally:
        close_device(dev)
