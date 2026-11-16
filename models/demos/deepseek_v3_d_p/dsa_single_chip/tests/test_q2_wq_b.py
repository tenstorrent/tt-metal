"""Q2 wq_b linear: new generic op vs stock ttnn.linear on the exact per-chip shape.

in0 [1,1,160,1536] bf16 TILE (DRAM or L1 interleaved), wq_b [1536,4096] bf8 TILE DRAM interleaved,
out [1,1,160,4096] bf16 TILE L1 interleaved. Stock: 1D mcast config from mla_config.py, HiFi2, packer L1 acc."""
import statistics

import pytest
import torch
import ttnn
from models.demos.deepseek_v3_d_p.dsa_single_chip.ops import _program as P
from models.demos.deepseek_v3_d_p.dsa_single_chip.ops.noc_coords import _CACHE
from models.demos.deepseek_v3_d_p.dsa_single_chip.ops.wq_b_linear import wq_b_linear
from models.demos.deepseek_v3_d_p.dsa_single_chip.tests.bench_utils import open_device, close_device, to_dev, time_trace, pcc


def stock_fn(dev, x, w, in0_block_w=16):
    cc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False,
        packer_l1_acc=True)
    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(11, 10), in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=2,
        per_core_M=5, per_core_N=2, fuse_batch=True, mcast_in0=True)
    return lambda: ttnn.linear(x, w, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, program_config=pc,
                               compute_kernel_config=cc)


@pytest.mark.parametrize("in0_mem", ["dram", "l1"])
def test_q2_wq_b(in0_mem):
    dev = open_device()
    try:
        mem = ttnn.DRAM_MEMORY_CONFIG if in0_mem == "dram" else ttnn.L1_MEMORY_CONFIG
        for seed in (0, 1):
            torch.manual_seed(seed)
            xt = torch.randn(1, 1, 160, 1536)
            wt = torch.randn(1536, 4096) * 0.02
            x = to_dev(xt, dev, mem=mem)
            w = to_dev(wt, dev, dtype=ttnn.bfloat8_b)
            ref = (ttnn.to_torch(x).double() @ ttnn.to_torch(w).double()).float()
            s_out = ttnn.to_torch(stock_fn(dev, x, w)())
            out = P.alloc([1, 1, 160, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.L1_MEMORY_CONFIG)
            wq_b_linear(x, w, out, dev)
            got = ttnn.to_torch(out)
            p_new, p_stock = pcc(got, ref), pcc(s_out, ref)
            e_new, e_stock = (got - ref).abs().max().item(), (s_out - ref).abs().max().item()
            print(f"Q2[{in0_mem},seed{seed}] pcc vs fp64: new={p_new:.6f} stock={p_stock:.6f} | max abs err: "
                  f"new={e_new:.4f} stock={e_stock:.4f} | new vs stock pcc={pcc(got, s_out):.6f}")
            assert p_new >= p_stock

        # 200 traced replays must reproduce the first result bit for bit (claims and partial arrival are timing-dependent)
        first = ttnn.to_torch(out)
        for _ in range(4):
            time_trace(dev, lambda: wq_b_linear(x, w, out, dev), iters=50, repeats=1)
        assert torch.equal(ttnn.to_torch(out), first)

        rows = []
        for name, fn in (("stock in0_block_w=16 (requested config)", stock_fn(dev, x, w, 16)),
                         ("stock in0_block_w=8", stock_fn(dev, x, w, 8)),
                         ("new generic op", lambda: wq_b_linear(x, w, out, dev))):
            samples = [time_trace(dev, fn) for _ in range(5)]
            rows.append((name, statistics.median(samples), min(samples), max(samples)))
        base = rows[0][1]
        for name, med, lo, hi in rows:
            print(f"Q2[{in0_mem}] {name:40s} median {med:6.2f} us (min {lo:6.2f}, max {hi:6.2f})  "
                  f"speedup vs requested stock {base / med:.2f}x")
    finally:
        _CACHE.clear()
        close_device(dev)
