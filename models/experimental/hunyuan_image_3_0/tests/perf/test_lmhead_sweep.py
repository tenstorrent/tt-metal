# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Config sweep for the LM-head vocab projection matmul: M x 4096 x 133120
(HiFi2, BF16 act x BFP8 weight => BF16). This is tt/lm_head.py's ttnn.linear,
which has NO program_config — and at 3658 us / 10.2%% of total device time it is
the single largest op in the recaption AR profile (29%% DRAM, 9.5 TFLOPs on 110
cores; auto already spreads it wide, but the weight is genuinely huge: [4096,
133120] BFP8 ~= 545 MB, so DRAM read bandwidth is the real floor).

Sweeps auto vs an explicit 1D-mcast (split-N) config vs a DRAM-width-sharded
weight + width-sharded activation (the standard large-vocab-projection pattern:
each core's DRAM bank streams only ITS weight column-slice, instead of every
core reading through the interleaved weight via NOC hops) at Mt=1 (decode) and
Mt=2, with PCC vs a torch reference.

    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_lmhead_sweep.py -s

Result (WH, program cache on): 1D split-N is 2.39x at Mt=1 (3664 -> 1535 us) and
4.86x at Mt=2 (7867 -> 1619 us) — PCC 1.0001, no exotic requirements. Auto gets
WORSE than linearly as M grows (28.5 ms at M=256, 56.7 ms at M=512) but the 1D
config only tolerates Mt<=2 — Mt>=3 TT_THROWs (fuse_batch=True + growing
per_core_M overflows some CB/kernel-arg limit at this N=4160-tile width). So this
fix only covers the decode shapes (single/paired token); wider-M forward calls
(if any) still hit unmitigated-auto and would need a different config (e.g. a 2D
grid or DRAM-width-sharding — see below) to fix separately.

DRAM-sharded (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig) was also
tried and is NOT included in the sweep below — it needs input A, weight, AND
output ALL sharded consistently (3 chained TT_FATALs fixed in sequence:
DRAM-bank-aligned shard grid -> input-A-must-be-sharded -> output-must-be-
sharded), and even then TT_THROWs on a circular-buffer overflow (30 MB requested
vs 1.5 MB L1/core) that wasn't resolved — likely a per_core_n too large for 8
DRAM banks at N=133120 tiles. Not pursued further given 1D split-N's clean win
at the shapes that matter (decode); revisit for wide-M if needed.
"""
from __future__ import annotations

import time

import pytest
import torch
import ttnn

K, N = 4096, 133120  # hidden, vocab
_TILE = 32


def _bench(dev, fn, it=20):
    for _ in range(4):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(it):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / it * 1e6


def _pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _div_leq(x, cap):
    return next((d for d in range(min(x, cap), 0, -1) if x % d == 0), 1)


def _cfg_1d_split_n(grid, Mt, Kt, Nt):
    ncores = grid.x * grid.y
    ncols = _div_leq(Nt, ncores)
    pcn = Nt // ncols
    osw = next((w for w in (4, 3, 2, 1) if pcn % w == 0), 1)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=_div_leq(Kt, 8),
        out_subblock_h=1,
        out_subblock_w=osw,
        per_core_M=Mt,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


@pytest.mark.parametrize("S", [32, 64, 96], ids=["decode_mt1", "mt2", "mt3_over_boundary"])
def test_lmhead_sweep(device, S):
    device.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    M = ((S + _TILE - 1) // _TILE) * _TILE
    torch.manual_seed(0)
    x = torch.randn(1, S, K, dtype=torch.bfloat16) * 0.05
    w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02  # [H, V], already transposed for ttnn.linear
    ref = x.float().reshape(-1, K) @ w.float()

    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    w_il = ttnn.from_torch(
        w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    grid = device.compute_with_storage_grid_size()
    Mt, Kt, Nt = M // _TILE, K // _TILE, N // _TILE

    def mm(pc, weight=w_il):
        return ttnn.linear(
            xt,
            weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=ckc,
            program_config=pc,
        )

    print(f"\n=== lm_head {M} x {K} x {N} (S={S}, grid={grid.x}x{grid.y}) ===")
    results = []

    for name, pc, weight in [
        ("auto", None, w_il),
        ("1D split-N", _cfg_1d_split_n(grid, Mt, Kt, Nt), w_il),
    ]:
        try:
            out = ttnn.to_torch(mm(pc, weight)).reshape(-1, N)
            us = _bench(device, lambda pc=pc, weight=weight: mm(pc, weight))
            results.append((name, us, _pcc(out, ref)))
        except Exception as e:
            results.append((name, None, str(e)[:80]))

    # DRAM-width-sharded weight (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig)
    # was also tried and is NOT included here — see the module docstring "Result" note
    # for the 3 chained requirements it hit and the CB-overflow it didn't resolve.

    base = next((us for _, us, _ in results if us is not None), None)
    for name, us, extra in results:
        if us is None:
            print(f"  {name:26}    FAIL  {extra}")
        else:
            print(f"  {name:26} {us:9.1f} us  ({base/us:4.2f}x)  PCC={extra:.6f}")
