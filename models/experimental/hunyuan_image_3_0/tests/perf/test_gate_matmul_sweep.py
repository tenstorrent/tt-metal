# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Config sweep for the MoE router (gate) projection matmul: M x 4096 x 64
(HiFi2, BF16 act x BFP8 weight => BF16). At the decode shape this is
bandwidth-bound — auto schedules it single-core SLOW (~0.4 TFLOPs / 3 GB/s).

The sweep A/Bs the candidate schedules against the auto baseline and prints
us/iter, speedup, and PCC vs auto so the fastest PCC-safe config can be adopted
in tt/moe/gate.py. Run:

    pytest models/experimental/hunyuan_image_3_0/tests/perf/test_gate_matmul_sweep.py -s

Result (WH, program cache on): auto ~90 us -> ~13 us (~6.8x), PCC 0.99998.
"1D split-N" (the config gate.py already uses via _skinny_mm_program_config) is
the winner. Splitting the K reduction across more cores ("gather-K") does NOT
help — with M and N both tiny, cross-core reduction overhead grows with core
count (nc=64 -> 35 us), and the ~512 KB BFP8 weight stream is already the floor.
"""
from __future__ import annotations

import time

import pytest
import torch
import ttnn

K, N = 4096, 64  # hidden, num_experts
_TILE = 32


def _bench(dev, fn, it=50):
    for _ in range(5):  # warm (program-cache + JIT)
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


# --- candidate program configs ------------------------------------------------
def _cfg_1d_split_n(grid, Mt, Kt, Nt):
    """1D multicast, split N across cores (the current gate.py / QKV style)."""
    ncores = grid.x * grid.y
    ncols = next(c for c in range(min(ncores, Nt), 0, -1) if Nt % c == 0)
    pcn = Nt // ncols
    osw = next(w for w in (4, 3, 2, 1) if pcn % w == 0)
    return "1D split-N", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=Kt // 8 if Kt % 8 == 0 else Kt,
        out_subblock_h=1,
        out_subblock_w=osw,
        per_core_M=Mt,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _cfg_1d_gather_k(grid, Mt, Kt, Nt, ncores):
    """1D multicast with mcast_in0=False (gather-in1): split the K reduction
    across `ncores` cores. K=128 tiles is the dominant axis for this shape, so
    parallelizing the reduction — not the tiny N — is the real lever."""
    if Kt % ncores:
        return None
    return f"1D gather-K nc={ncores}", ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=Kt // ncores,
        out_subblock_h=1,
        out_subblock_w=Nt,
        per_core_M=Mt,
        per_core_N=Nt,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )


@pytest.mark.parametrize("S", [1, 32, 33, 64], ids=["decode", "seq32", "seq33", "seq64"])
def test_gate_matmul_sweep(device, S):
    device.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    M = ((S + _TILE - 1) // _TILE) * _TILE
    torch.manual_seed(0)
    x = torch.randn(1, S, K, dtype=torch.bfloat16) * 0.05
    w = torch.randn(N, K, dtype=torch.bfloat16) * 0.02  # [experts, hidden], stored transposed
    ref = x.float().reshape(-1, K) @ w.float().t()

    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    wt = ttnn.from_torch(
        w.t().contiguous(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    grid = device.compute_with_storage_grid_size()
    Mt, Kt, Nt = M // _TILE, K // _TILE, N // _TILE

    def mm(program_config):
        return ttnn.linear(
            xt,
            wt,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=ckc,
            program_config=program_config,
        )

    candidates = [("auto", None), _cfg_1d_split_n(grid, Mt, Kt, Nt)]
    for nc in (2, 4, 8, 16, 32, 64, 128):
        c = _cfg_1d_gather_k(grid, Mt, Kt, Nt, nc)
        if c and nc <= grid.x * grid.y:
            candidates.append(c)

    base = None
    print(f"\n=== gate matmul {M} x {K} x {N} (S={S}) ===")
    for name, pc in candidates:
        try:
            out = ttnn.to_torch(mm(pc)).reshape(-1, N)
            us = _bench(device, lambda pc=pc: mm(pc))
            if base is None:
                base = us
            print(f"  {name:22} {us:8.1f} us  ({base/us:4.2f}x)  PCC={_pcc(out, ref):.6f}")
        except Exception as e:
            print(f"  {name:22}    FAIL  {str(e)[:80]}")
