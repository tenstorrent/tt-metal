# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode projection matmuls (M = 32 rows, bfp8 weights) on one chip: default `ttnn.linear` on interleaved
weights vs DRAM-width-sharded weights with the DRAM-sharded program config (one reader per DRAM bank, the
tt_transformers decode recipe) vs a 1D multicast config on the full grid. Traced replay wall per call and the
implied weight bandwidth. Shapes = the per-chip TP4 projections of GLM-5.3-Flash.

    TT_VISIBLE_DEVICES=4 pytest models/demos/glm53_flash/tests/test_matmul_configs.py -s
"""

from __future__ import annotations

import math
import time

import pytest
import torch
from loguru import logger

import ttnn

SHAPES = [  # (name, K, N)
    ("dsa q_a", 4096, 1536),
    ("dsa kv_a", 4096, 512),
    ("dsa q_b (16 heads)", 1536, 4096),
    ("dsa o_proj (TP4)", 4096, 4096),
    ("kda qkv (TP4)", 4096, 6144),
    ("kda o_proj (TP4)", 2048, 4096),
    ("shared w13 (TP4)", 4096, 1024),
    ("shared w2 (TP4)", 512, 4096),
    ("dense w13 (TP4)", 4096, 6144),
    ("dense w2 (TP4)", 3072, 4096),
]
M = 32
N_REP = 50


def _largest_divisor(n, max_divisor=8):
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _timed(device, fn, label, bytes_):
    out = fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(N_REP):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / N_REP * 1e6
    logger.info(f"  {label:44s} {us:8.1f} us  {bytes_ / us / 1e3:7.1f} GB/s")
    ttnn.release_trace(device, tid)
    return out, us


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30_000_000, "l1_small_size": 32768}], indirect=True)
def test_decode_matmul_configs(mesh_device):
    device = mesh_device
    dram = device.dram_grid_size()
    dram_cores = dram.x  # 8 on P150
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    hifi2 = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    for name, K, N in SHAPES:
        w = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
        x = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        ref = (x.float() @ w.float())[0, 0]
        bytes_ = K * N  # bfp8 ~1 B/elem
        logger.info(f"{name}: [{M} x {K}] x [{K} x {N}]")
        x_i = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        w_i = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        out, _ = _timed(
            device,
            lambda: ttnn.linear(x_i, w_i, dtype=ttnn.bfloat16, compute_kernel_config=hifi2),
            "default linear (interleaved)",
            bytes_,
        )
        p0 = torch.corrcoef(torch.stack([ref.flatten(), ttnn.to_torch(out).float()[0, 0].flatten()]))[0, 1].item()
        # DRAM width-sharded weights, one reader per bank
        padded = math.ceil(N / (32 * dram_cores)) * (32 * dram_cores)
        w_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (K, padded // dram_cores), ttnn.ShardOrientation.ROW_MAJOR),
        )
        try:
            w_ds = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=w_mem)
            n_cores = dram_cores
            act_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cores - 1, 0))})
            x_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(act_grid, (M, K // n_cores), ttnn.ShardOrientation.ROW_MAJOR),
            )
            x_s = ttnn.to_memory_config(x_i, x_mem)
            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=_largest_divisor(K // (32 * n_cores)),
                per_core_M=M // 32,
                per_core_N=math.ceil(N / (32 * n_cores)),
                fused_activation=None,
            )
            out_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(act_grid, (M, padded // n_cores), ttnn.ShardOrientation.ROW_MAJOR),
            )
            out2, _ = _timed(
                device,
                lambda: ttnn.linear(
                    x_s,
                    w_ds,
                    program_config=pc,
                    memory_config=out_mem,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=hifi2,
                ),
                f"DRAM-sharded weights ({n_cores} readers)",
                bytes_,
            )
            o2 = ttnn.to_torch(out2).float()[0, 0][:, :N]
            p2 = torch.corrcoef(torch.stack([ref.flatten(), o2.flatten()]))[0, 1].item()
        except Exception as e:  # noqa: BLE001
            p2 = float("nan")
            logger.info(f"  DRAM-sharded variant failed: {str(e).splitlines()[0][:160]}")
        # 1D mcast on the full grid: cores must divide Nt
        Nt, Kt = N // 32, K // 32
        best = 1
        for c in range(1, grid.x * grid.y + 1):
            if Nt % c == 0:
                best = c
        cx = min(grid.x, best)
        cy = best // cx if best % cx == 0 else 1
        cores = cx * cy
        try:
            pc1 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(cx, cy),
                in0_block_w=_largest_divisor(Kt, 8),
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=Nt // cores,
                per_core_M=1,
                per_core_N=Nt // cores,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            out3, _ = _timed(
                device,
                lambda: ttnn.linear(x_i, w_i, program_config=pc1, dtype=ttnn.bfloat16, compute_kernel_config=hifi2),
                f"1D mcast_in0 {cx}x{cy} cores",
                bytes_,
            )
            p3 = torch.corrcoef(torch.stack([ref.flatten(), ttnn.to_torch(out3).float()[0, 0].flatten()]))[0, 1].item()
        except Exception as e:  # noqa: BLE001
            p3 = float("nan")
            logger.info(f"  1D mcast variant failed: {str(e).splitlines()[0][:160]}")
        logger.info(f"  pcc default {p0:.5f} dram-sharded {p2:.5f} 1d-mcast {p3:.5f}")
