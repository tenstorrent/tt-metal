# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program-config sweep for the decode QKV matmul (32 × 3072 × 1536) on QB2 (TP4).

Holds FIXED (matches the production model so results transfer directly):
  * math fidelity  : HiFi2 (math_approx_mode + fp32_dest_acc + packer_l1_acc)
  * dtypes         : in0 BF16  ×  weight BFP8_B  =>  out BF16
  * weight layout  : DRAM width-sharded (``create_dram_sharded_mem_config``)
  * output layout  : L1 width-sharded (auto, as the model uses for decode)

SWEEPS only the DRAM-sharded matmul program config. For a DRAM-sharded matmul the
program config *is* the compute core grid plus the derived ``in0_block_w`` /
``per_core_N`` block sizes, so the activation's L1 width-shard grid necessarily
follows the swept core count (it cannot differ from the matmul grid). The baseline
the model ships today is the 4×8 = 32-core grid.

Run on the 1×4 mesh:
    MESH_DEVICE=P150x4 pytest \
      models/experimental/voxtraltts/tests/perf/test_qkv_matmul_config_sweep.py -sv
"""
from __future__ import annotations

import math
import time

import pytest
import torch
import ttnn
from loguru import logger

# Matmul shape (per device, after TP4): M × K × N
_M, _K, _N = 32, 3072, 1536

# Candidate compute grids as (y_rows, x_cols); num_cores = y*x must divide K/32 (=96)
# so the activation width-shard splits evenly. (4, 8) = 32 cores is the model baseline.
_SWEEP_GRIDS = [(1, 8), (2, 6), (2, 8), (3, 8), (4, 8), (6, 8), (8, 12)]

# Trace-replay timing: ops per captured trace, and number of trace executions.
_ITERS_IN_TRACE = 32
_LOOPS = 50


def _find_largest_divisor(n: int, max_divisor: int = 8) -> int:
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _dram_sharded_weight_mem_config(device, k: int, n: int) -> ttnn.MemoryConfig:
    """Replicates ModelArgs.create_dram_sharded_mem_config — FIXED across the sweep."""
    dram_cores = device.dram_grid_size().x
    assert device.dram_grid_size().y == 1, "DRAM sharding assumes y == 1"
    padded = math.ceil(n / (ttnn.TILE_SIZE * dram_cores)) * (ttnn.TILE_SIZE * dram_cores)
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _time_program_config(device, x, w, pc, ckc) -> float:
    """Per-op device time (µs) via trace replay (host overhead amortized)."""
    kw = dict(
        program_config=pc,
        compute_kernel_config=ckc,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    # Compile / warm up (also primes the program cache so the trace is steady-state).
    warm = ttnn.linear(x, w, **kw)
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [ttnn.linear(x, w, **kw) for _ in range(_ITERS_IN_TRACE)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(_LOOPS):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0

    ttnn.release_trace(device, tid)
    for o in outs:
        if o.is_allocated():
            ttnn.deallocate(o)
    return elapsed / (_LOOPS * _ITERS_IN_TRACE) * 1e6


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_qkv_matmul_program_config_sweep(device):
    """Sweep DRAM-sharded program configs for 32×3072×1536; report best by device time."""
    num_devices = device.get_num_devices()
    logger.info(f"Mesh: {num_devices} device(s); matmul {_M}×{_K}×{_N} per device (HiFi2, BF16×BFP8=>BF16)")

    # HiFi2 — identical to ModelArgs.compute_kernel_config_hifi2 (FIXED).
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Weight: BFP8_B, DRAM width-sharded (FIXED). Replicated onto every mesh rank.
    w_torch = torch.randn(1, 1, _K, _N, dtype=torch.bfloat16)
    weight = ttnn.from_torch(
        w_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_dram_sharded_weight_mem_config(device, _K, _N),
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    x_torch = torch.randn(1, 1, _M, _K, dtype=torch.bfloat16)

    results = []
    for rows, cols in _SWEEP_GRIDS:
        num_cores = rows * cols
        label = f"{rows}x{cols} ({num_cores}c)"
        if (_K // ttnn.TILE_SIZE) % num_cores != 0:
            logger.warning(f"[skip {label}] K/32={_K // ttnn.TILE_SIZE} not divisible by {num_cores}")
            continue
        try:
            # Activation: BF16, L1 width-sharded across exactly the matmul's cores.
            in_mem = ttnn.create_sharded_memory_config(
                (_M, _K // num_cores),
                ttnn.CoreGrid(y=rows, x=cols),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            x = ttnn.from_torch(
                x_torch,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=in_mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
            in0_block_w = _find_largest_divisor(_K // (ttnn.TILE_SIZE * num_cores))
            per_core_N = math.ceil(_N / (ttnn.TILE_SIZE * num_cores))
            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=in0_block_w,
                per_core_M=math.ceil(_M / ttnn.TILE_SIZE),
                per_core_N=per_core_N,
                fused_activation=None,
            )
            us = _time_program_config(device, x, weight, pc, ckc)
            ttnn.deallocate(x)
            results.append((us, label, num_cores, in0_block_w, per_core_N))
            logger.info(f"[{label}] in0_block_w={in0_block_w} per_core_N={per_core_N} -> {us:.2f} µs/op")
        except Exception as exc:  # invalid config for this shape/grid — record and continue
            logger.warning(f"[{label}] FAILED: {exc}")

    assert results, "No program config ran successfully"
    results.sort(key=lambda r: r[0])

    logger.info("=" * 78)
    logger.info(f"QKV matmul {_M}×{_K}×{_N} (HiFi2, BF16×BFP8=>BF16) — sorted best→worst")
    logger.info(f"{'grid':14} {'cores':>5} {'in0_block_w':>11} {'per_core_N':>10} {'µs/op':>8}")
    for us, label, nc, ib, pn in results:
        tag = "  <-- baseline" if nc == 32 else ""
        logger.info(f"{label:14} {nc:5d} {ib:11d} {pn:10d} {us:8.2f}{tag}")
    best = results[0]
    logger.info(f"BEST: {best[1]}  in0_block_w={best[3]} per_core_N={best[4]}  @ {best[0]:.2f} µs/op")
    logger.info("=" * 78)
