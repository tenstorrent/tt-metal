# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: Fused Parallel(Q_RMS, KV_RMS) vs unfused ttnn.rms_norm + ttnn.rms_norm.

Pre-allocates tensor sets to isolate dispatch cost from tensor creation.

Run:  python -m pytest tests/ttnn/unit_tests/operations/fused/parallel_sequential/bench_fused_vs_unfused_rms.py -xvs
"""

import time

import torch
import ttnn

from models.experimental.ops.descriptors.fusion import Parallel, clear_build_cache
from models.experimental.ops.descriptors.normalization.rms_norm import rms_norm


def cores(x1, y1, x2=None, y2=None):
    if x2 is None:
        x2, y2 = x1, y1
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x1, y1), ttnn.CoreCoord(x2, y2))})


def _make_configs():
    q_cores = cores(0, 0, 3, 3)
    kv_cores = cores(5, 0, 6, 7)
    q_shard_spec = ttnn.ShardSpec(q_cores, [32, 96], ttnn.ShardOrientation.ROW_MAJOR)
    q_mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=q_shard_spec,
    )
    q_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 4), subblock_w=3, block_h=1, block_w=3, inplace=False
    )
    kv_shard_spec = ttnn.ShardSpec(kv_cores, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
    kv_mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=kv_shard_spec,
    )
    kv_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 8), subblock_w=1, block_h=1, block_w=1, inplace=False
    )
    return q_cores, kv_cores, q_mem, kv_mem, q_pc, kv_pc


def _make_tensors(device, seed, q_mem, kv_mem):
    torch.manual_seed(seed)
    q_total_w = 16 * 96
    kv_total_w = 16 * 32
    tt_q_in = ttnn.from_torch(
        torch.rand(1, 1, 32, q_total_w, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem,
    )
    tt_q_w = ttnn.from_torch(
        torch.rand(1, 1, 1, q_total_w, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_kv_in = ttnn.from_torch(
        torch.rand(1, 1, 32, kv_total_w, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=kv_mem,
    )
    tt_kv_w = ttnn.from_torch(
        torch.rand(1, 1, 1, kv_total_w, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    return tt_q_in, tt_q_w, tt_kv_in, tt_kv_w


def test_fused_vs_unfused_rms(device):
    WARMUP = 20
    ITERS = 500
    N_TENSORS = 8

    q_cores, kv_cores, q_mem, kv_mem, q_pc, kv_pc = _make_configs()
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )

    # Pre-allocate tensor sets
    tensor_sets = [_make_tensors(device, seed=1000 + i, q_mem=q_mem, kv_mem=kv_mem) for i in range(N_TENSORS)]

    # Allocate unfused output tensors (reused across iterations)
    q_out_mem = q_mem
    kv_out_mem = kv_mem

    # ---- FUSED: Parallel(Q_RMS, KV_RMS) ----
    clear_build_cache()
    tt_q_in, tt_q_w, tt_kv_in, tt_kv_w = tensor_sets[0]
    q_op = rms_norm(
        tt_q_in,
        epsilon=1e-5,
        weight=tt_q_w,
        memory_config=q_mem,
        core_range_set=q_cores,
        program_config=q_pc,
        compute_kernel_config=compute_config,
    )
    kv_op = rms_norm(
        tt_kv_in,
        epsilon=1e-5,
        weight=tt_kv_w,
        memory_config=kv_mem,
        core_range_set=kv_cores,
        program_config=kv_pc,
        compute_kernel_config=compute_config,
    )
    p = Parallel(q_op, kv_op)
    fused = p.build()

    # Warmup
    for i in range(WARMUP):
        q_in, q_w, kv_in, kv_w = tensor_sets[i % N_TENSORS]
        q_op.input_tensors[0] = q_in
        kv_op.input_tensors[0] = kv_in
        fused.launch()
    ttnn.synchronize_device(device)

    # Timed
    t0 = time.perf_counter()
    for i in range(ITERS):
        q_in, q_w, kv_in, kv_w = tensor_sets[i % N_TENSORS]
        q_op.input_tensors[0] = q_in
        kv_op.input_tensors[0] = kv_in
        fused.launch()
    ttnn.synchronize_device(device)
    fused_us = (time.perf_counter() - t0) / ITERS * 1e6

    # ---- UNFUSED: ttnn.rms_norm Q then ttnn.rms_norm KV ----
    # Warmup
    for i in range(WARMUP):
        q_in, q_w, kv_in, kv_w = tensor_sets[i % N_TENSORS]
        ttnn.rms_norm(
            q_in,
            weight=q_w,
            epsilon=1e-5,
            memory_config=q_out_mem,
            program_config=q_pc,
            compute_kernel_config=compute_config,
        )
        ttnn.rms_norm(
            kv_in,
            weight=kv_w,
            epsilon=1e-5,
            memory_config=kv_out_mem,
            program_config=kv_pc,
            compute_kernel_config=compute_config,
        )
    ttnn.synchronize_device(device)

    # Timed
    t0 = time.perf_counter()
    for i in range(ITERS):
        q_in, q_w, kv_in, kv_w = tensor_sets[i % N_TENSORS]
        ttnn.rms_norm(
            q_in,
            weight=q_w,
            epsilon=1e-5,
            memory_config=q_out_mem,
            program_config=q_pc,
            compute_kernel_config=compute_config,
        )
        ttnn.rms_norm(
            kv_in,
            weight=kv_w,
            epsilon=1e-5,
            memory_config=kv_out_mem,
            program_config=kv_pc,
            compute_kernel_config=compute_config,
        )
    ttnn.synchronize_device(device)
    unfused_us = (time.perf_counter() - t0) / ITERS * 1e6

    # ---- UNFUSED same-address (no tensor cycling) ----
    q_in, q_w, kv_in, kv_w = tensor_sets[0]
    for _ in range(WARMUP):
        ttnn.rms_norm(
            q_in,
            weight=q_w,
            epsilon=1e-5,
            memory_config=q_out_mem,
            program_config=q_pc,
            compute_kernel_config=compute_config,
        )
        ttnn.rms_norm(
            kv_in,
            weight=kv_w,
            epsilon=1e-5,
            memory_config=kv_out_mem,
            program_config=kv_pc,
            compute_kernel_config=compute_config,
        )
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        ttnn.rms_norm(
            q_in,
            weight=q_w,
            epsilon=1e-5,
            memory_config=q_out_mem,
            program_config=q_pc,
            compute_kernel_config=compute_config,
        )
        ttnn.rms_norm(
            kv_in,
            weight=kv_w,
            epsilon=1e-5,
            memory_config=kv_out_mem,
            program_config=kv_pc,
            compute_kernel_config=compute_config,
        )
    ttnn.synchronize_device(device)
    unfused_same_us = (time.perf_counter() - t0) / ITERS * 1e6

    # ---- FUSED same-address ----
    q_op.input_tensors[0] = tensor_sets[0][0]
    kv_op.input_tensors[0] = tensor_sets[0][2]
    for _ in range(WARMUP):
        fused.launch()
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        fused.launch()
    ttnn.synchronize_device(device)
    fused_same_us = (time.perf_counter() - t0) / ITERS * 1e6

    print(f"\n  === Address-changing (cycle {N_TENSORS} tensor sets) ===")
    print(f"  Fused  Parallel(Q,KV):     {fused_us:.1f} us/iter")
    print(f"  Unfused ttnn.rms x2:       {unfused_us:.1f} us/iter")
    print(f"  Speedup:                   {unfused_us / fused_us:.2f}x  (delta {unfused_us - fused_us:.1f} us)")
    print(f"")
    print(f"  === Same address (steady state, no tensor swap) ===")
    print(f"  Fused  Parallel(Q,KV):     {fused_same_us:.1f} us/iter")
    print(f"  Unfused ttnn.rms x2:       {unfused_same_us:.1f} us/iter")
    print(
        f"  Speedup:                   {unfused_same_us / fused_same_us:.2f}x  (delta {unfused_same_us - fused_same_us:.1f} us)"
    )
