# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""A/B benchmark: patchable_generic_op vs generic_op for Q/KV parallel RMS fusion.

Isolates dispatch cost by pre-allocating tensor pairs and cycling through them
in the timed loop (no from_torch inside the hot path).

Run:  python -m pytest tests/ttnn/unit_tests/operations/fused/parallel_sequential/bench_patchable_vs_generic.py -xvs
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


def _make_branches(device, seed, q_mem, kv_mem, q_cores, kv_cores, q_pc, kv_pc):
    torch.manual_seed(seed)
    q_total_w = 16 * 96
    kv_total_w = 16 * 32

    tt_q_input = ttnn.from_torch(
        torch.rand(1, 1, 32, q_total_w, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem,
    )
    tt_q_weight = ttnn.from_torch(
        torch.rand(1, 1, 1, q_total_w, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_kv_input = ttnn.from_torch(
        torch.rand(1, 1, 32, kv_total_w, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=kv_mem,
    )
    tt_kv_weight = ttnn.from_torch(
        torch.rand(1, 1, 1, kv_total_w, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    q = rms_norm(
        tt_q_input, epsilon=1e-5, weight=tt_q_weight, memory_config=q_mem, core_range_set=q_cores, program_config=q_pc
    )
    kv = rms_norm(
        tt_kv_input,
        epsilon=1e-5,
        weight=tt_kv_weight,
        memory_config=kv_mem,
        core_range_set=kv_cores,
        program_config=kv_pc,
    )
    return q, kv


def test_patchable_vs_generic(device):
    """Compare patchable_generic_op vs generic_op e2e dispatch latency."""
    WARMUP = 20
    ITERS = 500
    N_TENSORS = 8  # pre-allocated tensor pairs to cycle through

    q_cores = cores(0, 0, 3, 3)
    kv_cores = cores(5, 0, 6, 7)
    q_shard_spec = ttnn.ShardSpec(q_cores, [32, 96], ttnn.ShardOrientation.ROW_MAJOR)
    q_mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type=ttnn.BufferType.L1, shard_spec=q_shard_spec
    )
    q_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 4), subblock_w=3, block_h=1, block_w=3, inplace=False
    )
    kv_shard_spec = ttnn.ShardSpec(kv_cores, [32, 32], ttnn.ShardOrientation.ROW_MAJOR)
    kv_mem = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type=ttnn.BufferType.L1, shard_spec=kv_shard_spec
    )
    kv_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 8), subblock_w=1, block_h=1, block_w=1, inplace=False
    )

    # Pre-allocate N_TENSORS sets of input tensors (different addresses)
    tensor_sets = []
    for i in range(N_TENSORS):
        q, kv = _make_branches(
            device,
            seed=1000 + i,
            q_mem=q_mem,
            kv_mem=kv_mem,
            q_cores=q_cores,
            kv_cores=kv_cores,
            q_pc=q_pc,
            kv_pc=kv_pc,
        )
        tensor_sets.append((q, kv))

    # Build fused op from first set
    clear_build_cache()
    q0, kv0 = tensor_sets[0]
    fused = Parallel(q0, kv0).build()

    # ---- patchable_generic_op ----
    for i in range(WARMUP):
        q_t, kv_t = tensor_sets[i % N_TENSORS]
        q0.input_tensors[0] = q_t.input_tensors[0]
        kv0.input_tensors[0] = kv_t.input_tensors[0]
        fused.launch()
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for i in range(ITERS):
        q_t, kv_t = tensor_sets[i % N_TENSORS]
        q0.input_tensors[0] = q_t.input_tensors[0]
        kv0.input_tensors[0] = kv_t.input_tensors[0]
        fused.launch()
    ttnn.synchronize_device(device)
    patchable_us = (time.perf_counter() - t0) / ITERS * 1e6

    # ---- generic_op (same fused descriptor, different dispatch path) ----
    def launch_generic():
        fused.refresh_merged_io(list(fused._branch_ops))
        io = list(fused.input_tensors) + list(fused.output_tensors)
        ttnn.generic_op(io, fused.descriptor)

    for i in range(WARMUP):
        q_t, kv_t = tensor_sets[i % N_TENSORS]
        q0.input_tensors[0] = q_t.input_tensors[0]
        kv0.input_tensors[0] = kv_t.input_tensors[0]
        launch_generic()
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for i in range(ITERS):
        q_t, kv_t = tensor_sets[i % N_TENSORS]
        q0.input_tensors[0] = q_t.input_tensors[0]
        kv0.input_tensors[0] = kv_t.input_tensors[0]
        launch_generic()
    ttnn.synchronize_device(device)
    generic_us = (time.perf_counter() - t0) / ITERS * 1e6

    # ---- Same address (no tensor swap — pure dispatch overhead, no patching) ----
    for _ in range(WARMUP):
        fused.launch()
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        fused.launch()
    ttnn.synchronize_device(device)
    same_addr_us = (time.perf_counter() - t0) / ITERS * 1e6

    def launch_generic_same():
        fused.refresh_merged_io(list(fused._branch_ops))
        io = list(fused.input_tensors) + list(fused.output_tensors)
        ttnn.generic_op(io, fused.descriptor)

    for _ in range(WARMUP):
        launch_generic_same()
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        launch_generic_same()
    ttnn.synchronize_device(device)
    generic_same_us = (time.perf_counter() - t0) / ITERS * 1e6

    print(f"\n  === Address-changing (cycle {N_TENSORS} pre-allocated tensor pairs) ===")
    print(f"  patchable_generic_op: {patchable_us:.1f} us/iter")
    print(f"  generic_op:           {generic_us:.1f} us/iter")
    print(f"  delta:                {generic_us - patchable_us:.1f} us  ({generic_us / patchable_us:.2f}x)")
    print(f"")
    print(f"  === Same address (no tensor swap — pure dispatch, no patching needed) ===")
    print(f"  patchable_generic_op: {same_addr_us:.1f} us/iter")
    print(f"  generic_op:           {generic_same_us:.1f} us/iter")
    print(f"  delta:                {generic_same_us - same_addr_us:.1f} us  ({generic_same_us / same_addr_us:.2f}x)")
