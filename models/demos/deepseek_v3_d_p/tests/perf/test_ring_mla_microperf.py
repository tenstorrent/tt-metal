# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Steady-state per-call ring_mla micro-benchmark: loops ring_mla N times in SCALAR (host kv_actual_isl)
mode then N times in METADATA (on-device read) mode, on identical inputs, bracketed by signposts, so a
profiler run yields a clean median device-kernel time per call per mode (old vs new). Repeated kv sizes
expose the FIXED metadata-read overhead vs the size-scaled compute. Driven by ring_mla_microperf_driver.py.
"""
import pytest
import torch
import ttnn
from loguru import logger
from tracy import signpost

import tests.nightly.blackhole.sdpa.test_ring_joint_sdpa as T

import os

N_ITERS = 30  # calls per mode (driver drops a warmup prefix, takes the median of the rest)
# prior-KV tile-multiples. 256 exposes the fixed metadata-read overhead; 5120 = the real prefill chunk.
KV_SIZES = [256, 1024, 5120]
# META_L1=1 places the metadata tensor in L1 (vs DRAM) to test whether the fixed per-call overhead is
# DRAM-read-latency / bank-contention bound.
_META_MEM = ttnn.L1_MEMORY_CONFIG if os.environ.get("META_L1") == "1" else ttnn.DRAM_MEMORY_CONFIG


def _build(runtime, kv_actual_isl):
    """Mirror test_ring_mla_metadata_matches_scalar_rotation's input build for one kv_actual_isl."""
    mesh_config = T.MESH_CONFIG
    sp_size = mesh_config.sp_size
    chunk_size_local = 64
    chunk_size_global = chunk_size_local * sp_size
    new_actual_isl = chunk_size_global
    b, local_heads = 1, 4
    nhq = local_heads * mesh_config.tp_size
    nhk = 1
    d_q, d_k, d_v = 64, 64, 32
    logical_n = kv_actual_isl + new_actual_isl

    torch.manual_seed(1234)
    old_cache_kv = T.fa_rand(b, nhk, kv_actual_isl, d_k)
    new_tokens_q = T.fa_rand(b, nhq, new_actual_isl, d_q)
    new_tokens_kv = T.fa_rand(b, nhk, new_actual_isl, d_k)
    q_host, kv_host, valid_rows, _, num_cache_slabs = T.build_kv_pad_rotation_mla_inputs(
        old_cache_kv, new_tokens_q, new_tokens_kv, kv_actual_isl, sp_size, chunk_size_local
    )
    cache_seq_per_dev = num_cache_slabs * chunk_size_local

    mesh_device = runtime.mesh_device
    sp_axis, tp_axis = runtime.sp_axis, runtime.tp_axis
    q_shard_dims = [None, None]
    q_shard_dims[sp_axis] = 2
    if mesh_config.tp_size > 1:
        q_shard_dims[tp_axis] = 1
    kv_shard_dims = [None, None]
    kv_shard_dims[sp_axis] = 2
    persistent_shard_dims = [None, None]

    tt_q = ttnn.from_torch(
        q_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=q_shard_dims),
    )
    tt_kv = ttnn.from_torch(
        kv_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_shard_dims),
    )
    pbuf = ttnn.from_torch(
        torch.zeros(b, nhk, sp_size * cache_seq_per_dev, d_k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_shard_dims),
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=runtime.sdpa_compute_grid, q_chunk_size=32, k_chunk_size=32, exp_approx_mode=False
    )
    # Two 1-element uint32 tensors: slot_id (=0) and kv_actual_isl (was metadata[0]/metadata[1]).
    def _scalar_tensor(value):
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=_META_MEM, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    tt_slot_id = _scalar_tensor(0)
    tt_kv_actual_isl = _scalar_tensor(kv_actual_isl)
    return tt_q, tt_kv, pbuf, pc, tt_slot_id, tt_kv_actual_isl, logical_n, d_v


@pytest.mark.parametrize("mesh", [None], ids=["auto"])
def test_ring_mla_microperf(mesh):
    runtime = T.open_ring_joint_sdpa_runtime(T.MESH_CONFIG)
    mesh_device = runtime.mesh_device
    try:
        for kv in KV_SIZES:
            tt_q, tt_kv, pbuf, pc, tt_slot_id, tt_kv_actual_isl, logical_n, d_v = _build(runtime, kv)

            def call(use_metadata):
                ttnn.transformer.ring_mla(
                    tt_q, tt_kv, persistent_output_buffer_kv=pbuf, head_dim_v=d_v, logical_n=logical_n,
                    is_balanced=False, program_config=pc, compute_kernel_config=runtime.compute_kernel_config, dim=2,
                    multi_device_global_semaphore=runtime.ccl_semaphore_handles, num_links=runtime.num_links,
                    cluster_axis=runtime.sp_axis, mesh_device=mesh_device, topology=runtime.topology,
                    subdevice_id=runtime.worker_sub_device_id, ccl_core_grid_offset=(runtime.ccl_column, 0),
                    use_column_major_ccl=True,
                    kv_cache_batch_idx=None if use_metadata else 0,
                    kv_actual_isl=None if use_metadata else kv,
                    slot_id=tt_slot_id if use_metadata else None,
                    kv_actual_isl_tensor=tt_kv_actual_isl if use_metadata else None,
                )

            logger.info(f"[microperf] kv={kv}: {N_ITERS} scalar then {N_ITERS} metadata calls")
            signpost(header=f"SCALAR_kv{kv}_START")
            for _ in range(N_ITERS):
                call(use_metadata=False)
            ttnn.synchronize_device(mesh_device)
            signpost(header=f"SCALAR_kv{kv}_END")

            signpost(header=f"META_kv{kv}_START")
            for _ in range(N_ITERS):
                call(use_metadata=True)
            ttnn.synchronize_device(mesh_device)
            signpost(header=f"META_kv{kv}_END")
    finally:
        T.close_ring_joint_sdpa_runtime(runtime)
