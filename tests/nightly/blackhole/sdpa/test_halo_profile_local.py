# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LOCAL ONLY: device-profile the Gemma4 sliding halo (TT_METAL_DEVICE_PROFILER=1, probe zones in the kernels).

HP_B=1 runs one single-request call; HP_B>1 one lanes call of HP_B requests. Model layout: row-sharded K/V cache
and halo buffers. 1 compile call, then HP_CALLS profiled calls.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.operations.ccl import Topology
from models.demos.gemma4_d_p.tt.attention.ring_prefill import migration_ring_memory_config
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    _ring_joint_lane_metadata_tensor,
    call_sdpa,
    close_ring_joint_sdpa_runtime,
    fa_rand,
    open_ring_joint_sdpa_runtime,
)

LOCAL, WINDOW, HQ, HKV, D, QC, KC = 256, 1024, 8, 4, 256, 128, 128
B = int(os.environ.get("HP_B", "1"))
CALLS = int(os.environ.get("HP_CALLS", "3"))
KS = int(os.environ.get("HP_KS", "3" if B == 1 else "1"))
PREFIX_CHUNKS = 2


def test_halo_profile():
    sp, tp = MESH_CONFIG.sp_size, MESH_CONFIG.tp_size
    nhq, nhk = HQ * tp, HKV * tp
    chunk_global = LOCAL * sp
    capacity = sp * (PREFIX_CHUNKS + 2) * LOCAL
    halo_tokens = -(-(WINDOW - 1) // KC) * KC
    from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import RING_JOINT_TRACE_REGION_SIZE

    runtime = open_ring_joint_sdpa_runtime(
        MESH_CONFIG, topology=Topology.Linear, trace_region_size=RING_JOINT_TRACE_REGION_SIZE
    )
    mesh = runtime.mesh_device
    try:
        nd = migration_ring_memory_config(mesh, D) if os.environ.get("HP_ND", "1") == "1" else ttnn.DRAM_MEMORY_CONFIG
        in_dims = [None, None]
        in_dims[runtime.sp_axis], in_dims[runtime.tp_axis] = 2, 1
        halo_dims = [None, None]
        halo_dims[runtime.tp_axis] = 1

        def upload(host, dtype, dims, mc):
            return ttnn.from_torch(
                host,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=mc,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
            )

        k = upload(torch.randn(B, nhk, capacity, D), ttnn.bfloat8_b, in_dims, nd)
        v = upload(torch.randn(B, nhk, capacity, D), ttnn.bfloat8_b, in_dims, nd)
        lane_q = [fa_rand(1, nhq, chunk_global, D) for _ in range(B)]
        q_host = (
            torch.stack(lane_q, dim=2)
            .reshape(1, nhq, B, sp, LOCAL, D)
            .transpose(2, 3)
            .reshape(1, nhq, sp * B * LOCAL, D)
        )
        q = upload(q_host, ttnn.bfloat16, in_dims, ttnn.DRAM_MEMORY_CONFIG)
        halos = [upload(torch.zeros(B, nhk, halo_tokens, D), ttnn.bfloat8_b, halo_dims, nd) for _ in range(2)]
        meta = [_ring_joint_lane_metadata_tensor(mesh, list(x)) for x in (range(B), [PREFIX_CHUNKS * chunk_global] * B)]
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=runtime.sdpa_compute_grid,
            q_chunk_size=QC,
            k_chunk_size=KC,
            exp_approx_mode=False,
            k_splits=KS,
        )

        def run():
            out = call_sdpa(
                q,
                k,
                v,
                capacity,
                is_causal=True,
                is_balanced=False,
                p_buf_k=halos[0],
                p_buf_v=halos[1],
                program_config=cfg,
                compute_kernel_config=runtime.compute_kernel_config,
                ccl_semaphore_handles=runtime.ccl_semaphore_handles,
                num_links=runtime.num_links,
                sp_axis=runtime.sp_axis,
                mesh_device=mesh,
                topology=runtime.topology,
                worker_sub_device_id=runtime.worker_sub_device_id,
                ccl_column=runtime.ccl_column,
                sliding_window_size=WINDOW,
                slot_id=meta[0],
                kv_actual_isl_tensor=meta[1],
            )
            ttnn.deallocate(out)

        run()
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        if os.environ.get("HP_TRACE") == "1":
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            for _ in range(CALLS):
                run()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            ttnn.release_trace(mesh, tid)
        else:
            for _ in range(CALLS):
                run()
                ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
        logger.info(f"[halo_profile] B={B} ks={KS} calls={CALLS} done")
    finally:
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)
