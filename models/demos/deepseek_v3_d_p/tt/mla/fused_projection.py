# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in MMRS integration for the measured LoudBox chunked-prefill geometry."""

import math
import os

import ttnn


def fused_projection(owner, name, x, weight, cfg, dim=3, rs_compute=None):
    if os.getenv("DS_MLA_FUSED_MMRS", "0") != "1":
        return None
    if (
        tuple(owner.mesh_device.shape) != (2, 4)
        or owner.tp_axis != 1
        or tuple(x.shape)[:3] != (1, 1, 640)
        or cfg is None
    ):
        return None
    n = weight.shape[-1]
    if (name, x.shape[-1], n) not in {
        ("o_proj", 4096, 6144),
        ("q_a_proj", 1536, 2048),
        ("indexer.wk", 1536, 128),
        ("indexer.weights_proj", 1536, 32),
    }:
        return None
    cache = getattr(owner, "_fused_projection_buffers", None)
    if cache is None:
        cache = owner._fused_projection_buffers = {}
    key = (name, owner.tp_ccl_topology)
    if key not in cache:
        pc = cfg["program_config"]
        pm, pn = (2, 1) if n <= 128 else (pc.per_core_M, pc.per_core_N)
        gx, gy = math.ceil(n / 32 / pn), math.ceil(20 / pm)
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=8 if n <= 128 else pc.in0_block_w,
            per_core_M=pm,
            per_core_N=pn,
            out_block_h=2 if n <= 128 else pc.out_block_h,
            out_block_w=1 if n <= 128 else pc.out_block_w,
            out_subblock_h=2 if n <= 128 else pc.out_subblock_h,
            out_subblock_w=1 if n <= 128 else pc.out_subblock_w,
            transpose_mcast=False,
            fuse_batch=False,
            allowed_worker_cores=cores,
        )
        scratch = ttnn.empty(
            (2 if owner.tp_ccl_topology == ttnn.Topology.Linear else 1, 1, 640, n),
            device=owner.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ccl_y = 2 if name == "o_proj" and owner.tp_ccl_topology == ttnn.Topology.Linear else 0
        cache[key] = (config, scratch, ttnn.CoreCoord(gx, ccl_y))
    config, scratch, offset = cache[key]
    # Output ownership stays with the existing caller; scratch is persistent across forwards.
    output = ttnn.empty(
        (1, 1, 160, n) if dim == 2 else (1, 1, 640, n // 4),
        device=owner.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mm, reduced = ttnn.experimental.matmul_reduce_scatter_async(
        x,
        weight,
        scratch,
        output,
        dim,
        owner.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=owner.tp_axis),
        offset,
        barrier_semaphore=owner.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=owner.tp_axis),
        num_links=owner.ccl_num_links,
        cluster_axis=owner.tp_axis,
        num_workers_per_link=1,
        topology=owner.tp_ccl_topology,
        program_config=config,
        memory_config_mm=ttnn.L1_MEMORY_CONFIG,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=owner.default_compute_kernel_config,
        rs_compute_kernel_config=rs_compute,
        chunks_per_sync=6 if name == "o_proj" and owner.tp_ccl_topology == ttnn.Topology.Linear else None,
        num_buffers_per_channel=4 if name == "o_proj" and owner.tp_ccl_topology == ttnn.Topology.Linear else None,
    )
    ttnn.deallocate(mm)
    return reduced
