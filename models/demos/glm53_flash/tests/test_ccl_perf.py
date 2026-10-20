# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collective micro-benchmark for the decode shapes on a (rows, 4) block: the generic composites used by
``tt/mesh.py`` (all_gather / reduce_scatter over the rows, all_reduce over the columns) vs the async
primitives with persistent semaphores (``ttnn.experimental.*_async`` + ``TT_CCL``) and a row-fold trick for
the non-tile-aligned row gather. Each variant is captured in a trace and replayed; wall per call is reported.

    TT_VISIBLE_DEVICES=25,24,29,28,21,20,17,16 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \
    MESH_DEVICE=P150x8 pytest models/demos/glm53_flash/tests/test_ccl_perf.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.modules.tt_ccl import get_tt_ccl
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
N_REP = 50


def _timed(mesh_device, fn, label, check=None):
    out = fn()  # eager warm-up (compile)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(N_REP):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / N_REP * 1e6
    ok = check(out) if check else True
    logger.info(f"CCL {label}: {us:.1f} us per call (traced replay){'' if ok else '  **WRONG RESULT**'}")
    ttnn.release_trace(mesh_device, tid)
    return us


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=50_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("B", [4, 32])
def test_row_collectives(mesh_device, B):
    mesh = GLMMesh.from_device(mesh_device)
    rows, H = mesh.rows, 4096
    ccl = get_tt_ccl(mesh_device)
    x = torch.randn(1, 1, rows * B, H, dtype=torch.bfloat16)
    tt_x = mesh.upload(x, ttnn.bfloat16, rows=2)  # [1,1,B,H] per row

    def gathered_ok(t):
        return all(torch.equal(ttnn.to_torch(d), x) for d in mesh.device_tensors(t))

    # --- row gather: generic composite vs async vs row-fold (gather one tile-aligned row per chip)
    _timed(
        mesh_device,
        lambda: mesh.all_gather_rows(tt_x, dim=2),
        f"B={B} all_gather_rows generic [1,1,{B},{H}] dim2",
        gathered_ok,
    )
    _timed(
        mesh_device,
        lambda: ttnn.experimental.all_gather_async(
            tt_x,
            persistent_output_buffer=None,
            dim=2,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(0),
            num_links=ccl.get_num_links(0),
            cluster_axis=0,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(0),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        ),
        f"B={B} all_gather_async dim2 (Linear)",
        gathered_ok,
    )
    if B < 32:
        # row-fold: [1,1,B,H] -> [1,1,1,B*H] (one tile-row), gather on the last dim, unfold
        def fold_gather():
            rm = ttnn.untilize(tt_x)  # [1,1,B,H] row-major
            row = ttnn.reshape(rm, [1, 1, 1, B * H])
            row_t = ttnn.tilize_with_zero_padding(row, use_multicore=True)
            g = mesh.all_gather_rows(row_t, dim=3)  # [1,1,1,rows*B*H]
            g_rm = ttnn.untilize(g)
            return ttnn.tilize_with_zero_padding(ttnn.reshape(g_rm, [1, 1, rows * B, H]), use_multicore=True)

        _timed(
            mesh_device, fold_gather, f"B={B} row-fold gather (untilize/reshape/tilize + last-dim gather)", gathered_ok
        )

    # --- reduce-scatter over the rows of the per-row partials [1,1,rows*B,H] -> [1,1,B,H]
    parts = torch.randn(rows, 1, 1, rows * B, H, dtype=torch.bfloat16)
    tt_p = ttnn.from_torch(
        parts.reshape(rows, 1, rows * B, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=mesh.shard_rows(0),
    )
    ref = parts.float().sum(0)[0, 0]

    def rs_ok(t):
        for i, d in enumerate(mesh.device_tensors(t)):
            r = i // mesh.cols
            got = ttnn.to_torch(d).float().reshape(B, H)
            if (got - ref[r * B : (r + 1) * B]).abs().max() > 0.1:
                return False
        return True

    _timed(
        mesh_device,
        lambda: mesh.reduce_scatter_rows(tt_p, dim=2),
        f"B={B} reduce_scatter_rows generic [1,1,{rows * B},{H}] dim2",
        rs_ok,
    )
    _timed(
        mesh_device,
        lambda: ttnn.experimental.reduce_scatter_minimal_async(
            tt_p,
            persistent_output_buffers=None,
            dim=2,
            multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(0),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(0),
            num_links=ccl.get_num_links(0),
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        ),
        f"B={B} reduce_scatter_minimal_async dim2 (Linear)",
        rs_ok,
    )
    # --- TP all-reduce over the columns of [1,1,B,H]
    _timed(mesh_device, lambda: mesh.all_reduce_cols(tt_x), f"B={B} all_reduce_cols generic [1,1,{B},{H}]")
    _timed(
        mesh_device,
        lambda: ttnn.experimental.all_reduce_async(
            tt_x,
            cluster_axis=1,
            mesh_device=mesh_device,
            barrier_semaphores=ccl.barrier_semaphore_handles[1],
            rs_global_semaphores=ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1),
            ag_global_semaphores=ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1),
            num_links=ccl.get_num_links(1),
            math_op=ttnn.ReduceType.Sum,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        ),
        f"B={B} all_reduce_async cols (Linear)",
    )
