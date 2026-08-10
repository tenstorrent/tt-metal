# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole-galaxy port test for ttnn.experimental.all_gather_concat (fused SDPA-output
users-gather + nlp_concat_heads_decode).

The writer built raw 1D MulticastRoutingCommandHeaders, which no-op on the BH 2D-torus fabric
(HybridMeshPacketHeader routing) - the gather never delivered and the op deadlocked, which is why
the BH decode path fell back to the unfused all_gather_async + nlp_concat_heads_decode + reshard
chain. The port emits host-computed route info via ccl_routing_utils (same mechanism as the
fused-RMS / llama_reduce_scatter ports); dynamic-alternate header swapping is disabled on 2D
fabrics because routes are direction-specific. 1D fabrics keep the original behavior.

Geometry mirrors the Qwen/Llama galaxy decode: gather 8-user SDPA outputs across the 4 devices of
each mesh row into the 32-user concat-heads layout.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import (
    DECODE_FABRIC_CONFIG as _FABRIC_CONFIG,
)
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.unit_tests.operations.ccl.fusion_subtests.concat_fuse_test import run_concat_fuse_impl


_WH_ORDER_OUTPUT_GRID = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(6, 6), ttnn.CoreCoord(6, 6)),
        ttnn.CoreRange(ttnn.CoreCoord(6, 7), ttnn.CoreCoord(6, 7)),
        ttnn.CoreRange(ttnn.CoreCoord(6, 9), ttnn.CoreCoord(6, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0)),
        ttnn.CoreRange(ttnn.CoreCoord(6, 1), ttnn.CoreCoord(6, 1)),
        ttnn.CoreRange(ttnn.CoreCoord(6, 2), ttnn.CoreCoord(6, 2)),
        ttnn.CoreRange(ttnn.CoreCoord(6, 4), ttnn.CoreCoord(6, 4)),
        ttnn.CoreRange(ttnn.CoreCoord(6, 5), ttnn.CoreCoord(6, 5)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 5), ttnn.CoreCoord(5, 5)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 6), ttnn.CoreCoord(5, 6)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 7), ttnn.CoreCoord(5, 7)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 9), ttnn.CoreCoord(5, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 0)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 1), ttnn.CoreCoord(5, 1)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 2), ttnn.CoreCoord(5, 2)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 4), ttnn.CoreCoord(5, 4)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 4), ttnn.CoreCoord(1, 4)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 5), ttnn.CoreCoord(1, 5)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 9), ttnn.CoreCoord(1, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0)),
        ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
        ttnn.CoreRange(ttnn.CoreCoord(2, 4), ttnn.CoreCoord(2, 4)),
        ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(2, 5)),
        ttnn.CoreRange(ttnn.CoreCoord(2, 9), ttnn.CoreCoord(2, 9)),
    ]
)

# BH model geometry: SHARDED_ATTN_WO_INPUT_RING_MEMCFG puts the concat output on the matmul ring
# cores in ring order (PREFETCHER_NOC1_GRID_BH: cols 1-3, x fast, y slow). The first 16 ring cores
# include (1,0),(2,0),(3,0), which used to be wrongly excluded by the factory's hardcoded WH
# worker-core check (no runtime args -> TT_FATAL in override), and the concat-ready semaphore mcast
# targeted the WH cols-5-6 rectangles (cores without the concat kernel).
_BH_RING_ORDER_OUTPUT_GRID = ttnn.CoreRangeSet(
    [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for y in range(8) for x in (1, 2, 3)]
)


@torch.no_grad()
@pytest.mark.parametrize(
    "num_devices, output_shape, dim, layout, input_shard_shape, input_shard_grid, output_shard_shape, output_shard_grid, tensor_mem_layout",
    [
        # Before Concat Heads (same geometry as the WH 6U test; all grids fit the BH worker area)
        pytest.param(
            4,
            [1, 32, 32, 128],
            1,
            ttnn.ROW_MAJOR_LAYOUT,
            (32, 128),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                }
            ),
            (32, 64),
            _WH_ORDER_OUTPUT_GRID,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            id="wh_order",
        ),
        # BH model geometry: output on the matmul-ring cores in ring order (see comment above)
        pytest.param(
            4,
            [1, 32, 32, 128],
            1,
            ttnn.ROW_MAJOR_LAYOUT,
            (32, 128),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
                }
            ),
            (32, 64),
            _BH_RING_ORDER_OUTPUT_GRID,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            id="bh_ring_order",
        ),
    ],
)
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("num_iters, warmup_iters", [[8, 2]])
@pytest.mark.parametrize("trace_mode", [False, True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 23887872,
            "fabric_config": _FABRIC_CONFIG,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_bh_all_gather_concat(
    mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    num_iters,
    warmup_iters,
    function_level_defaults,
    input_shard_shape,
    input_shard_grid,
    output_shard_shape,
    output_shard_grid,
    tensor_mem_layout,
    trace_mode,
):
    logger.info(f"BH all_gather_concat: trace={trace_mode} links={num_links} iters={num_iters}")
    profiler = BenchmarkProfiler()
    run_concat_fuse_impl(
        mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        function_level_defaults,
        input_shard_shape,
        input_shard_grid,
        all_gather_topology=ttnn.Topology.Ring,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
        output_shard_shape=output_shard_shape,
        output_shard_grid=output_shard_grid,
        tensor_mem_layout=tensor_mem_layout,
        trace_mode=trace_mode,
        profiler=profiler,
    )
