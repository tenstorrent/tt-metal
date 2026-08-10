# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole-galaxy port test for ttnn.experimental.llama_reduce_scatter.

The writer set fabric routes with hop counts only (fabric_set_unicast_route<false>(num_hops)),
which the BH 2D-torus routers cannot resolve - packets were never delivered and the op deadlocked
(this was the long-standing QWEN_BH_FUSED_RS_MATMUL blocker). The port emits host-computed
{mesh_id, chip_id} route info per target device via ccl_routing_utils, the same mechanism as the
BH-proven all_gather_async / fused-RMS writers. 1D fabrics keep hop counts, so WH is unchanged.

Geometry mirrors the Qwen3-32B decode FF1/FF3 reduce-scatter: 8x4 mesh, scatter over the 4 devices
of each row (cluster_axis=1), Ring topology, 24 input cores, bfp8.
"""

import os

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import (
    DECODE_FABRIC_CONFIG as _FABRIC_CONFIG,
)
from tests.ttnn.unit_tests.operations.ccl.test_llama_reduce_scatter_async_TG import run_reduce_scatter_test


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": _FABRIC_CONFIG,
            "trace_region_size": 23887872,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("trace_mode", [False, True])
@pytest.mark.parametrize("num_links", [2])
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring])
def test_bh_llama_reduce_scatter(mesh_device, trace_mode, num_links, topology):
    num_iters = int(os.environ.get("BH_RS_TEST_ITERS", "8"))
    warmup_iters = 2 if trace_mode else 0

    logger.info(f"BH llama_reduce_scatter: trace={trace_mode} links={num_links} iters={num_iters}")
    run_reduce_scatter_test(
        mesh_device,
        dim=3,
        shard_height=32,
        shard_width=160,
        num_devices_scatter=4,
        num_devices_fracture=8,
        num_cores=24,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        trace_mode=trace_mode,
        num_links=num_links,
        scheme="random",
        topology=topology,
    )
