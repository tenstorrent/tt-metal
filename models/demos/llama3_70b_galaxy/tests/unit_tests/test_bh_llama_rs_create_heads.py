# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Blackhole-galaxy port test for ttnn.experimental.llama_rs_create_heads (fused QKV
reduce-scatter + nlp_create_qkv_heads_decode).

The underlying llama_reduce_scatter_create_heads writer used hop-count-only unicast routes
(fabric_set_unicast_route(num_hops)), which the BH 2D-torus routers cannot resolve - packets were
never delivered and the op deadlocked, which is why the BH decode path fell back to the unfused
column all-reduce + off-receiver reshard + nlp_create_qkv_heads_decode chain. The port emits
host-computed {mesh_id, chip_id} route info per target device via ccl_routing_utils (same
mechanism as the llama_reduce_scatter / fused-RMS ports). 1D fabrics keep hop counts, WH unchanged.

Geometry mirrors the galaxy decode QKV: 8x4 mesh, reduce-scatter over the 4 devices of each row
(cluster_axis=1), 20 input cores x 64-wide shards = 1280 = (8 q + 2*1 kv) * 128 head_dim.
"""

import os

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import (
    DECODE_FABRIC_CONFIG as _FABRIC_CONFIG,
)
from tests.ttnn.unit_tests.operations.ccl.test_llama_reduce_scatter_create_heads_async_TG import (
    run_reduce_scatter_test,
)


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
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_bh_llama_rs_create_heads(mesh_device, trace_mode, num_links, topology, dtype):
    num_iters = int(os.environ.get("BH_RS_TEST_ITERS", "8"))
    warmup_iters = 2 if trace_mode else 0

    logger.info(f"BH llama_rs_create_heads: trace={trace_mode} links={num_links} iters={num_iters}")
    run_reduce_scatter_test(
        mesh_device,
        dim=3,
        shard_height=32,
        shard_width=64,
        num_devices_scatter=4,
        num_devices_fracture=8,
        num_cores=20,
        num_iters=num_iters,
        warmup_iters=warmup_iters,
        trace_mode=trace_mode,
        num_links=num_links,
        scheme="random",
        dtype=dtype,
        topology=topology,
    )
