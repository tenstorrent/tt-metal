# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Manual Falcon3 TP4 fused matmul/reduce-scatter closure probe."""

from __future__ import annotations

import os

import pytest

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tt.multichip_decoder import TARGET_MESH_SHAPE
from tests.ttnn.unit_tests.operations.ccl.test_new_matmul_reduce_scatter import run_reduce_scatter_impl


@pytest.mark.skipif(os.getenv("FALCON3_RUN_FUSED_MMRS") != "1", reason="manual fused MMRS probe")
@pytest.mark.parametrize(
    "projection,weight_shape",
    [
        ("o", [1, 1, 3072, 3072]),
        ("down_padded", [1, 1, 24576, 3072]),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [TARGET_MESH_SHAPE], indirect=True)
@pytest.mark.timeout(600)
def test_falcon3_tp4_fused_matmul_reduce_scatter(mesh_device, projection, weight_shape):
    """Exercise exact Falcon O/down K/N shapes, BFP4 weights, M=32, TP4, and two links.

    ``FALCON3_FUSED_MMRS_NON_FUSED=1`` retains the same persistent asynchronous
    reduce-scatter contract but emits separate matmul and CCL operations.  It is
    the like-for-like profiler control for the fused candidate.
    """
    run_reduce_scatter_impl(
        mesh_device=mesh_device,
        num_devices=4,
        rs_input_shape=[1, 1, 32, 3072],
        mm_shard_dim=2,
        rs_scatter_dim=3,
        num_links=2,
        mm_weights_shape=weight_shape,
        rs_input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        matmul_weights_dtype=ttnn.bfloat4_b,
        max_in0_block_w=8,
        use_bias=False,
        mem_config_input=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        rs_topology=ttnn.Topology.Ring,
        use_non_fused=os.getenv("FALCON3_FUSED_MMRS_NON_FUSED") == "1",
        num_iters=2 if os.getenv("FALCON3_FUSED_MMRS_TRACE") == "1" else 1,
        enable_trace=os.getenv("FALCON3_FUSED_MMRS_TRACE") == "1",
    )
