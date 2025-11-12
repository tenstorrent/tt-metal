#  SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
#  SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("use_new", [False])
@pytest.mark.parametrize("enable_trace", [True])
@pytest.mark.parametrize(
    "rs_input_shape, dim, layout, rs_input_dtype, mem_config_input, mem_config_rs, num_iters",
    [
        ([1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 10),
        ([1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 10),
        # ([1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
        # ([1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
        # ([1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
        # ([1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
        # ([1, 1, 32, 1536], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
        ([1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, 10),
        # ([1, 1, 32, 7168], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 10),  # duplicate
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology, cluster_axis",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1171456}, ttnn.Topology.Linear, 1),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_reduce_scatter_async_training_shapes(
    mesh_device,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    use_new,
    enable_trace,
    num_iters,
    mem_config_input,
    mem_config_rs,
    rs_topology,
    cluster_axis,
):
    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        ones_tensor=False,
        use_barrier=True,
        use_persistent_buffers=False,
        cluster_axis=cluster_axis,
        use_new=use_new,
    )
