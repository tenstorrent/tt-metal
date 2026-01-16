# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal
from models.common.utility_functions import skip_for_blackhole
from tests.nightly.t3000.ccl.test_deepseek_reduce_scatter import run_reduce_scatter_impl


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("cluster_axis", [0], ids=["cluster_axis_0"])
@pytest.mark.parametrize(
    "num_links, rs_input_shape, dim, layout, rs_input_dtype, enable_trace, num_iters",
    [
        (
            1,
            [1, 1, 32, 2048],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            5,
        ),  # one_link
        (
            2,
            [1, 1, 32, 4096],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            5,
        ),  # two_links
        (
            3,
            [1, 1, 32, 6144],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            5,
        ),  # three_links
        (
            3,
            [1, 1, 32, 5120],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            5,
        ),  # three_links_partial (forward core on last link not used)
        (
            4,
            [1, 1, 32, 8192],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            5,
        ),  # four_links
        (
            4,
            [1, 1, 32, 7168],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            5,
        ),  # four_links_partial_deepseek (forward core on last link not used) (shape used in deepseek)
    ],
    ids=[
        "one_link",
        "two_links",
        "three_links",
        "three_links_partial",
        "four_links",
        "four_links_partial_deepseek",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        )
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_reduce_scatter_async(
    mesh_device,
    cluster_axis,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    enable_trace,
    num_iters,
    mem_config_input,
    mem_config_rs,
    rs_topology,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((8, 1)))
    cluster_axis = 0

    shard_shape = [1, 1, 32, 128]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
    nd_shard_spec = ttnn.NdShardSpec(
        ttnn.Shape(shard_shape), grid, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
    )
    mem_config_input = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)

    run_reduce_scatter_impl(
        submesh_device,
        submesh_device.get_num_devices(),
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        cluster_axis=cluster_axis,
        num_iters=num_iters,
        ones_tensor=False,
    )
