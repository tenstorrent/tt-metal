# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from models.common.utility_functions import skip_for_blackhole
from tests.nightly.t3000.ccl.test_deepseek_moe_reduce_scatter import run_deepseek_moe_reduce_scatter_impl


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("dtype, layout", [(ttnn.bfloat16, ttnn.TILE_LAYOUT)])
@pytest.mark.parametrize("pre_rs_reduction_dim", [(0)])
@pytest.mark.parametrize(
    "pre_rs_reduction_input_shape, sum_input_memory_config, rs_input_memory_config, rs_output_memory_config, rs_dim, rs_num_links",
    [
        (
            [8, 1, 32, 2048],
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(
                ttnn.BufferType.L1,
                ttnn.NdShardSpec(
                    ttnn.Shape([1, 1, 32, 128]),
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
                ),
            ),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            3,
            1,
        ),  # one_link
        (
            [8, 1, 32, 4096],
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(
                ttnn.BufferType.L1,
                ttnn.NdShardSpec(
                    ttnn.Shape([1, 1, 32, 128]),
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
                ),
            ),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            3,
            2,
        ),  # two_links
        (
            [8, 1, 32, 6144],
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(
                ttnn.BufferType.L1,
                ttnn.NdShardSpec(
                    ttnn.Shape([1, 1, 32, 128]),
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))]),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
                ),
            ),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            3,
            3,
        ),  # three_links
        (
            [8, 1, 32, 5120],
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(
                ttnn.BufferType.L1,
                ttnn.NdShardSpec(
                    ttnn.Shape([1, 1, 32, 128]),
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))]),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
                ),
            ),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            3,
            3,
        ),  # three_links_partial (forward core on last link not used)
        (
            [8, 1, 32, 8192],
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(
                ttnn.BufferType.L1,
                ttnn.NdShardSpec(
                    ttnn.Shape([1, 1, 32, 128]),
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))]),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
                ),
            ),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            3,
            4,
        ),  # four_links
        (
            [8, 1, 32, 7168],
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(
                ttnn.BufferType.L1,
                ttnn.NdShardSpec(
                    ttnn.Shape([1, 1, 32, 128]),
                    ttnn.CoreRangeSet(
                        [
                            ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
                            ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(2, 5)),
                            ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 0)),
                            ttnn.CoreRange(ttnn.CoreCoord(3, 5), ttnn.CoreCoord(3, 5)),
                            ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0)),
                            ttnn.CoreRange(ttnn.CoreCoord(6, 5), ttnn.CoreCoord(6, 5)),
                            ttnn.CoreRange(ttnn.CoreCoord(7, 0), ttnn.CoreCoord(7, 0)),
                            # ttnn.CoreRange(ttnn.CoreCoord(7, 5), ttnn.CoreCoord(7, 5)), # hypothetical final core
                        ]
                    ),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
                ),
            ),
            ttnn.MemoryConfig(
                ttnn.BufferType.L1,
                ttnn.NdShardSpec(
                    ttnn.Shape([1, 1, 32, 32]),
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))]),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
                ),
            ),
            3,
            4,
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
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize("enable_trace, num_iters", [(True, 10)])
def test_deepseek_moe_reduce_scatter_async(
    mesh_device,
    dtype,
    layout,
    pre_rs_reduction_dim,
    pre_rs_reduction_input_shape,
    sum_input_memory_config,
    rs_input_memory_config,
    rs_output_memory_config,
    rs_dim,
    rs_num_links,
    rs_topology,
    enable_trace,
    num_iters,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((8, 1)))
    cluster_axis = 0

    run_deepseek_moe_reduce_scatter_impl(
        mesh_device=submesh_device,
        num_devices=submesh_device.get_num_devices(),
        dtype=dtype,
        layout=layout,
        pre_rs_reduction_dim=pre_rs_reduction_dim,
        pre_rs_reduction_input_shape=pre_rs_reduction_input_shape,
        sum_input_memory_config=sum_input_memory_config,
        rs_input_memory_config=rs_input_memory_config,
        rs_output_memory_config=rs_output_memory_config,
        rs_dim=rs_dim,
        rs_num_links=rs_num_links,
        rs_topology=rs_topology,
        rs_cluster_axis=cluster_axis,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )
