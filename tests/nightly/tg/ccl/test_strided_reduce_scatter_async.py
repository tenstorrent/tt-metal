# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from dataclasses import astuple

from tests.nightly.t3000.ccl.test_strided_reduce_scatter_async import (
    ReduceScatterTestConfig,
    run_reduce_scatter_impl,
)


def _make_fabric_router_config(max_packet_payload_size_bytes):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_packet_payload_size_bytes
    return config


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("num_links", [2], ids=["2link"])
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["axis_0", "axis_1"])
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[4, 1, 128, 1024],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=1,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_full_block_wt=4,
                chunk_width_in_mm_blocks=2,
            ),
            id="strided_toy_4x4",
            marks=pytest.mark.skip(reason="run manually"),
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[4, 1, 128, 1024],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=2,
                mm_block_wt=2,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="strided_4x4_2core_grid",
            marks=pytest.mark.skip(reason="run manually"),
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[4, 1, 512, 2048],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=8,
                mm_block_wt=2,
                mm_N_full_block_wt=8,
                chunk_width_in_mm_blocks=2,
            ),
            id="strided_asymmetric_tall_narrow_8x2_blocks",
            marks=pytest.mark.skip(reason="run manually"),
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[4, 1, 512, 2560],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=4,
                mm_block_wt=4,
                mm_N_full_block_wt=10,
                chunk_width_in_mm_blocks=2,
            ),
            id="strided_partial_last_chunk",
            marks=pytest.mark.skip(reason="run manually"),
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[2, 1, 9472, 5120],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=7,
                mm_block_ht=8,
                mm_block_wt=8,
                mm_N_full_block_wt=20,
                chunk_width_in_mm_blocks=1,
            ),
            id="strided_large_non_div_Ht",
            marks=pytest.mark.skip(reason="run manually"),
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[8, 1, 128, 1792],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=1,
                mm_block_ht=4,
                mm_block_wt=3,
                mm_N_full_block_wt=9,
                chunk_width_in_mm_blocks=1,
            ),
            id="strided_non_div_Wt_cross_col",
            marks=pytest.mark.skip(reason="run manually"),
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[8, 1, 224, 768],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=2,
                mm_block_ht=3,
                mm_block_wt=1,
                mm_N_full_block_wt=2,
                chunk_width_in_mm_blocks=1,
            ),
            id="strided_non_div_Wt_and_Ht",
            marks=pytest.mark.skip(reason="run manually"),
        ),
        pytest.param(
            ReduceScatterTestConfig(
                rs_input_shape=[4, 1, 4096, 4096],
                dim=3,
                layout=ttnn.TILE_LAYOUT,
                rs_input_dtype=ttnn.bfloat16,
                use_new=False,
                enable_trace=False,
                num_iters=1,
                use_barrier=True,
                use_persistent_buffers=True,
                use_strided=True,
                verify_output_shape=True,
                verify_output_pcc=True,
                small_random_ints=True,
                mm_cores_y=8,
                mm_block_ht=4,
                mm_block_wt=4,
                mm_N_full_block_wt=8,
                chunk_width_in_mm_blocks=1,
                num_workers_per_link=6,
            ),
            id="strided_large_8_cores_6_workers",
        ),
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
    ids=["DRAM_memconfig"],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring),
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": _make_fabric_router_config(8192),
                "trace_region_size": 1531456,
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_ring_8kib_payload"],
)
def test_strided_reduce_scatter_async(
    mesh_device,
    test_config,
    num_links,
    mem_config_input,
    mem_config_rs,
    topology,
    cluster_axis,
):
    num_devices = mesh_device.shape[cluster_axis]
    if num_devices == 1:
        pytest.skip(f"cluster_axis={cluster_axis} has only 1 device, reduce-scatter ring size must be > 1")

    if cluster_axis == 0:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(num_devices, 1))
    else:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(1, num_devices))

    (
        rs_input_shape,
        dim,
        layout,
        rs_input_dtype,
        use_new,
        enable_trace,
        num_iters,
        use_barrier,
        use_persistent_buffers,
        use_strided,
        verify_output_shape,
        verify_output_pcc,
        small_random_ints,
        mm_cores_y,
        mm_block_ht,
        mm_block_wt,
        mm_N_full_block_wt,
        chunk_width_in_mm_blocks,
        num_workers_per_link,
        num_buffers_per_channel,
    ) = astuple(test_config)

    run_reduce_scatter_impl(
        submesh,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        rs_topology=topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        small_random_ints=small_random_ints,
        use_barrier=use_barrier,
        use_persistent_buffers=use_persistent_buffers,
        use_strided=use_strided,
        verify_output_shape=verify_output_shape,
        verify_output_pcc=verify_output_pcc,
        mm_cores_y=mm_cores_y,
        mm_block_ht=mm_block_ht,
        mm_block_wt=mm_block_wt,
        mm_N_full_block_wt=mm_N_full_block_wt,
        chunk_width_in_mm_blocks=chunk_width_in_mm_blocks,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
        cluster_axis=cluster_axis,
    )
