# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

# from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from tests.ttnn.unit_tests.operations.ccl.test_new_all_gather_matmul import run_all_gather_impl


@pytest.mark.parametrize("num_links", [3], ids=["3links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype",
    [
        (8, [1, 1, 1024, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        # (8, [1, 1, 352, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
    # ids=[
    #     "sd35_spatial",
    #     "sd35_prompt",
    # ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        # (True, 10),
        (False, 1),
    ],
    # ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
@pytest.mark.parametrize("chunks_per_sync", [None])
@pytest.mark.parametrize("num_workers_per_link", [None])
@pytest.mark.parametrize("num_buffers_per_channel", [None])
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_all_gather_async(
    mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    all_gather_topology,
    num_iters,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
    run_all_gather_impl(
        submesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        960,
        ttnn.bfloat16,
        2,
        True,
        mem_config_input,
        mem_config_ag,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        all_gather_topology=all_gather_topology,
        use_non_fused=True,
        use_legacy_allgather=False,
        enable_trace=enable_trace,
        num_iters=num_iters,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


# t3k_mesh_device,
#         num_devices,
#         ag_output_shape,
#         dim,
#         num_links,
#         ag_input_dtype,
#         layout,
#         matmul_output_dim,
#         matmul_weights_dtype,
#         max_in0_block_w,
#         use_bias,
#         mem_config_input,
#         mem_config_ag,
#         mem_config_mm,
#         all_gather_topology=all_gather_topology,
#         enable_trace=enable_trace,
#         use_non_fused=use_non_fused,
#         use_legacy_allgather=use_legacy_allgather,
#         num_iters=num_iters,
#         use_barrier=use_barrier,
#         chunks_per_sync=chunks_per_sync,
#         num_workers_per_link=num_workers_per_link,
#         num_buffers_per_channel=num_buffers_per_channel,
