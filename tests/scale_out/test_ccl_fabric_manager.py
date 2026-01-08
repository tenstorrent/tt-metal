# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Fabric Manager CCL Tests
Tests for CCL operations with fabric_manager enabled mode.
"""

import ttnn

from tests.nightly.t3000.ccl.test_all_to_all_combine import run_all_to_all_combine_test
from tests.nightly.t3000.ccl.test_all_to_all_dispatch import run_all_to_all_dispatch_test
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
from models.common.utility_functions import skip_for_blackhole


# ===========================
# All-to-All Combine Tests
# ===========================


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_manager": ttnn.FabricManagerMode.ENABLED,
        },
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_manager": ttnn.FabricManagerMode.ENABLED,
        },
    ],
    ids=[
        "fabric_manager_enabled_2d",
        "fabric_manager_enabled_1d_line",
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((8, 4), (8, 4), id="8x4_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7000])
@pytest.mark.parametrize("seq", [2], ids=["s2"])
@pytest.mark.parametrize("local_reduce", [False], ids=["dense"])
@pytest.mark.parametrize("num_iters", [2])
@pytest.mark.parametrize("num_links", [4], ids=["num_links_4"])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bfloat16"])
@pytest.mark.parametrize(
    "input_memory_config, output_memory_config, axis",
    [
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, 0),
    ],
    ids=["dram_in_l1_out_axis0"],
)
def test_all_to_all_combine_fabric_manager_8x4(
    mesh_device,
    trace_mode,
    mesh_shape,
    axis,
    batches_per_device,
    experts,
    select_experts_k,
    hidden_size,
    seq,
    local_reduce,
    num_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    """Test all-to-all combine with fabric manager enabled on 8x4 grid."""
    batch = batches_per_device * mesh_shape[axis]

    run_all_to_all_combine_test(
        mesh_device,
        mesh_shape,
        axis,
        batch,
        seq,
        local_reduce,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        num_links=num_links,
        scheme="sequential",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
    )


# ===========================
# All-to-All Dispatch Tests
# ===========================


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_manager": ttnn.FabricManagerMode.ENABLED,
        },
    ],
    ids=["fabric_manager_enabled_1d_line"],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((8, 4), (8, 4), id="8x4_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [
        (2, 5, 1),
    ],
    ids=["s2"],
)
@pytest.mark.parametrize("num_links", [4])
@pytest.mark.parametrize("topology", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "input_memory_config, output_memory_config, cluster_axis",
    [
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, 0),
    ],
    ids=["l1_in_dram_out"],
)
def test_all_to_all_dispatch_fabric_manager_8x4(
    mesh_device,
    trace_mode,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices

    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="random",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
    )


# ===========================
# All-Gather Async Tests
# ===========================


@skip_for_blackhole("This test is for wormhole")
@pytest.mark.parametrize("num_links", [3], ids=["3links"])
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, ag_input_dtype, enable_trace, num_iters",
    [
        # Check variant (without tracing)
        (8, [1, 1, 352, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),
    ],
    ids=["sd35_prompt_check"],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
    ids=["DRAM_memconfig"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_manager": ttnn.FabricManagerMode.ENABLED,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_manager_enabled_linear"],
)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
@pytest.mark.parametrize(
    "all_gather_function",
    [ttnn.experimental.all_gather_async],
    ids=["normal"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_all_gather_async_fabric_manager(
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
    all_gather_function,
):
    """Test all-gather async with fabric manager enabled."""
    if num_devices < 8:
        submesh_shape = (1, num_devices)
        cluster_axis = 1
    else:
        submesh_shape = (num_devices, 1)
        cluster_axis = 0
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape))
    run_all_gather_impl(
        submesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
        all_gather_function=all_gather_function,
    )
    ttnn.ReadDeviceProfiler(submesh_device)


# ===========================
# Reduce-Scatter Async Tests
# ===========================


@pytest.mark.parametrize("num_links", [3], ids=["3links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout, rs_input_dtype, enable_trace, num_iters",
    [
        # Perf variants (with tracing)
        (8, [8, 1, 512, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # use batching when fused
    ],
    ids=[
        "batch_8-perf",
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
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_manager": ttnn.FabricManagerMode.ENABLED,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["fabric_manager_enabled_linear"],
)
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_reduce_scatter_async_fabric_manager(
    mesh_device,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    rs_topology,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
    cluster_axis = 0
    run_reduce_scatter_impl(
        submesh_device,
        num_devices,
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
        cluster_axis=cluster_axis,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )
    ttnn.ReadDeviceProfiler(submesh_device)
