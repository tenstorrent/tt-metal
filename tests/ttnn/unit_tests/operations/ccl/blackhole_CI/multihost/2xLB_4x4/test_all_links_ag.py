# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.nightly.t3000.ccl.test_minimal_reduce_scatter_async import run_reduce_scatter_impl
import ttnn

from tests.nightly.t3000.ccl.test_minimal_all_gather_async import run_all_gather_impl
from tests.nightly.t3000.ccl.test_new_all_broadcast import run_all_broadcast_impl
from models.common.utility_functions import skip_for_wormhole_b0, run_for_n_dev
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


# P300 with 2 harvested columns so 110 cores are available.
# Test utilizes 1'478'492.16 bytes per core to nearly maximize 1.5MB size
@skip_for_wormhole_b0()
@pytest.mark.parametrize("num_links", [2])  # 2 links rather than all links?
@pytest.mark.parametrize(
    "num_devices, ag_output_shape, dim, layout, all_gather_topology, ag_input_dtype",
    [
        (4, [1, 1, 256, 256], 3, ttnn.TILE_LAYOUT, ttnn.Topology.Linear, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
    ],
    ids=[
        "L1_ONLY",
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
    ],
    ids=["non-trace"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 4), id="4x4_grid")], indirect=True)
@pytest.mark.parametrize("chunks_per_sync", [20])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [2])
def test_all_links_ag(
    mesh_device,
    num_devices,
    ag_output_shape,
    cluster_axis,
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
    if cluster_axis == 0:
        print(f"Testing horizontal all-gather with {num_devices} devices and {num_links} links")
    else:
        print(f"Testing vertical all-gather with {num_devices} devices and {num_links} links")
    for i in range(mesh_device.shape[(cluster_axis - 1) % 2]):
        if cluster_axis == 0:
            print(f"Validating row {i} of {mesh_device.shape}")
        else:
            print(f"Validating column {i} of {mesh_device.shape}")
        if cluster_axis == 0:
            submesh_device = mesh_device.create_submesh(
                ttnn.MeshShape((num_devices, 1)), offset=ttnn.MeshCoordinate(0, i)
            )
        else:
            submesh_device = mesh_device.create_submesh(
                ttnn.MeshShape((1, num_devices)), offset=ttnn.MeshCoordinate(i, 0)
            )
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
            allowed_pcc=0.9999,  # equality check
            num_l1_banks=110,
        )
    ttnn.ReadDeviceProfiler(submesh_device)


@pytest.mark.parametrize("num_links", [2], ids=["2_links"])
@pytest.mark.parametrize(
    "num_devices, rs_input_shape, dim, layout",
    [
        (4, [1, 1, 128, 256], 3, ttnn.TILE_LAYOUT),
    ],
    ids=["4_device"],
)
@pytest.mark.parametrize(
    "rs_input_dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "float_16",
    ],
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
    ],
    ids=["dram_only"],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 3),
    ],
    ids=[
        "trace",
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=[
        "fabric_linear",
    ],
)
@pytest.mark.parametrize("chunks_per_sync", [2])
@pytest.mark.parametrize("num_workers_per_link", [2])
@pytest.mark.parametrize("num_buffers_per_channel", [8])
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 4), id="4x4_grid")], indirect=True)
def test_rs(
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
    cluster_axis,
    num_buffers_per_channel,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))
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


@pytest.mark.parametrize("num_links", [2], ids=["2_links"])
@pytest.mark.parametrize(
    "num_devices, output_shape, layout",
    [
        (4, [1, 1, 128, 128], ttnn.TILE_LAYOUT),
    ],
    ids=["4_device"],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
    ids=[
        "bfloat16",
    ],
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize(
    "mem_config_input, mem_config_output",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ),
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ),
    ],
    ids=[
        "L1_to_DRAM",
        "L1_to_L1",
        "DRAM_to_L1",
        "DRAM_to_DRAM",
    ],
)
@pytest.mark.parametrize(
    "num_iters",
    [
        3,
    ],
    ids=["3_iters"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}),
    ],
    indirect=["device_params"],
    ids=["fabric"],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 4), id="4x4_grid")], indirect=True)
def test_all_broadcast(
    mesh_device,
    num_devices,
    num_links,
    output_shape,
    layout,
    input_dtype,
    mem_config_input,
    mem_config_output,
    num_iters,
    cluster_axis,
    function_level_defaults,
):
    from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
    from loguru import logger
    import torch

    if cluster_axis == 0:
        print(f"Testing horizontal all-broadcast with {num_devices} devices and {num_links} links")
    else:
        print(f"Testing vertical all-broadcast with {num_devices} devices and {num_links} links")

    for i in range(mesh_device.shape[(cluster_axis - 1) % 2]):
        if cluster_axis == 0:
            print(f"Validating row {i} of {mesh_device.shape}")
        else:
            print(f"Validating column {i} of {mesh_device.shape}")

        if cluster_axis == 0:
            submesh_device = mesh_device.create_submesh(
                ttnn.MeshShape((num_devices, 1)), offset=ttnn.MeshCoordinate(0, i)
            )
        else:
            submesh_device = mesh_device.create_submesh(
                ttnn.MeshShape((1, num_devices)), offset=ttnn.MeshCoordinate(i, 0)
            )

        # Setup sub-devices for CCL
        compute_grid_size = submesh_device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_stall_group = [worker_sub_device_id]
        submesh_device.set_sub_device_stall_group(sub_device_stall_group)

        mesh_mapper_config = ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)], ttnn.MeshShape(1, num_devices)
        )

        torch.manual_seed(0)

        for iter_idx in range(num_iters):
            # Create input tensors - one per device
            output_tensors = []
            for k in range(num_devices):
                output_tensor = torch.rand(output_shape).bfloat16()
                output_tensors.append(output_tensor)

            # Concatenate for mesh input
            temp_output_tensor = torch.cat(output_tensors, -1)

            # Create mesh tensor with input memory config
            input_tensor_mesh = ttnn.from_torch(
                temp_output_tensor,
                device=submesh_device,
                layout=layout,
                dtype=input_dtype,
                memory_config=mem_config_input,
                mesh_mapper=ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config),
            )

            # Perform all-broadcast with output memory config
            tt_out_tensors = ttnn.all_broadcast(
                input_tensor_mesh,
                num_links=num_links,
                memory_config=mem_config_output,
                cluster_axis=cluster_axis,
                topology=ttnn.Topology.Linear,
                subdevice_id=worker_sub_device_id,
            )

            # Synchronize
            ttnn.synchronize_device(submesh_device, sub_device_ids=sub_device_stall_group)

            # Validate results - each device should have received all tensors
            passed = True
            for k in range(num_devices):
                expected_tensor = output_tensors[k]
                for dev_idx, t in enumerate(ttnn.get_device_tensors(tt_out_tensors[k])):
                    tt_output_tensor = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=-1))
                    eq, output = comp_equal(tt_output_tensor, expected_tensor)
                    if not eq:
                        logger.error(f"Output mismatch for tensor {k} on device {dev_idx}, iteration {iter_idx}")
                        passed = False
                        assert eq, f"FAILED: {output}"

        submesh_device.reset_sub_device_stall_group()

    ttnn.ReadDeviceProfiler(submesh_device)
