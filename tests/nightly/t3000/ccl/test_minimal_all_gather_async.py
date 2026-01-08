# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.common.utility_functions import skip_for_blackhole

from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from tracy import signpost


def is_unsupported_case(
    input_shape,
    dim,
    mem_config,
    num_devices,
    num_links,
    input_dtype,
    layout,
    tile,
    num_l1_banks=64,
    mem_config_input=None,
):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype == ttnn.bfloat8_b:
        return True, "Invalid combination"

    if input_shape[dim] % num_devices != 0:
        return True, "Unsupported test case"
    if tile != (32, 32) and input_dtype != ttnn.bfloat16:
        return True, "Tiny tile only supports bfloat16"

    ## Check that we can readback results
    fast_dispatch_page_size_limit = 55 * 1024
    elem_size_map = {
        ttnn.uint32: 4,
        ttnn.bfloat16: 2,
        ttnn.bfloat8_b: 1,
    }
    elem_size = elem_size_map.get(input_dtype, 4)
    if layout == ttnn.ROW_MAJOR_LAYOUT and (input_shape[dim] * elem_size) > fast_dispatch_page_size_limit:
        # Fast dispatch currently can't breakup readback of large pages into multiple smaller pages and is
        # limited to ~55K pages.
        return True, "Fast dispatch can't support reading back this page size in one shot"

    # Check that we can fit in L1 (if L1 config)
    tensor_size_bytes = elem_size
    for i in input_shape:
        tensor_size_bytes *= i
    L1_util = 0
    if mem_config.buffer_type == ttnn.BufferType.L1:
        L1_util = L1_util + tensor_size_bytes
    if mem_config_input is not None:
        if mem_config_input.buffer_type == ttnn.BufferType.L1:
            L1_util += tensor_size_bytes / num_devices

    if L1_util > num_l1_banks * 1536 * 1024:
        return True, "Test_Infrastructure_Skip L1 test requires more memory than the total available in the device"

    # Check that each chip has a non-zero amount of data available
    if input_shape[dim] < num_devices:
        return (
            True,
            f"Input shape {input_shape} incompatible with {num_devices} on dim {dim} because some chips will have no tensor",
        )

    if (
        input_shape == [8, 8, 256, 384]
        and dim == 1
        and layout == ttnn.TILE_LAYOUT
        and (input_dtype == ttnn.bfloat8_b or tile != (32, 32))
    ):
        return True, "Known failure"

    return False, ""


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def run_all_gather_impl(
    mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    all_gather_topology,
    num_iters=1,
    enable_trace=True,
    cluster_axis=None,
    use_barrier=False,
    use_persistent_buffers=True,
    chunks_per_sync=None,
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    allowed_pcc=1,
    skip_check=False,
    num_l1_banks=64,
    all_gather_function=ttnn.experimental.all_gather_async,
    use_semaphore_free_all_gather_impl=False,
    sub_core_grids=None,
):
    use_sub_devices = False
    torch.manual_seed(0)

    tile = (32, 32)

    # Skip unsupported cases
    (is_known_failure, message) = is_unsupported_case(
        ag_output_shape,
        dim,
        mem_config_ag,
        num_devices,
        num_links,
        ag_input_dtype,
        layout,
        tile,
        num_l1_banks,
        mem_config_input,
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    ##### All gather setup #####
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    if use_sub_devices:
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphores(mesh_device, num_devices, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    barrier_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    if use_persistent_buffers:
        if enable_trace:
            persistent_output_buffers = [
                ttnn.from_torch(
                    torch.zeros(ag_output_shape),
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ag_input_dtype,
                    memory_config=mem_config_ag,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            ]
        else:
            persistent_output_buffers = [
                ttnn.from_torch(
                    torch.zeros(ag_output_shape),
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ag_input_dtype,
                    memory_config=mem_config_ag,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                for _ in range(num_iters)
            ]
    else:
        persistent_output_buffers = []

    logger.info("Done creating persistent buffers")

    ##### All gather input setup #####
    logger.info(f"All gather output shape: {ag_output_shape}")
    logger.info(f"All gather dim: {dim}")

    input_tensor_mesh_list = []
    ag_output_tensor_goldens_list = []
    _, _, _, hidden_dim = ag_output_shape

    for i in range(num_iters):
        ag_output_tensor = torch.rand(ag_output_shape).bfloat16()
        ag_output_tensor_goldens_list.append(ag_output_tensor)

        input_tensor_mesh = ttnn.from_torch(
            ag_output_tensor,
            device=mesh_device,
            layout=layout,
            dtype=ag_input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        )

        input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform the TT ops #####
    tt_all_gather_out_tensor_list = []

    def run_op(i):  # absolutely disgusting if-else condition because changing every call site is a humongous PITA
        if use_semaphore_free_all_gather_impl and all_gather_function == ttnn.experimental.all_gather_async:
            logger.info(f"Using new all-gather")
            tt_all_gather_out_tensor = ttnn.all_gather(
                input_tensor_mesh_list[i],
                dim=dim,
                cluster_axis=cluster_axis,
                num_links=num_links,
                memory_config=mem_config_ag,
                topology=all_gather_topology,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
                subdevice_id=worker_sub_device_id,
                sub_core_grids=sub_core_grids,
            )
        else:
            logger.info(f"Using experimental all-gather")
            tt_all_gather_out_tensor = all_gather_function(
                input_tensor_mesh_list[i],
                persistent_output_buffer=persistent_output_buffers[i] if use_persistent_buffers else None,
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                memory_config=mem_config_ag,
                topology=all_gather_topology,
                subdevice_id=worker_sub_device_id,
                barrier_semaphore=barrier_semaphore_handles[i] if use_barrier else None,
                cluster_axis=cluster_axis,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
                sub_core_grids=sub_core_grids,
            )

        return tt_all_gather_out_tensor

    if enable_trace:
        # Compile the op
        tt_all_gather_out_tensor = run_op(0)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_all_gather_out_tensor = run_op(0)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done capturing trace")

        # Execute trace
        signpost("start")
        for i in range(num_iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)
        logger.info(f"Done executing trace")
        signpost("stop")
    else:
        for i in range(num_iters):
            tt_all_gather_out_tensor = run_op(i)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    if not skip_check:
        for i in range(num_iters):
            tt_ag_out_tensor = tt_all_gather_out_tensor_list[i]
            torch_ag_out_tensor = ag_output_tensor_goldens_list[i if not enable_trace else 0]

            # Create expected output tensor based on which function is used
            is_reversed = all_gather_function == ttnn.experimental.all_gather_async_reversed
            if is_reversed:
                # For reversed all-gather, we need to reverse the order along the gather dimension
                expected_tensor = torch_ag_out_tensor.clone()
                shard_size = torch_ag_out_tensor.shape[dim] // num_devices

                # Reverse the shards along the gather dimension
                for device_id in range(num_devices):
                    src_start = device_id * shard_size
                    src_end = (device_id + 1) * shard_size
                    dst_start = (num_devices - 1 - device_id) * shard_size
                    dst_end = (num_devices - device_id) * shard_size

                    if dim == 0:
                        expected_tensor[dst_start:dst_end] = torch_ag_out_tensor[src_start:src_end]
                    elif dim == 1:
                        expected_tensor[:, dst_start:dst_end] = torch_ag_out_tensor[:, src_start:src_end]
                    elif dim == 2:
                        print(f"dst_start: {dst_start}, dst_end: {dst_end}, src_start: {src_start}, src_end: {src_end}")
                        expected_tensor[:, :, dst_start:dst_end] = torch_ag_out_tensor[:, :, src_start:src_end]
                    elif dim == 3:
                        expected_tensor[:, :, :, dst_start:dst_end] = torch_ag_out_tensor[:, :, :, src_start:src_end]
                    else:
                        raise NotImplementedError(f"Reverse all-gather not implemented for dim {dim}")
            else:
                expected_tensor = torch_ag_out_tensor

            tt_ag_out = ttnn.from_device(tt_ag_out_tensor)
            tt_ag_out = ttnn.to_torch(tt_ag_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
            tt_ag_out = tt_ag_out[:, :, :, 0 : expected_tensor.shape[3]]
            eq, output = comp_pcc(tt_ag_out, expected_tensor, allowed_pcc)
            logger.info(f"{output}, iteration {i}, reversed={is_reversed}")
            assert eq, f"{i} FAILED ag: {output}"

    mesh_device.reset_sub_device_stall_group()
    if use_sub_devices:
        mesh_device.clear_loaded_sub_device_manager()


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype, enable_trace, num_iters, use_barrier, use_persistent_buffers, pcc_threshold",
    [
        (
            [1, 1, 1024, 5120],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            10,
            True,
            True,
            1.0,
        ),  # perf, barrier_with_persistent
        (
            [8, 1, 512, 512],
            0,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            1,
            True,
            False,
            1.0,
        ),  # check, barrier_without_persistent
        (
            [1, 1, 1024, 1024],
            2,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            10,
            False,
            True,
            1.0,
        ),  # perf, no_barrier_with_persistent
        (
            [1, 1, 48, 1024],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            1,
            True,
            True,
            1.0,
        ),  # check, barrier_with_persistent
        # Composite-AG tests
        (
            [1, 1, 1, 8],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            10,
            True,
            False,
            1.0,
        ),  # perf, barrier_without_persistent
        (
            [1, 16, 32, 32],
            1,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            False,
            1,
            False,
            True,
            1.0,
        ),  # check, no_barrier_with_persistent
        (
            [1, 1, 1024, 5120],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat8_b,
            False,
            1,
            True,
            True,
            0.9999,
        ),  # perf, barrier_with_persistent
    ],
    ids=[
        "sd35_spatial-perf-barrier_with_persistent",
        "gather_dim_0-check-barrier_without_persistent",
        "gather_dim_2-perf-no_barrier_with_persistent",
        "gather_dim_3_padded_dim_2-check-barrier_with_persistent",
        "composite_ag_test_two-perf-barrier_without_persistent",
        "composite_ag_test_four-check-no_barrier_with_persistent",
        "sd35_spatial-perf-barrier_with_persistent_bfloat8_b",
    ],
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
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_async(
    mesh_device,
    num_links,
    ag_output_shape,
    dim,
    layout,
    ag_input_dtype,
    enable_trace,
    num_iters,
    use_barrier,
    use_persistent_buffers,
    mem_config_input,
    mem_config_ag,
    all_gather_topology,
    pcc_threshold,
):
    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        use_barrier=use_barrier,
        use_persistent_buffers=use_persistent_buffers,
        use_semaphore_free_all_gather_impl=False,
        allowed_pcc=pcc_threshold,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype, enable_trace, num_iters, chunks_per_sync, num_workers_per_link, num_buffers_per_channel,",
    [
        ([1, 1, 3072, 8192], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10, None, None, None),  # perf
        ([1, 1, 352, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, 2, 2, 8),  # check
        ([1, 8, 512, 512], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10, None, None, None),  # perf
        ([1, 1, 512, 48], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, 2, 2, 8),  # check
        # Composite-AG tests
        ([1, 1, 17, 64], 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, True, 10, None, None, None),  # perf
        ([1, 1, 64, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, None, None, None),  # check
    ],
    ids=[
        "dit_shape-perf",  # this one triggers the default chunks_per_sync
        "sd35_prompt-check",
        "gather_dim_1-perf",
        "gather_dim_2_padded_dim_3-check",
        "composite_ag_test_one-perf",
        "composite_ag_test_three-check",
    ],
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
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_ttnn_all_gather(
    mesh_device,
    num_links,
    ag_output_shape,
    dim,
    layout,
    ag_input_dtype,
    enable_trace,
    num_iters,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
    mem_config_input,
    mem_config_ag,
    all_gather_topology,
):
    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
        use_semaphore_free_all_gather_impl=True,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype, enable_trace, num_iters",
    [
        ([1, 1, 3072, 8192], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # perf
        ([1, 1, 352, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # check
        ([1, 8, 512, 512], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # perf
        ([1, 1, 512, 48], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # check
    ],
    ids=[
        "dit_shape-perf",  # this one triggers the default chunks_per_sync
        "sd35_prompt-check",
        "gather_dim_1-perf",
        "gather_dim_2_padded_dim_3-check",
    ],
)
@pytest.mark.parametrize(
    "sub_core_grids",
    (
        # multiple disjoint cores
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            ]
        ),
    ),
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
@pytest.mark.parametrize("use_new_allgather", [True, False])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_subgrid(
    mesh_device,
    num_links,
    ag_output_shape,
    dim,
    layout,
    ag_input_dtype,
    enable_trace,
    num_iters,
    mem_config_input,
    mem_config_ag,
    all_gather_topology,
    sub_core_grids,
    use_new_allgather,
):
    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        use_semaphore_free_all_gather_impl=use_new_allgather,
        sub_core_grids=sub_core_grids,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype, use_semaphore_free_all_gather_impl, enable_trace, num_iters",
    [
        # Gather on dim 0
        ([24, 3, 128, 96], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
        ([16, 1, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, False, 1),  # check
        ([16, 16, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
        ([8, 16, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, False, 1),  # check
        # Gather on dim 1
        ([3, 24, 128, 96], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
        ([1, 16, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, False, 1),  # check
        ([16, 16, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
        ([16, 8, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, False, 1),  # check
        # Gather on dim 2
        ([1, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
        ([16, 1, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, False, 1),  # check
        ([16, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
        # # Gather on dim 3
        ([1, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, False, 1),  # check
        ([16, 1, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, True, 10),  # perf
        ([16, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, False, 1),  # check
    ],
    ids=[
        "tt_training_test_one-perf",
        "tt_training_test_two-check",
        "tt_training_test_three-perf",
        "tt_training_test_four-check",
        "tt_training_test_five-perf",
        "tt_training_test_six-check",
        "tt_training_test_seven-perf",
        "tt_training_test_eight-check",
        "tt_training_test_nine-perf",
        "tt_training_test_ten-check",
        "tt_training_test_eleven-perf",
        "tt_training_test_twelve-check",
        "tt_training_test_thirteen-perf",
        "tt_training_test_fourteen-check",
    ],
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
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_async_training_shapes(
    mesh_device,
    num_links,
    ag_output_shape,
    dim,
    layout,
    ag_input_dtype,
    use_semaphore_free_all_gather_impl,
    enable_trace,
    num_iters,
    mem_config_input,
    mem_config_ag,
    all_gather_topology,
):
    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        use_barrier=True,
        use_persistent_buffers=False,
        use_semaphore_free_all_gather_impl=use_semaphore_free_all_gather_impl,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "num_links, layout, ag_input_dtype",
    [
        (1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "ag_output_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout, output_shard_shape, output_shard_grid, output_mem_layout, buffer_type, use_semaphore_free_all_gather_impl, enable_trace, num_iters",
    [
        (
            [1, 1, 32, 3072],
            3,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (32, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            True,
            True,
            10,  # perf
        ),
        (
            [1, 1, 384, 1024],
            3,
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 1024),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            False,
            False,
            1,  # check
        ),
        (
            [1, 1, 384, 3072],
            3,
            (64, 384),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (384, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            True,
            True,
            10,  # perf
        ),
        # Composite-AG
        (
            [1, 1, 384, 240],
            3,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 256),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            False,
            False,
            1,  # check
        ),
    ],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_async_sharded_to_sharded(
    mesh_device,
    num_links,
    layout,
    ag_input_dtype,
    ag_output_shape,
    dim,
    input_shard_shape,
    input_shard_grid,
    input_mem_layout,
    output_shard_shape,
    output_shard_grid,
    output_mem_layout,
    buffer_type,
    use_semaphore_free_all_gather_impl,
    enable_trace,
    num_iters,
    all_gather_topology,
):
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_shard_spec = ttnn.ShardSpec(
        output_shard_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(input_mem_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)
    mem_config_ag = ttnn.MemoryConfig(output_mem_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)

    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        use_semaphore_free_all_gather_impl=use_semaphore_free_all_gather_impl,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "num_links, layout, ag_input_dtype",
    [
        (1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "ag_output_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout, buffer_type, use_semaphore_free_all_gather_impl, enable_trace, num_iters",
    [
        (
            [1, 1, 32, 3072],
            3,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            False,
            True,
            10,  # perf
        ),
        (
            [1, 1, 384, 1024],
            3,
            (64, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            True,
            False,
            1,  # check
        ),
        # Composite AG
        (
            [1, 1, 384, 240],
            3,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            False,
            True,
            10,  # perf
        ),
    ],
    ids=[
        "i2s_shape0-perf",
        "i2s_shape1-check",
        "i2s_shape2-perf",
    ],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_async_sharded_to_interleaved(
    mesh_device,
    num_links,
    layout,
    ag_input_dtype,
    ag_output_shape,
    dim,
    input_shard_shape,
    input_shard_grid,
    input_mem_layout,
    buffer_type,
    use_semaphore_free_all_gather_impl,
    enable_trace,
    num_iters,
    all_gather_topology,
):
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(input_mem_layout, buffer_type=buffer_type, shard_spec=input_shard_spec)
    mem_config_ag = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        use_semaphore_free_all_gather_impl=use_semaphore_free_all_gather_impl,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "num_links, layout, ag_input_dtype",
    [
        (1, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "ag_output_shape, dim, output_shard_shape, output_shard_grid, output_mem_layout, buffer_type, use_semaphore_free_all_gather_impl, enable_trace, num_iters",
    [
        (
            [1, 1, 32, 3072],
            3,
            (32, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            True,
            False,
            1,  # check
        ),
        (
            [1, 1, 384, 1024],
            3,
            (64, 1024),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            False,
            True,
            10,  # perf
        ),
        # Composite AG
        (
            [1, 1, 384, 240],
            3,
            (64, 256),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            True,
            False,
            1,  # check
        ),
    ],
    ids=[
        "i2s_shape0-check",
        "i2s_shape1-perf",
        "i2s_shape2-check",
    ],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_async_interleaved_to_sharded(
    mesh_device,
    num_links,
    layout,
    ag_input_dtype,
    ag_output_shape,
    dim,
    output_shard_shape,
    output_shard_grid,
    output_mem_layout,
    buffer_type,
    use_semaphore_free_all_gather_impl,
    enable_trace,
    num_iters,
    all_gather_topology,
):
    output_shard_spec = ttnn.ShardSpec(
        output_shard_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    mem_config_input = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)
    mem_config_ag = ttnn.MemoryConfig(output_mem_layout, buffer_type=buffer_type, shard_spec=output_shard_spec)

    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        use_semaphore_free_all_gather_impl=use_semaphore_free_all_gather_impl,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype, use_semaphore_free_all_gather_impl",
    [
        # Gather on dim 0
        ([1, 1, 8, 4096], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True),
    ],
    ids=[
        "multiprocess",
    ],
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
        (False, 3),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_linear", "fabric2d_linear"],
)
def test_all_gather_async_2x4(
    mesh_device,
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
    use_semaphore_free_all_gather_impl,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 4)))
    run_all_gather_impl(
        submesh_device,
        submesh_device.get_num_devices(),
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
        use_barrier=True,
        use_persistent_buffers=False,
        cluster_axis=1,
        use_semaphore_free_all_gather_impl=use_semaphore_free_all_gather_impl,
    )


def _get_tensors(input_shape, mesh_shape, dim, cluster_axis, dtype, memory_config, layout, device):
    num_devices = math.prod(mesh_shape)
    replicate = mesh_shape[cluster_axis] if cluster_axis is not None else num_devices
    torch_input = torch.cat([torch.rand(input_shape).bfloat16() for _ in range(replicate)], dim=dim)

    shard_dims = (None, dim) if cluster_axis == 1 else (dim, None)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=mesh_shape),
        device=device,
    )

    torch_reference = torch_input.repeat([num_devices] + [1] * (len(input_shape) - 1))

    return tt_input, torch_reference


MESH_SHAPE = (2, 4)
LAYOUT = ttnn.TILE_LAYOUT

NUM_ITERS = 2


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [2, 2, 32, 32],
        [5, 32, 32],
        [2, 2, 2, 32, 32],
        [2, 2, 2, 2, 32, 32],
        [2, 2, 2, 16, 16],
        [2, 16, 16],
        [16, 16],
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
def test_nd(mesh_device, input_shape, dim, cluster_axis, dtype, memory_config, topology):
    if dim >= len(input_shape):
        pytest.skip("Invalid gather dim")

    tt_input, torch_reference = _get_tensors(
        input_shape,
        tuple(mesh_device.shape),
        dim,
        cluster_axis,
        dtype,
        memory_config,
        ttnn.TILE_LAYOUT,
        mesh_device,
    )

    tile_size = tt_input.spec.tile.tile_shape[0]
    rank = len(tt_input.shape)
    gather_dim_normalized = dim if dim >= 0 else rank + dim
    is_tile_padded = gather_dim_normalized >= rank - 2 and (
        (gather_dim_normalized == rank - 2 and tt_input.shape[-2] % tile_size != 0)
        or (gather_dim_normalized == rank - 1 and tt_input.shape[-1] % tile_size != 0)
    )

    input_topology = tt_input.tensor_topology()

    # Create expected topology based on which all-gather path was used
    if is_tile_padded:
        expected_topology = ttnn.TensorTopology(
            input_topology.distribution_shape(), list(input_topology.placements()), input_topology.mesh_coords()
        )
    else:
        expected_placements = list(input_topology.placements())
        expected_placements[cluster_axis] = ttnn.PlacementReplicate()
        expected_topology = ttnn.TensorTopology(
            input_topology.distribution_shape(), expected_placements, input_topology.mesh_coords()
        )

    for i in range(NUM_ITERS):
        tt_out_tensor = ttnn.all_gather(
            tt_input,
            dim,
            cluster_axis=cluster_axis,
            topology=topology,
        )

        tt_output_tensor = torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(tt_out_tensor)])

        eq, mess = comp_pcc(torch_reference, tt_output_tensor)
        assert eq, mess

        actual_topology = tt_out_tensor.tensor_topology()
        assert (
            actual_topology == expected_topology
        ), f"output TensorTopology mismatch (tile_padded={is_tile_padded}):\n  Expected: {expected_topology}\n  Actual: {actual_topology}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}, {"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
    ids=["fabric_linear", "fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 2, 32, 32],
    ],
)
def test_all_gather_async_2x4_non_flat_mesh(mesh_device, input_shape):
    torch.manual_seed(2005)
    devices = mesh_device.get_num_devices()
    input_shape[-1] *= devices

    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        device=mesh_device,
    )  # [2, 2, 32, 32] per device

    tt_output = ttnn.all_gather(tt_input, dim=3)  # [2, 2, 32, 32*devices] per device

    torch_output = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )  # [2*devices, 2, 32, 32*devices]

    torch_reference = torch_input.repeat([devices, 1, 1, 1])
    eq, output = comp_equal(torch_output, torch_reference)
    assert eq, f"Output mismatch between torch and ttnn all-gather: {output}"

    output_placements = tt_output.tensor_topology().placements()
    assert len(output_placements) == 1, f"Expected 1 placement, got {len(output_placements)}"
