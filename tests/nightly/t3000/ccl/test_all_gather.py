# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype
from models.common.utility_functions import skip_for_blackhole

from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from tracy import signpost


def is_unsupported_case(
    output_shape,
    dim,
    mem_config,
    num_devices,
    input_dtype,
    layout,
    tile,
    num_l1_banks=64,
    mem_config_input=None,
):
    if layout == ttnn.ROW_MAJOR_LAYOUT and input_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        return True, "Row-major layout with block-float datatype is an invalid combination"
    if layout == ttnn.TILE_LAYOUT and input_dtype == ttnn.fp8_e4m3:
        return True, "Tile layout with fp8_e4m3 datatype is an invalid combination"

    if output_shape[dim] % num_devices != 0:
        return True, f"Output shape {output_shape} dim{dim} must be a multiple of num devices {num_devices}"
    if tile != (32, 32) and input_dtype != ttnn.bfloat16:
        return True, "Tiny tile only supports bfloat16"

    if layout == ttnn.TILE_LAYOUT:
        # Average elem size, since block float tiles pack a shared exponent per tile
        elem_size = ttnn.Tile(tile).get_tile_size(input_dtype) / (tile[0] * tile[1])
    else:
        elem_size = ttnn.element_size(input_dtype)

    def tensor_size_bytes(shape):
        padded = list(shape)
        if layout == ttnn.TILE_LAYOUT:
            # TensorLayout promotes a tiled rank-<2 shape to rank 2 before padding it
            padded = [1] * (2 - len(padded)) + padded
            padded[-2] = math.ceil(padded[-2] / tile[0]) * tile[0]
            padded[-1] = math.ceil(padded[-1] / tile[1]) * tile[1]
        return math.prod(padded) * elem_size

    ## Check that we can readback results
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        if mem_config.shard_spec is not None and mem_config.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            page_width = mem_config.shard_spec.shape[-1]
        elif mem_config.nd_shard_spec is not None:
            page_width = mem_config.nd_shard_spec.shard_shape[-1]
        else:
            page_width = output_shape[-1]
        # Fast dispatch currently can't breakup readback of large pages into multiple smaller pages and is
        # limited to ~55K pages. Reference: BufferReadDispatchParams in tt_metal/impl/buffers/dispatch.hpp,
        # and calculate_max_prefetch_data_size_bytes().
        fast_dispatch_page_size_limit = 55 * 1024
        if page_width * elem_size > fast_dispatch_page_size_limit:
            return True, "Fast dispatch can't support reading back this page size in one shot"

    # Check that we can fit in L1 (if L1 config)
    L1_util = 0
    if mem_config.buffer_type == ttnn.BufferType.L1:
        L1_util += tensor_size_bytes(output_shape)
    if mem_config_input is not None and mem_config_input.buffer_type == ttnn.BufferType.L1:
        input_shape = list(output_shape)
        input_shape[dim] //= num_devices
        L1_util += tensor_size_bytes(input_shape)

    if L1_util > num_l1_banks * 1536 * 1024:
        return True, "Test_Infrastructure_Skip L1 test requires more memory than the total available in the device"

    # Check that each chip has a non-zero amount of data available
    if output_shape[dim] < num_devices:
        return (
            True,
            f"Output shape {output_shape} incompatible with {num_devices} on dim {dim} because some chips will have no tensor",
        )

    return False, ""


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def create_fabric_router_config(max_payload_size: int):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def run_all_gather_impl(
    mesh_device,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    num_links=1,
    all_gather_topology=None,
    num_iters=1,
    enable_trace=True,
    cluster_axis=None,
    use_barrier=False,
    use_persistent_buffers=True,
    chunks_per_sync=None,
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    allowed_pcc=1.0,
    skip_check=False,
    num_l1_banks=64,
    all_gather_function=None,
    sub_core_grids=None,
    use_broadcast=False,
    use_explicit_subdevice_id=True,
):
    torch.manual_seed(0)
    torch_dtype = tt_dtype_to_torch_dtype[ag_input_dtype]
    tile = (32, 32)
    use_sub_devices = False

    num_devices = mesh_device.get_num_devices()
    mesh_shape = tuple(mesh_device.shape)
    replicate = mesh_shape[cluster_axis] if cluster_axis is not None else num_devices

    # mem_config_ag=None makes the CCL internally derive output config = input config.
    # Do that here for validation and persistent buffer creation.
    mem_config_ag_resolved = mem_config_ag if mem_config_ag is not None else mem_config_input

    # Skip unsupported cases
    (is_known_failure, message) = is_unsupported_case(
        ag_output_shape,
        dim,
        mem_config_ag_resolved,
        replicate,
        ag_input_dtype,
        layout,
        tile,
        num_l1_banks,
        mem_config_input,
    )
    if is_known_failure:
        pytest.skip(f"{message}")

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
                    layout=layout,
                    dtype=ag_input_dtype,
                    memory_config=mem_config_ag_resolved,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            ]
        else:
            persistent_output_buffers = [
                ttnn.from_torch(
                    torch.zeros(ag_output_shape),
                    device=mesh_device,
                    layout=layout,
                    dtype=ag_input_dtype,
                    memory_config=mem_config_ag_resolved,
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

    for i in range(num_iters):
        if torch_dtype in (torch.bfloat16, torch.float32):
            torch_input = torch.randn(ag_output_shape, dtype=torch_dtype)
        else:
            torch_input = torch.randint(0, 100, ag_output_shape, dtype=torch_dtype)

        # torch -> ttnn dtype conversion may be lossy, so exclude that to isolate if CCL is lossy
        # (i.e. by converting golden to ttnn dtype we can expect pcc==1.0 for any dtype)
        ag_output_tensor = ttnn.to_torch(ttnn.from_torch(torch_input, dtype=ag_input_dtype, layout=layout))
        ag_output_tensor_goldens_list.append(ag_output_tensor)

        if cluster_axis is None:
            mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=dim)
        else:
            shard_dims = (None, dim) if cluster_axis == 1 else (dim, None)
            mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape)

        input_tensor_mesh = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=layout,
            dtype=ag_input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=mesh_mapper,
        )

        input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform the TT ops #####
    tt_all_gather_out_tensor_list = []

    def run_op(i):  # absolutely disgusting if-else condition because changing every call site is a humongous PITA
        if all_gather_function is None:
            logger.info(f"Using production all-gather")
            tt_all_gather_out_tensor = ttnn.all_gather(
                input_tensor_mesh_list[i],
                dim=dim,
                memory_config=mem_config_ag,
                output_tensor=persistent_output_buffers[i] if use_persistent_buffers else None,
                cluster_axis=cluster_axis,
                subdevice_id=worker_sub_device_id if use_explicit_subdevice_id else None,
                sub_core_grids=sub_core_grids,
            )
        else:
            logger.info(f"Using experimental {all_gather_function.python_fully_qualified_name}")
            all_gather_async_kwargs = {
                "persistent_output_buffer": persistent_output_buffers[i] if use_persistent_buffers else None,
                "dim": dim,
                "multi_device_global_semaphore": ccl_semaphore_handles[i],
                "num_links": num_links,
                "memory_config": mem_config_ag,
                "topology": all_gather_topology,
                "barrier_semaphore": barrier_semaphore_handles[i] if use_barrier else None,
                "cluster_axis": cluster_axis,
                "chunks_per_sync": chunks_per_sync,
                "num_workers_per_link": num_workers_per_link,
                "num_buffers_per_channel": num_buffers_per_channel,
                "sub_core_grids": sub_core_grids,
                "use_broadcast": use_broadcast,
            }
            if use_explicit_subdevice_id:
                all_gather_async_kwargs["subdevice_id"] = worker_sub_device_id
            tt_all_gather_out_tensor = all_gather_function(input_tensor_mesh_list[i], **all_gather_async_kwargs)

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
        # For functional testing, inject arbitrary skew between devices to test
        # semaphore syncs
        delays = [[0 for j in range(mesh_shape[1])] for i in range(mesh_shape[0])]
        delays[0][0] = 400_000
        delays[-1][-1] = 800_000
        ttnn.apply_device_delay(mesh_device, delays)

        for i in range(num_iters):
            tt_all_gather_out_tensor = run_op(i)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    if not skip_check:
        # Check output_topology
        actual = [repr(p) for p in tt_all_gather_out_tensor_list[0].tensor_topology().placements()]
        expected = ["PlacementReplicate()"] * len(actual)
        assert actual == expected, f"FAILED output_topology: expected {expected}, got {actual}"

        for i in range(num_iters):
            tt_ag_out_tensor = tt_all_gather_out_tensor_list[i]
            torch_ag_out_tensor = ag_output_tensor_goldens_list[i if not enable_trace else 0]

            # Create expected output tensor based on which function is used
            if all_gather_function == ttnn.experimental.all_gather_async_reversed:
                # For reversed all-gather, we need to reverse the order along the gather dimension
                expected_tensor = torch_ag_out_tensor.clone()
                shard_size = torch_ag_out_tensor.shape[dim] // replicate

                # Reverse the shards along the gather dimension (only across cluster-axis devices)
                for device_id in range(replicate):
                    src_start = device_id * shard_size
                    src_end = (device_id + 1) * shard_size
                    dst_start = (replicate - 1 - device_id) * shard_size
                    dst_end = (replicate - device_id) * shard_size

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

            # Per-device compare: every device should hold `expected_tensor` after the all-gather
            # (gather along cluster_axis + replicate along the other mesh axis).
            coords = list(tt_ag_out_tensor.tensor_topology().mesh_coords())
            view = mesh_device.get_view() if ttnn.using_distributed_env() else None
            device_tensors = ttnn.get_device_tensors(tt_ag_out_tensor)
            coord_iter = coords
            if view is not None and len(device_tensors) != len(coords):
                coord_iter = [coord for coord in coords if view.is_local(coord)]

            for coord, tt_out in zip(coord_iter, device_tensors):
                if view is not None and not view.is_local(coord):
                    continue
                eq, output = comp_pcc(ttnn.to_torch(tt_out), expected_tensor, allowed_pcc)
                logger.info(f"{output}, iteration {i}, device {coord}")
                assert eq, f"iter {i} device {coord} FAILED ag: {output}"

    mesh_device.reset_sub_device_stall_group()
    if use_sub_devices:
        mesh_device.clear_loaded_sub_device_manager()


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype, enable_trace, num_iters, use_persistent_buffers",
    [
        ([1, 1, 3072, 8192], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10, True),  # perf
        ([1, 1, 352, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, True),  # check
        ([1, 1, 1024, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10, True),  # perf
        ([1, 1, 1024, 5120], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, False, 1, True),  # check, bf8
        ([8, 1, 512, 512], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, False),  # check
        ([1, 8, 512, 512], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10, True),  # perf
        ([1, 1, 1024, 1024], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10, True),  # perf
        ([1, 1, 512, 48], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, True),  # check
        ([1, 1, 48, 1024], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, True),  # check, padded
        ([1, 1, 1024, 1024], -2, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10, True),  # perf
        ([1, 1, 48, 1024], -1, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, True),  # check, padded
        ([256], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, True),  # check, rank 1
        # rank 1 row-major, no persistent buffer: padded rank stays 1, and the output is allocated from the spec
        ([256], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, False, 1, False),  # check, rank 1
        # composite (RM last-dim unaligned pages)
        ([1, 1, 32, 136], 3, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, True, 10, True),  # perf, composite
        # composite (tile padding on gather dim)
        ([1, 1, 48, 32], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, True),  # check, composite
        ([1600], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1, True),  # check, composite, rank 1
        # per-device 16 elements is a whole bfloat8_b block, so the concat re-quantizes exactly
        ([128], 0, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, False, 1, True),  # check, composite, rank 1
    ],
    ids=[
        "dit_shape-perf",
        "sd35_prompt-check",
        "sd35_spatial-perf",
        "sd35_spatial-check-bfloat8_b",
        "gather_dim_0-check",
        "gather_dim_1-perf",
        "gather_dim_2-perf",
        "gather_dim_2_padded_dim_3-check",
        "gather_dim_3_padded_dim_2-check",
        "gather_dim_negative_2-perf",
        "gather_dim_negative_1_padded_dim_2-check",
        "rank1_persistent_buffer-check",
        "rank1_row_major-check",
        "composite_ag_test_one-perf",
        "composite_ag_test_two-check",
        "composite_rank1-check",
        "composite_rank1_bfloat8_b-check",
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
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "fabric_router_config": create_fabric_router_config(6144),
            "trace_region_size": 90112,
        },
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(6144),
            "trace_region_size": 90112,
        },
    ],
    indirect=True,
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather(
    mesh_device,
    ag_output_shape,
    dim,
    layout,
    ag_input_dtype,
    enable_trace,
    num_iters,
    use_persistent_buffers,
    mem_config_input,
    mem_config_ag,
):
    run_all_gather_impl(
        mesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        use_persistent_buffers=use_persistent_buffers,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim",
    [
        ([1, 1, 1024, 5120], 3),
    ],
)
@pytest.mark.parametrize(
    "ag_input_dtype, layout",
    [
        (ttnn.float32, ttnn.TILE_LAYOUT),
        # (ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT),  # no WH support (Issue #43909)
        (ttnn.bfloat16, ttnn.TILE_LAYOUT),
        (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        (ttnn.bfloat4_b, ttnn.TILE_LAYOUT),
        (ttnn.uint32, ttnn.TILE_LAYOUT),
        (ttnn.uint16, ttnn.TILE_LAYOUT),
        (ttnn.uint8, ttnn.TILE_LAYOUT),
        (ttnn.int32, ttnn.TILE_LAYOUT),
    ],
    ids=[
        "float32_tile",
        # "fp8_e4m3_rm",
        "bfloat16_tile",
        "bfloat16_rm",
        "bfloat8_b_tile",
        "bfloat4_b_tile",
        "uint32_tile",
        "uint16_tile",
        "uint8_tile",
        "int32_tile",
    ],
)
@pytest.mark.parametrize("mem_config_input, mem_config_ag", [(ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)])
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
    ],
    indirect=True,
    ids=["fabric_ring"],
)
def test_all_gather_dtype(
    mesh_device,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
):
    run_all_gather_impl(
        mesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=False,
        num_iters=1,
        use_persistent_buffers=True,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
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
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_subgrid(
    mesh_device,
    ag_output_shape,
    dim,
    layout,
    ag_input_dtype,
    enable_trace,
    num_iters,
    mem_config_input,
    mem_config_ag,
    sub_core_grids,
):
    run_all_gather_impl(
        mesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        sub_core_grids=sub_core_grids,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype, enable_trace, num_iters",
    [
        # Gather on dim 0
        ([24, 3, 128, 96], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # perf
        ([16, 1, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # check
        ([16, 16, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # perf
        ([8, 16, 8, 8], 0, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # check
        # Gather on dim 1
        ([3, 24, 128, 96], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # perf
        ([1, 16, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # check
        ([16, 16, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # perf
        ([16, 8, 8, 8], 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # check
        # Gather on dim 2
        ([1, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # perf
        ([16, 1, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # check
        ([16, 16, 512, 8], 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # perf
        # # Gather on dim 3
        ([1, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # check
        ([16, 1, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, True, 10),  # perf
        ([16, 16, 8, 512], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, False, 1),  # check
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
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_training_shapes(
    mesh_device,
    ag_output_shape,
    dim,
    layout,
    ag_input_dtype,
    enable_trace,
    num_iters,
    mem_config_input,
    mem_config_ag,
):
    run_all_gather_impl(
        mesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        use_persistent_buffers=False,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "layout, ag_input_dtype",
    [
        (ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "ag_output_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout, output_shard_shape, output_shard_grid, output_mem_layout, buffer_type, enable_trace, num_iters",
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
            10,  # perf
        ),
        # Composite-AG (tile padding on gather dim)
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
            1,  # check
        ),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_sharded_to_sharded(
    mesh_device,
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
    enable_trace,
    num_iters,
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
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "layout, ag_input_dtype",
    [
        (ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "ag_output_shape, dim, input_shard_shape, input_shard_grid, input_mem_layout, buffer_type, enable_trace, num_iters",
    [
        (
            [1, 1, 32, 3072],
            3,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
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
            False,
            1,  # check
        ),
        # Composite AG (tile padding on gather dim)
        (
            [1, 1, 384, 240],
            3,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
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
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_sharded_to_interleaved(
    mesh_device,
    layout,
    ag_input_dtype,
    ag_output_shape,
    dim,
    input_shard_shape,
    input_shard_grid,
    input_mem_layout,
    buffer_type,
    enable_trace,
    num_iters,
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
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "layout, ag_input_dtype",
    [
        (ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "ag_output_shape, dim, output_shard_shape, output_shard_grid, output_mem_layout, buffer_type, enable_trace, num_iters",
    [
        (
            [1, 1, 32, 3072],
            3,
            (32, 512),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
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
            True,
            10,  # perf
        ),
        # Composite AG (tile padding on gather dim)
        (
            [1, 1, 384, 240],
            3,
            (64, 256),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
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
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
    ],
    indirect=True,
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_interleaved_to_sharded(
    mesh_device,
    layout,
    ag_input_dtype,
    ag_output_shape,
    dim,
    output_shard_shape,
    output_shard_grid,
    output_mem_layout,
    buffer_type,
    enable_trace,
    num_iters,
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
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )


# Width-sharded RM L1 memory config with shard shape (shard_height, shard_width)
# spread across num_cores cores of the 8x8 worker grid.
def _l1_width_sharded(shard_height, shard_width, num_cores):
    grid = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(8, 8), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


# Width-sharded RM DRAM config
def _dram_width_sharded(shard_height, shard_width, num_cores):
    grid = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(8, 8), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


# ND-sharded L1 memory config. Shards are cut based on ceil(tensor_dim / shard_dim) per dim.
# Unlike legacy 2D sharding a core may hold several shards, and the last shard along a dim may be partially filled.
def _l1_nd_sharded(
    shard_shape,
    num_cores=8,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
):
    grid = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(8, 8), row_wise=True)
    return ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape(shard_shape), grid, orientation, strategy))


# ND-sharded DRAM config.
def _dram_nd_sharded(
    shard_shape,
    num_banks=8,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
):
    # For DRAM buffers the bank id is the core's x-coord alone, so the grid must be a single ROW of cores.
    grid = ttnn.num_cores_to_corerangeset(num_banks, ttnn.CoreCoord(8, 1), row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.BufferType.DRAM, ttnn.NdShardSpec(ttnn.Shape(shard_shape), grid, orientation, strategy)
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("layout, ag_input_dtype", [(ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)], ids=["rm_bf16"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, mem_config_input, mem_config_ag",
    [
        # Page-indexing code paths of the all_gather kernel's chunk iterator.
        # Glossary (see all_gather_factory.cpp): a "chunk" = one NOC write = min(input_page, output_page).
        #   m = output_chunks_per_page (chunks packed into one output page; m>1 == concat)
        #   s = split_factor           (chunks an input page splits into; s>1 == split)
        #   k = input_pages_per_stripe (input width-shards per row; k>1 == multi-shard input)
        # Modes: matched (m=s=1) | concat (m>1) | split (s>1).
        #
        # matched (m=1,s=1,k=1): RM last-dim, equal in/out shard widths.
        ([1, 1, 32, 512], -1, _l1_width_sharded(32, 64, 1), _l1_width_sharded(32, 64, 8)),
        # matched at rank 1 (m=1,s=1,k=1): a sub-rank-2 padded shape is promoted to [1, N] for the shard grid.
        ([256], 0, _l1_width_sharded(1, 32, 1), _l1_width_sharded(1, 32, 8)),
        # concat full (m=8,s=1,k=1): sharded -> interleaved; one chunk/device packed per output row.
        ([1, 1, 32, 512], -1, _l1_width_sharded(32, 64, 1), ttnn.L1_MEMORY_CONFIG),
        # concat partial (m=2,s=1,k=1): 2 device contributions per output page (multiple pages/row).
        ([1, 1, 32, 512], -1, _l1_width_sharded(32, 64, 1), _l1_width_sharded(32, 128, 4)),
        # split (m=1,s=2,k=1): output sharded twice as finely as input.
        ([1, 1, 32, 512], -1, _l1_width_sharded(32, 64, 1), _l1_width_sharded(32, 32, 16)),
        # matched multi-shard (m=1,s=1,k=2): >1 input page per row, equal output shard width.
        ([1, 1, 32, 1024], -1, _l1_width_sharded(32, 64, 2), _l1_width_sharded(32, 64, 16)),
        # split multi-shard (m=1,s=2,k=2): split combined with multi-shard input.
        ([1, 1, 32, 1024], -1, _l1_width_sharded(32, 64, 2), _l1_width_sharded(32, 32, 32)),
        # multi-shard concat, no straddle (m=4,s=1,k=2): m a multiple of k; per-chunk byte offsets.
        ([1, 1, 32, 1024], -1, _l1_width_sharded(32, 64, 2), _l1_width_sharded(32, 256, 4)),
        # multi-shard concat WITH straddle (m=4,s=1,k=3): a device's 3 chunks cross a 4-chunk
        #   output-page boundary (page carry mid-stripe). m | N*k: 4|24.
        ([1, 1, 32, 1536], -1, _l1_width_sharded(32, 64, 3), _l1_width_sharded(32, 256, 6)),
        # multi-shard concat, run spans MULTIPLE output pages (m=2,s=1,k=3, m<k): byte-offset wrap
        #   + repeated page carry mid-stripe. m | N*k: 2|24.
        ([1, 1, 32, 1536], -1, _l1_width_sharded(32, 64, 3), _l1_width_sharded(32, 128, 12)),
        # RM interleaved last-dim (concat m=8,k=1, interleaved both sides): native RM-interleaved
        #   accessor path; per-device page 64*2=128B is aligned so it stays off the composite route.
        ([1, 1, 32, 512], -1, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        # non-last-dim gather (m=1,s=1,k from extents), RM interleaved both sides: gather along height.
        ([1, 1, 256, 64], 2, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        # split, padded output (m=1,s=2,k=1): DRAM shard content 16B pads to 32B, so split uses content.
        ([8, 1, 32, 16], 0, _dram_width_sharded(32, 16, 1), _dram_width_sharded(256, 8, 2)),
        # matched, padded, sharded -> interleaved, non-last-dim (m=1,s=1,k=1): a full-row DRAM shard
        # (content 136*2=272B, padded to the 32B DRAM page) gathered on the height dim into interleaved output.
        ([1, 1, 256, 136], 2, _dram_width_sharded(32, 136, 1), ttnn.DRAM_MEMORY_CONFIG),
        # matched, output page slot smaller than the input's aligned page (m=1,s=1,k=1): 272B content
        # pads to 288B in DRAM (32B align) but stays 272B in L1 (16B align).
        ([1, 1, 256, 136], 2, _dram_nd_sharded([1, 1, 32, 136], 1), _l1_nd_sharded([1, 1, 32, 136])),
    ],
    ids=[
        "matched",
        "matched_rank1",
        "concat_full_to_interleaved",
        "concat_partial",
        "split",
        "matched_multishard",
        "split_multishard",
        "multishard_concat_no_straddle",
        "multishard_concat_straddle",
        "multishard_concat_straddle_multipage",
        "rm_interleaved_last_dim",
        "non_last_dim_interleaved",
        "split_padded_output",
        "matched_sharded_to_interleaved_padded",
        "matched_out_page_smaller_than_in_aligned",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}],
    indirect=True,
    ids=["fabric_ring"],
)
def test_all_gather_page_indexing(
    mesh_device,
    layout,
    ag_input_dtype,
    ag_output_shape,
    dim,
    mem_config_input,
    mem_config_ag,
):
    run_all_gather_impl(
        mesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=False,
        num_iters=1,
        use_persistent_buffers=False,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("ag_input_dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, mem_config_input, mem_config_ag",
    [
        # Contiguous-run paths of the all_gather kernels. A run is one column of the walk's tile:
        # chunks contiguous at the destination, sent as one transfer. stride comes from
        # TensorAccessor::contiguous_page_stride(), xfer from the packet size.
        #
        # A long stripe is what lets a run form at all, so most cases gather on a non-last dim.
        # Slice lengths rarely divide a tile, so every case also exercises the ragged last tile.
        #
        # stride = DRAM bank count, runs along one bank.
        ([1, 1, 512, 128], 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        # stride (L1 bank count) exceeds the stripe, so no column can hold two chunks: page order.
        ([1, 1, 128, 128], 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        # One-page-wide shard, so stride is the tensor row in pages, not 1. Easiest thing to get wrong.
        (
            [1, 1, 256, 512],
            2,
            ttnn.ROW_MAJOR_LAYOUT,
            _l1_width_sharded(32, 64, 8),
            _l1_width_sharded(256, 64, 8),
        ),
        # Sharded, stride 1: runs walk plain page order to the shard edge.
        ([1, 1, 512, 128], 2, ttnn.TILE_LAYOUT, _l1_nd_sharded([1, 1, 64, 64]), _l1_nd_sharded([1, 1, 64, 64])),
        # split (s=2), stride 1: iteration 0's run reassembles the input page the split cut up.
        ([1, 1, 32, 512], -1, ttnn.ROW_MAJOR_LAYOUT, _l1_width_sharded(32, 64, 1), _l1_width_sharded(32, 32, 16)),
        # split (s=2) with stride 16: iteration 0 walks in the output's order, so its input chunks are
        # no longer adjacent and each must go on its own.
        (
            [1, 1, 256, 512],
            2,
            ttnn.ROW_MAJOR_LAYOUT,
            _l1_width_sharded(32, 64, 8),
            _l1_width_sharded(256, 32, 16),
        ),
        # concat (m=8) with k=4: chunks are packed inside an output page, so the run is intra-page.
        ([1, 1, 32, 2048], -1, ttnn.ROW_MAJOR_LAYOUT, _l1_width_sharded(32, 64, 4), _l1_width_sharded(32, 512, 4)),
        # Padded output page (16B content, 32B DRAM page): a run would step by the aligned size while
        # the CB is packed, so xfer drops to 1, i.e. page order with a chunk per transfer.
        ([8, 1, 32, 16], 0, ttnn.ROW_MAJOR_LAYOUT, _dram_width_sharded(32, 16, 1), _dram_width_sharded(256, 8, 2)),
        # One input page per device, so workers outnumber pages and the trailing slices are empty.
        ([1, 1, 1, 256], -1, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        # concat whose output pages are adjacent (ND shard spans the full width, so page stride is 1),
        # which is the only case that runs the cross-page term of the concat run. The concat case above
        # has a one-page-wide shard, where that term is off.
        ([1, 1, 64, 256], -1, ttnn.ROW_MAJOR_LAYOUT, _l1_nd_sharded([1, 1, 32, 32]), _l1_nd_sharded([1, 1, 32, 256])),
        # xfer == 1 because one chunk fills the packet, rather than because the page is padded.
        ([1, 1, 256, 2048], 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
    ids=[
        "interleaved_dram_bank_runs",
        "stride_exceeds_stripe",
        "width_sharded_strided_runs",
        "sharded_page_order_runs",
        "split_input_page_runs",
        "split_strided_no_input_runs",
        "concat_intra_page_runs",
        "padded_output_page_no_runs",
        "empty_worker_slices",
        "concat_cross_page_runs",
        "xfer1_fills_packet",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}],
    indirect=True,
    ids=["fabric_ring"],
)
def test_all_gather_contiguous_runs(
    mesh_device,
    ag_output_shape,
    dim,
    layout,
    ag_input_dtype,
    mem_config_input,
    mem_config_ag,
):
    run_all_gather_impl(
        mesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=False,
        num_iters=1,
        use_persistent_buffers=False,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")

# Two configurations that the unicast all_gather gets wrong, both reachable with the production
# factory heuristic. Verified to fail identically on the commit before the tiled-transpose walk, so
# they are pre-existing, not a regression from it. Skipped because one of them hangs the board.
#
#   mismatched_alignment: input L1 (16 B align) -> output DRAM (64 B align) with a 32 B page, so the
#       output page is padded relative to the chunk. Small shapes hang; large ones return garbage
#       (PCC 0.0005 before the rewrite, 0.003 after).
#   chunk_over_packet: a page larger than the fabric packet, which takes queue_segment's
#       "chunk bigger than a packet" split. PCC 0.5338325678716095, bit-identical before and after.
@pytest.mark.skip(reason="pre-existing all_gather unicast bugs; mismatched_alignment hangs the board")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, mem_config_input, mem_config_ag",
    [
        ([8, 1, 4096, 16], 0, ttnn.ROW_MAJOR_LAYOUT, _l1_width_sharded(4096, 16, 1), _dram_width_sharded(32768, 16, 1)),
        ([1, 1, 64, 4096], 2, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
    ids=["mismatched_alignment", "chunk_over_packet"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_all_gather_unicast_known_bad(
    mesh_device, ag_output_shape, dim, layout, mem_config_input, mem_config_ag, all_gather_topology
):
    run_all_gather_impl(
        mesh_device,
        ag_output_shape,
        dim,
        ttnn.bfloat16,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=False,
        num_iters=1,
        use_persistent_buffers=False,
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("ag_input_dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, mem_config_input, mem_config_ag",
    [
        # ND sharding: the shard shape is a full ND shape instead of being flattened to 2D.
        # The m/s/k iterator modes are covered by test_all_gather_page_indexing. What's exercised here
        # is page_id -> address for ND page grids, plus the host's output-spec/route plumbing.
        #
        # --- basic: matched mode (in page == out page), tile + row-major, gather dim varied ---
        ([1, 1, 128, 512], -1, ttnn.TILE_LAYOUT, _l1_nd_sharded([1, 1, 64, 64]), _l1_nd_sharded([1, 1, 64, 64])),
        ([1, 1, 512, 128], 2, ttnn.TILE_LAYOUT, _l1_nd_sharded([1, 1, 64, 64]), _l1_nd_sharded([1, 1, 64, 64])),
        ([2, 8, 64, 128], 1, ttnn.TILE_LAYOUT, _l1_nd_sharded([1, 1, 32, 64]), _l1_nd_sharded([1, 1, 32, 64])),
        ([1, 1, 64, 512], -1, ttnn.ROW_MAJOR_LAYOUT, _l1_nd_sharded([1, 1, 8, 64]), _l1_nd_sharded([1, 1, 8, 64])),
        ([2, 8, 32, 128], 1, ttnn.ROW_MAJOR_LAYOUT, _l1_nd_sharded([1, 1, 8, 128]), _l1_nd_sharded([1, 1, 8, 128])),
        # --- partial shards: tensor extent % shard extent != 0, so the last shard along a dim is
        #     only part-filled. Those slots are never addressed, so the page grid is unchanged.
        ([3, 8, 64, 64], 1, ttnn.TILE_LAYOUT, _l1_nd_sharded([2, 1, 32, 64]), _l1_nd_sharded([2, 1, 32, 64])),
        ([1, 1, 96, 768], -1, ttnn.TILE_LAYOUT, _l1_nd_sharded([1, 1, 64, 64]), _l1_nd_sharded([1, 1, 64, 64])),
        ([1, 24, 64, 64], 1, ttnn.TILE_LAYOUT, _l1_nd_sharded([1, 2, 32, 64]), _l1_nd_sharded([1, 2, 32, 64])),
        # --- shard rank < tensor rank (squeezed down to the shard's rank) ---
        ([2, 8, 64, 128], 1, ttnn.TILE_LAYOUT, _l1_nd_sharded([32, 64]), _l1_nd_sharded([32, 64])),
        # --- rank-1 tensors. Row-major keeps a padded rank of 1, so the shard shape must be rank 1 too.
        #     Tiled is padded to rank 2, so it takes a legacy 2D spec (an ND one trips the mesh mapper's
        #     own rank check, upstream of this op). ---
        ([256], 0, ttnn.ROW_MAJOR_LAYOUT, _l1_nd_sharded([32]), _l1_nd_sharded([32])),
        ([256], 0, ttnn.TILE_LAYOUT, _l1_width_sharded(32, 32, 1), ttnn.L1_MEMORY_CONFIG),
        # --- shard -> core mapping: round-robin wrap (several shards per core), CONTIGUOUS_1D
        #     (adjacent shards packed onto one core), non-rectangular grid, COL_MAJOR core order ---
        (
            [1, 1, 128, 512],
            -1,
            ttnn.TILE_LAYOUT,
            _l1_nd_sharded([1, 1, 32, 32], num_cores=4),
            _l1_nd_sharded([1, 1, 32, 32], num_cores=4),
        ),
        (
            [1, 1, 128, 512],
            -1,
            ttnn.TILE_LAYOUT,
            _l1_nd_sharded([1, 1, 32, 32], strategy=ttnn.ShardDistributionStrategy.CONTIGUOUS_1D),
            _l1_nd_sharded([1, 1, 32, 32], strategy=ttnn.ShardDistributionStrategy.CONTIGUOUS_1D),
        ),
        (
            [1, 1, 128, 512],
            -1,
            ttnn.TILE_LAYOUT,
            _l1_nd_sharded([1, 1, 64, 64], num_cores=12),
            _l1_nd_sharded([1, 1, 64, 64], num_cores=12),
        ),
        (
            [1, 1, 128, 512],
            -1,
            ttnn.TILE_LAYOUT,
            _l1_nd_sharded([1, 1, 64, 64], orientation=ttnn.ShardOrientation.COL_MAJOR),
            _l1_nd_sharded([1, 1, 64, 64], orientation=ttnn.ShardOrientation.COL_MAJOR),
        ),
        # --- DRAM banks (bank id == core x, see _dram_nd_sharded) ---
        ([2, 8, 64, 128], 1, ttnn.TILE_LAYOUT, _dram_nd_sharded([1, 1, 32, 64]), _dram_nd_sharded([1, 1, 32, 64])),
        ([2, 8, 32, 128], 1, ttnn.ROW_MAJOR_LAYOUT, _dram_nd_sharded([1, 1, 8, 128]), _dram_nd_sharded([1, 1, 8, 128])),
        # --- mixed: ND on one side only ---
        ([1, 1, 128, 512], -1, ttnn.TILE_LAYOUT, _l1_nd_sharded([1, 1, 64, 64]), _l1_width_sharded(128, 64, 8)),
        ([1, 1, 128, 512], -1, ttnn.TILE_LAYOUT, _l1_nd_sharded([1, 1, 64, 64]), ttnn.DRAM_MEMORY_CONFIG),
        ([1, 1, 128, 512], -1, ttnn.TILE_LAYOUT, _l1_width_sharded(128, 32, 2), _l1_nd_sharded([1, 1, 64, 64])),
        # in:out shard widths with no integer ratio (96 vs 64 elements). Native for tile, where the
        # page is one tile either way, so shard width doesn't enter page indexing.
        ([1, 1, 128, 768], -1, ttnn.TILE_LAYOUT, _l1_nd_sharded([1, 1, 64, 96]), _l1_width_sharded(128, 64, 12)),
        # --- concat (m=2) and split (s=2) driven by unequal ND shard widths ---
        ([1, 1, 64, 512], -1, ttnn.ROW_MAJOR_LAYOUT, _l1_nd_sharded([1, 1, 8, 64]), _l1_nd_sharded([1, 1, 8, 128])),
        ([1, 1, 64, 512], -1, ttnn.ROW_MAJOR_LAYOUT, _l1_nd_sharded([1, 1, 8, 64]), _l1_nd_sharded([1, 1, 8, 32])),
        # --- mem_config_ag=None: the plain ttnn.all_gather(x, dim) call, where the output config is
        #     derived from the input's. The input's ND spec has a legacy 2D equivalent, whose grid
        #     rules must not be re-applied to the num_devices-times-larger output.
        ([8, 2, 64, 64], 0, ttnn.TILE_LAYOUT, _l1_nd_sharded([1, 1, 32, 64], num_cores=9), None),
        ([1, 1, 64, 512], -1, ttnn.ROW_MAJOR_LAYOUT, _l1_nd_sharded([1, 1, 8, 64]), None),
        # --- composite AG, since non-integral in:out page ratio (192B vs 256B)
        ([1, 1, 32, 768], -1, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG, _l1_nd_sharded([1, 1, 32, 128])),
        # --- composite AG, since padding on the gather dim
        ([1, 1, 32, 384], -1, ttnn.ROW_MAJOR_LAYOUT, _l1_nd_sharded([1, 1, 8, 64]), _l1_nd_sharded([1, 1, 8, 64])),
        ([1, 1, 32, 256], -1, ttnn.ROW_MAJOR_LAYOUT, _l1_nd_sharded([1, 1, 32, 32]), _l1_nd_sharded([1, 1, 32, 96])),
    ],
    ids=[
        "tile_last_dim",
        "tile_height_dim",
        "tile_middle_dim",
        "rm_last_dim",
        "rm_middle_dim",
        "partial_shard_outer_dim",
        "partial_shard_hw",
        "partial_shard_gather_dim",
        "shard_rank_lt_tensor_rank",
        "rank1_rm",
        "rank1_tile_sharded_to_interleaved",
        "many_shards_per_core",
        "contiguous_1d",
        "non_rectangular_grid",
        "col_major_cores",
        "tile_dram",
        "rm_dram",
        "nd_to_legacy_sharded",
        "nd_to_interleaved",
        "legacy_sharded_to_nd",
        "tile_unequal_shard_widths",
        "nd_concat",
        "nd_split",
        "default_out_config_tile",
        "default_out_config_rm",
        "composite_nonintegral_page_ratio",
        "composite_pad_on_gather_dim_input",
        "pad_on_gather_dim_output",
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True, ids=["fabric_1d"]
)
def test_all_gather_nd_sharded(
    mesh_device,
    layout,
    ag_input_dtype,
    ag_output_shape,
    dim,
    mem_config_input,
    mem_config_ag,
):
    run_all_gather_impl(
        mesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=False,
        num_iters=1,
        use_persistent_buffers=False,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "mapper_dim, gather_dim, expect_replicated",
    [
        # The mapper keeps the dim as the caller spelled it while the gather dim is normalized, so the two
        # have to be compared as axes: -1 and 3 are the same axis of a rank-4 tensor.
        (-1, -1, True),
        # Different axes, so the Shard placement must survive.
        (-2, 3, False),
        # No Shard placement anywhere to begin with; gathering must not introduce a spurious one.
        (None, -1, True),
    ],
    ids=["same_axis", "different_axis", "already_replicated"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["fabric_ring"]
)
def test_all_gather_output_topology(mesh_device, mapper_dim, gather_dim, expect_replicated):
    # Gathering the sharded axis replicates it, and the output topology has to say so.
    devices = mesh_device.get_num_devices()
    mesh_mapper = (
        ttnn.ReplicateTensorToMesh(mesh_device)
        if mapper_dim is None
        else ttnn.ShardTensorToMesh(mesh_device, dim=mapper_dim)
    )

    tt_input = ttnn.from_torch(
        torch.rand([1, 1, 32 * devices, 32 * devices], dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
        device=mesh_device,
    )
    tt_output = ttnn.all_gather(tt_input, dim=gather_dim)

    actual = [repr(p) for p in tt_output.tensor_topology().placements()]
    expected = ["PlacementReplicate()"] if expect_replicated else [f"PlacementShard({mapper_dim})"]
    assert actual == expected, f"FAILED output_topology: expected {expected}, got {actual}"


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype",
    [
        # Gather on dim 0
        ([1, 1, 8, 4096], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
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
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112},
    ],
    indirect=True,
    ids=["fabric_linear", "fabric2d_linear"],
)
def test_all_gather_2x4(
    mesh_device,
    ag_output_shape,
    dim,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 4)))
    run_all_gather_impl(
        submesh_device,
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        enable_trace=enable_trace,
        num_iters=num_iters,
        use_persistent_buffers=False,
        cluster_axis=1,
    )


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "cluster_axis, gather_dim, other_dim",
    [
        # Different tensor dims sharded on each mesh axis; gathering one must not disturb the other's Shard.
        (0, 2, 3),
        (1, 3, 2),
    ],
    ids=["cluster_axis_0", "cluster_axis_1"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True, ids=["fabric_1d"]
)
def test_all_gather_output_topology_2x4(mesh_device, cluster_axis, gather_dim, other_dim):
    mesh_shape = tuple(mesh_device.shape)
    shard_dims = [None, None]
    shard_dims[cluster_axis] = gather_dim
    shard_dims[1 - cluster_axis] = other_dim

    tt_input = ttnn.from_torch(
        torch.rand([1, 1, 32 * mesh_shape[0], 32 * mesh_shape[1]], dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(shard_dims), mesh_shape=mesh_shape),
        device=mesh_device,
    )

    tt_output = ttnn.all_gather(tt_input, dim=gather_dim, cluster_axis=cluster_axis)

    actual = [repr(p) for p in tt_output.tensor_topology().placements()]
    expected = [None, None]
    expected[cluster_axis] = "PlacementReplicate()"
    expected[1 - cluster_axis] = f"PlacementShard({other_dim})"
    assert actual == expected, f"FAILED output_topology: expected {expected}, got {actual}"


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "input_shape, dim, output_dtype, output_layout, output_shape, msg_pattern",
    [
        ([256], -2, ttnn.bfloat16, ttnn.TILE_LAYOUT, [2048], "Invalid gather dim"),
        (
            [1, 1, 32, 32],
            -1,
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            [1, 1, 32, 256],
            "Output tensor dtype .* should be same as input tensor dtype",
        ),
        (
            [1, 1, 32, 32],
            -1,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            [1, 1, 32, 256],
            "Output tensor layout .* should be same as input tensor layout",
        ),
        (
            [1, 1, 32, 32],
            -1,
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            [1, 1, 32, 128],
            "Output tensor shape must be",
        ),
    ],
    ids=[
        "invalid_dim",
        "persistent_output_wrong_dtype",
        "persistent_output_wrong_layout",
        "persistent_output_wrong_shape",
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["fabric_ring"]
)
def test_all_gather_negative_tests(
    mesh_device, input_shape, dim, output_dtype, output_layout, output_shape, msg_pattern, expect_error
):
    tt_input = ttnn.from_torch(
        torch.rand(input_shape, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )
    persistent_output_buffer = ttnn.from_torch(
        torch.zeros(output_shape),
        layout=output_layout,
        dtype=output_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )

    with expect_error(RuntimeError, msg_pattern):
        ttnn.all_gather(tt_input, dim=dim, output_tensor=persistent_output_buffer)


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
def test_nd(mesh_device, input_shape, dim, cluster_axis, dtype, memory_config):
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

    # An all-gather along cluster_axis replicates the result across that axis
    input_topology = tt_input.tensor_topology()
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
        )

        tt_output_tensor = torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(tt_out_tensor)])

        eq, mess = comp_pcc(torch_reference, tt_output_tensor)
        assert eq, mess

        actual_topology = tt_out_tensor.tensor_topology()
        assert (
            actual_topology == expected_topology
        ), f"output TensorTopology mismatch:\n  Expected: {expected_topology}\n  Actual: {actual_topology}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 2, 32, 32],
    ],
)
def test_all_gather_2x4_non_flat_mesh(mesh_device, input_shape):
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


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "shape, gather_dim, num_iters",
    [
        ([1, 1, 4096, 4096], 3, 10),  # output over ~12 MB picks the unicast factory
        ([1, 1, 1024, 1024], 3, 10),  # small output picks the multicast factory
    ],
    ids=["unicast", "multicast"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
    ids=["fabric_linear"],
)
def test_all_gather_overlap(mesh_device, shape, gather_dim, num_iters):
    """Stress test overlapping multiple CCL invocations to exercise:
    - Semaphore increments (credits) shouldn't be lost -> failure results in hang
    - Reusing persistent output buffers shouldn't clobber data -> failure results in bad PCC

    Performed by running cached invocations back to back with no sync.
    """
    dtype, layout = ttnn.bfloat16, ttnn.TILE_LAYOUT
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    def to_device(torch_tensor, mesh_mapper):
        return ttnn.from_torch(
            torch_tensor,
            device=mesh_device,
            layout=layout,
            dtype=dtype,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )

    torch.manual_seed(0)
    torch_inputs = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(num_iters)]
    goldens = [ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=layout)) for t in torch_inputs]

    tt_inputs = [to_device(t, ttnn.ShardTensorToMesh(mesh_device, dim=gather_dim)) for t in torch_inputs]
    out_buffer = to_device(torch.zeros(shape), ttnn.ReplicateTensorToMesh(mesh_device))
    captures = [to_device(torch.zeros(shape), ttnn.ReplicateTensorToMesh(mesh_device)) for _ in range(num_iters)]

    # Stall every device except (0, 0), so the fast device's next all_gather writes into buffers
    # the others have not read yet.
    rows, cols = tuple(mesh_device.shape)
    delays = [[0 if (r, c) == (0, 0) else 2_000_000 for c in range(cols)] for r in range(rows)]

    signpost("start")
    for i in range(num_iters):
        out = ttnn.all_gather(tt_inputs[i], dim=gather_dim, memory_config=mem_config, output_tensor=out_buffer)
        # skip ttnn.synchronize_device(mesh_device) here to get CCLs to overlap
        ttnn.apply_device_delay(mesh_device, delays)  # insert skew here so CCL's syncs don't absorb this
        ttnn.copy(out, captures[i])  # preserve output since next iteration reuses same output buffer
    ttnn.synchronize_device(mesh_device)
    signpost("stop")

    for i in range(num_iters):
        for device_idx, tt_out in enumerate(ttnn.get_device_tensors(captures[i])):
            eq, output = comp_pcc(ttnn.to_torch(tt_out), goldens[i], 1.0)
            assert eq, f"iter {i} device {device_idx} FAILED: {output}"


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["mesh_1,8"])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "ag_output_shape, dim, layout, ag_input_dtype, enable_trace, num_iters, use_barrier, use_persistent_buffers, pcc_threshold, mem_config_input, mem_config_ag",
    [
        (
            [1, 1, 32, 128 * 128],
            2,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            True,
            30,
            None,
            None,
            1.0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
                    (1, 128 * 128),  # (shard_height, shard_width)
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))}),
                    (32, 2048),  # (shard_height, shard_width)
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        ),
        (
            [1, 8, 32, 2112],
            1,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            35,
            None,
            None,
            1.0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))}),
                    (32, 288),  # (shard_height, shard_width)
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7)),
                            ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 0)),
                        }
                    ),
                    (32 * 8, 64),  # (shard_height, shard_width)
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        ),
        (
            [1, 1, 32, 128 * 128],
            2,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            True,
            30,
            None,
            None,
            1.0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
                    (1, 128 * 128),  # (shard_height, shard_width)
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED,
                ttnn.BufferType.L1,
            ),
        ),
        (
            [1, 8, 32, 2112],
            1,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            True,
            35,
            None,
            None,
            1.0,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))}),
                    (32, 288),  # (shard_height, shard_width)
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED,
                ttnn.BufferType.L1,
            ),
        ),
    ],
    ids=["RM_sharded", "TILED_sharded", "RM_interleaved", "TILED_interleaved"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_async_broadcast(
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
        ag_output_shape,
        dim,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        num_links=num_links,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        use_barrier=use_barrier,
        use_persistent_buffers=use_persistent_buffers,
        all_gather_function=ttnn.experimental.all_gather_async,
        allowed_pcc=pcc_threshold,
        use_broadcast=True,
    )
