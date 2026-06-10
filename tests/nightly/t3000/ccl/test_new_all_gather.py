# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

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
    page_size = input_shape[dim] * elem_size
    if layout == ttnn.ROW_MAJOR_LAYOUT and page_size > fast_dispatch_page_size_limit:
        # Fast dispatch currently can't breakup readback of large pages into multiple smaller pages and is
        # limited to ~55K pages.
        return (
            True,
            f"Fast dispatch can't support reading back this page size {page_size} in one shot (limit {fast_dispatch_page_size_limit})",
        )

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
    # ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value, ttnn.types.BufferType.L1_SMALL) for _ in range(2)]
    return ccl_semaphore_handles


def create_fabric_router_config(max_payload_size: int):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _linear_core_range_set(num_cores, max_cols):
    """Row-major linear core grid: fill rows of max_cols, then overflow to next row."""
    if num_cores <= max_cols:
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    full_rows = num_cores // max_cols
    remainder = num_cores % max_cols
    ranges = set()
    if full_rows > 0:
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_cols - 1, full_rows - 1)))
    if remainder > 0:
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(remainder - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def create_sharded_mem_config(
    shape, gather_dim, num_devices, mem_layout, buffer_type, layout, mesh_device, is_input=False
):
    """Compute a valid sharded MemoryConfig for the given shape, or return None."""
    working_shape = list(shape)
    if is_input:
        working_shape[gather_dim] = working_shape[gather_dim] // num_devices

    # For TILE_LAYOUT, the last two dims are padded up to multiples of 32
    tile_size = 32 if layout == ttnn.TILE_LAYOUT else 1
    if layout == ttnn.TILE_LAYOUT and len(working_shape) >= 2:
        working_shape[-2] = math.ceil(working_shape[-2] / 32) * 32
        working_shape[-1] = math.ceil(working_shape[-1] / 32) * 32

    total_rows = 1
    for d in working_shape[:-1]:
        total_rows *= d
    width = working_shape[-1]

    if buffer_type == ttnn.BufferType.DRAM:
        grid_size = mesh_device.dram_grid_size()
    else:
        grid_size = mesh_device.compute_with_storage_grid_size()
    max_cols = grid_size.x
    max_rows = grid_size.y
    max_cores = max_cols * max_rows

    if mem_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        for nc in range(max_cores, 0, -1):
            if total_rows % nc == 0:
                shard_h = total_rows // nc
                if shard_h % tile_size == 0 and width % tile_size == 0:
                    core_grid = _linear_core_range_set(nc, max_cols)
                    shard_spec = ttnn.ShardSpec(core_grid, (shard_h, width), ttnn.ShardOrientation.ROW_MAJOR)
                    return ttnn.MemoryConfig(mem_layout, buffer_type, shard_spec=shard_spec)
        return None

    elif mem_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        for nc in range(max_cores, 0, -1):
            if width % nc == 0:
                shard_w = width // nc
                if shard_w % tile_size == 0 and total_rows % tile_size == 0:
                    core_grid = _linear_core_range_set(nc, max_cols)
                    shard_spec = ttnn.ShardSpec(core_grid, (total_rows, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
                    return ttnn.MemoryConfig(mem_layout, buffer_type, shard_spec=shard_spec)
        return None

    elif mem_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        for R in range(max_rows, 0, -1):
            for C in range(max_cols, 0, -1):
                if total_rows % R == 0 and width % C == 0:
                    sh, sw = total_rows // R, width // C
                    if sh % tile_size == 0 and sw % tile_size == 0:
                        core_grid = ttnn.CoreRangeSet(
                            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(C - 1, R - 1))}
                        )
                        shard_spec = ttnn.ShardSpec(core_grid, (sh, sw), ttnn.ShardOrientation.ROW_MAJOR)
                        return ttnn.MemoryConfig(mem_layout, buffer_type, shard_spec=shard_spec)
        return None

    return None


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
    sub_core_grids=None,
    use_new_ag=False,
    use_sync_ag=False,
    use_explicit_subdevice_id=True,
):
    use_sub_devices = False
    torch.manual_seed(0)

    tile = (32, 32)

    mesh_shape = tuple(mesh_device.shape)
    replicate = mesh_shape[cluster_axis] if cluster_axis is not None else num_devices

    # Skip unsupported cases
    (is_known_failure, message) = is_unsupported_case(
        ag_output_shape,
        dim,
        mem_config_ag,
        replicate,
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

    for i in range(num_iters):
        ag_output_tensor = torch.rand(ag_output_shape).bfloat16()
        ag_output_tensor_goldens_list.append(ag_output_tensor)

        if cluster_axis is None:
            mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=dim)
        else:
            shard_dims = (None, dim) if cluster_axis == 1 else (dim, None)
            mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape)

        input_tensor_mesh = ttnn.from_torch(
            ag_output_tensor,
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
        if use_new_ag:
            logger.info(f"Using new all-gather")
            tt_all_gather_out_tensor = ttnn.experimental.all_gather(
                input_tensor_mesh_list[i],
                dim=dim,
                memory_config=mem_config_ag,
                output_tensor=persistent_output_buffers[i] if use_persistent_buffers else None,
                cluster_axis=cluster_axis,
            )
        elif use_sync_ag:
            logger.info(f"Using sync all-gather")
            all_gather_kwargs = {
                "dim": dim,
                "cluster_axis": cluster_axis,
                "num_links": num_links,
                "memory_config": mem_config_ag,
                "topology": all_gather_topology,
                "chunks_per_sync": chunks_per_sync,
                "num_workers_per_link": num_workers_per_link,
                "num_buffers_per_channel": num_buffers_per_channel,
                "sub_core_grids": sub_core_grids,
            }
            if use_explicit_subdevice_id:
                all_gather_kwargs["subdevice_id"] = worker_sub_device_id
            tt_all_gather_out_tensor = ttnn.all_gather(input_tensor_mesh_list[i], **all_gather_kwargs)
        else:
            logger.info(f"Using experimental all-gather")
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
        for i in range(num_iters):
            tt_ag_out_tensor = tt_all_gather_out_tensor_list[i]
            torch_ag_out_tensor = ag_output_tensor_goldens_list[i if not enable_trace else 0]

            # Create expected output tensor based on which function is used
            is_reversed = all_gather_function == ttnn.experimental.all_gather_async_reversed
            if is_reversed:
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
                logger.info(f"{output}, iteration {i}, device {coord}, reversed={is_reversed}")
                assert eq, f"iter {i} device {coord} FAILED ag: {output}"

    mesh_device.reset_sub_device_stall_group()
    if use_sub_devices:
        mesh_device.clear_loaded_sub_device_manager()


# Edit this to match your machine
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
# T3K = 1 link, WH Galaxy = 4 links
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "ag_output_shape, dim",
    [
        ([1, 1, 32, 1024], 3),  # 32K elems
        ([1, 1, 64, 1024], 3),  # 65K
        ([1, 1, 128, 1024], 3),  # 131K
        ([1, 1, 256, 2048], 3),  # 512K
        ([1, 1, 288, 3072], 3),  # 884K
        ([1, 1, 176, 5120], 3),  # 901K
        ([1, 1, 128, 7168], 3),  # 917K
        ([1, 1, 1024, 1024], 3),  # 1M
        ([1, 1, 2048, 1024], 3),  # 2M
        ([1, 1, 512, 5120], 3),  # 2.6M
        ([1, 1, 1024, 5120], 3),  # 5M
        ([1, 1, 2048, 5120], 3),  # 10M
        ([1, 1, 3072, 8192], 3),  # 25M
        ([1, 2, 3072, 8192], 3),  # 50M
        ([2, 2, 3072, 8192], 3),  # 100M
    ],
    ids=["32K", "65K", "131K", "512K", "884K", "901K", "917K", "1M", "2M", "2.6M", "5M", "10M", "25M", "50M", "100M"],
)
@pytest.mark.parametrize(
    "mem_layout",
    [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        # ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        # ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        # ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
    # ids=["interleaved", "height_sharded", "width_sharded"],
    ids=["interleaved"],
)
@pytest.mark.parametrize(
    "device_params, ag_topology",
    [
        # WH packet size = 6144  (= 3 bloat16 tiles)
        # BH packet size = 14336 (= 7 bloat16 tiles)
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 110000,
                "fabric_router_config": create_fabric_router_config(6144),
            },
            ttnn.Topology.Linear,
        ),
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 110000,
                "fabric_router_config": create_fabric_router_config(6144),
            },
            ttnn.Topology.Ring,
        ),
        # ({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 110000, "fabric_router_config": create_fabric_router_config(6144)}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["linear", "ring"],
    # ids=["linear", "ring", "mesh"],
)
@pytest.mark.parametrize("use_barrier", [False])  # =False for async CCL
@pytest.mark.parametrize("use_new_ag", [True], ids=["new_ag"])
@pytest.mark.parametrize("enable_trace", [True])
def test_ag_perf_sweep(
    mesh_device, num_links, ag_output_shape, dim, mem_layout, ag_topology, use_barrier, use_new_ag, enable_trace
):
    num_devices = mesh_device.get_num_devices()
    buffer_type = ttnn.BufferType.DRAM
    layout = ttnn.TILE_LAYOUT

    if mem_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        mem_config = ttnn.MemoryConfig(mem_layout, buffer_type)
        mem_config_input = mem_config
    else:
        mem_config = create_sharded_mem_config(
            ag_output_shape, dim, num_devices, mem_layout, buffer_type, layout, mesh_device, is_input=False
        )
        if mem_config is None:
            pytest.skip(f"No valid shard config for output")
        mem_config_input = create_sharded_mem_config(
            ag_output_shape, dim, num_devices, mem_layout, buffer_type, layout, mesh_device, is_input=True
        )
        if mem_config_input is None:
            pytest.skip(f"No valid shard config for input")

    run_all_gather_impl(
        mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ttnn.bfloat16,
        layout,
        mem_config_input,
        mem_config,
        all_gather_topology=ag_topology,
        enable_trace=enable_trace,
        num_iters=10 if enable_trace else 1,
        use_barrier=use_barrier,
        use_persistent_buffers=False if use_barrier else True,
        skip_check=True if enable_trace else False,
        use_new_ag=use_new_ag,
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "ag_output_shape, dim",
    [
        ([1, 1, 3, 16384], 3),
    ],
)
@pytest.mark.parametrize(
    "mem_layout",
    [
        ttnn.TensorMemoryLayout.INTERLEAVED,
    ],
    ids=["interleaved"],
)
@pytest.mark.parametrize(
    "device_params, ag_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 110000,
                "fabric_router_config": create_fabric_router_config(2048),
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["ring"],
)
@pytest.mark.parametrize("use_barrier", [True])
@pytest.mark.parametrize("use_new_ag", [True])
@pytest.mark.parametrize("enable_trace", [False])
def test_ag_rm(
    mesh_device, num_links, ag_output_shape, dim, mem_layout, ag_topology, use_barrier, use_new_ag, enable_trace
):
    num_devices = mesh_device.get_num_devices()
    buffer_type = ttnn.BufferType.DRAM
    layout = ttnn.ROW_MAJOR_LAYOUT

    if mem_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        mem_config = ttnn.MemoryConfig(mem_layout, buffer_type)
        mem_config_input = mem_config
    else:
        mem_config = create_sharded_mem_config(
            ag_output_shape, dim, num_devices, mem_layout, buffer_type, layout, mesh_device, is_input=False
        )
        if mem_config is None:
            pytest.skip(f"No valid shard config for output")
        mem_config_input = create_sharded_mem_config(
            ag_output_shape, dim, num_devices, mem_layout, buffer_type, layout, mesh_device, is_input=True
        )
        if mem_config_input is None:
            pytest.skip(f"No valid shard config for input")

    run_all_gather_impl(
        mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ttnn.bfloat16,
        layout,
        mem_config_input,
        mem_config,
        all_gather_topology=ag_topology,
        enable_trace=enable_trace,
        num_iters=10 if enable_trace else 1,
        use_barrier=use_barrier,
        use_persistent_buffers=False if use_barrier else True,
        skip_check=True if enable_trace else False,
        use_new_ag=use_new_ag,
    )
