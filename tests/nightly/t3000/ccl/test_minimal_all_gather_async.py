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
    #ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value, ttnn.types.BufferType.L1_SMALL) for _ in range(2)]
    return ccl_semaphore_handles


def create_fabric_router_config(max_payload_size: int):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _linear_core_range_set(num_cores, max_cols):
    """Row-major linear core grid: fill rows of max_cols, then overflow to next row."""
    if num_cores <= max_cols:
        return ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))}
        )
    full_rows = num_cores // max_cols
    remainder = num_cores % max_cols
    ranges = set()
    if full_rows > 0:
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(max_cols - 1, full_rows - 1)))
    if remainder > 0:
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(remainder - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def create_sharded_mem_config(shape, gather_dim, num_devices, mem_layout, buffer_type, layout, mesh_device, is_input=False):
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
    use_semaphore_free_all_gather_impl=False,
    sub_core_grids=None,
    use_broadcast=False,
    use_blaze=False,
    use_explicit_subdevice_id=True,
    knobs=None,
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
                    layout=layout,
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
                    layout=layout,
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
    #_, _, _, hidden_dim = ag_output_shape

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
        if use_blaze:
            logger.info(f"Using blaze all-gather")
            tt_all_gather_out_tensor = ttnn.all_gather(
                input_tensor_mesh_list[i],
                dim=dim,
                memory_config=mem_config_ag,
                output_tensor=persistent_output_buffers[i] if use_persistent_buffers else None,
                cluster_axis=cluster_axis)
        elif use_semaphore_free_all_gather_impl and all_gather_function == ttnn.experimental.all_gather_async:
            logger.info(f"Using new all-gather")
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


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim",
    [
        ([1, 1, 256, 96], 2),  # 3 tiles per device
        ([1, 1, 3072, 8192], 2),
    ],
    #ids=["small", "25M"],
)
#@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT], ids=["tile"])
#@pytest.mark.parametrize("use_blaze", [True, False], ids=["blaze", "minimal"])
@pytest.mark.parametrize("use_blaze", [True], ids=["blaze"])
@pytest.mark.parametrize("use_trace", [False])
@pytest.mark.parametrize(
    "mem_layout",
    [
        ttnn.TensorMemoryLayout.INTERLEAVED,
    ],
    ids=["interleaved"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 110000, "fabric_router_config": create_fabric_router_config(15232)}, ttnn.Topology.Ring),
        #({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 110000, "fabric_router_config": create_fabric_router_config(15232)}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["ring"], #, "linear"],
)
def test_ag_quick(mesh_device, ag_output_shape, dim, layout, mem_layout, all_gather_topology, use_blaze, use_trace):
    num_devices = mesh_device.get_num_devices()
    buffer_type = ttnn.BufferType.DRAM

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
        2,  # num_links
        ttnn.bfloat16,
        layout,
        mem_config_input,
        mem_config,
        all_gather_topology=all_gather_topology,
        use_blaze=use_blaze,
        enable_trace=True if use_trace else False,
        num_iters=20 if use_trace else 1,
        use_barrier=True,
        use_persistent_buffers=True,
        use_semaphore_free_all_gather_impl=False,
        skip_check=True if use_trace else False,
        #num_workers_per_link=1,
    )

@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim",
    [
        ([1, 1, 32, 256], 3),       # 8K   - AG sweep baseline
        ([1, 1, 32, 1024], 3),      # 32K  - AG sweep
        #([1, 1, 8, 7168], 3),       # 57K  - DeepSeek V3 decode (emb_dim=7168)
        ([1, 1, 128, 512], 3),      # 65K  - AG sweep ([16, 1, 8, 512], 3)
        ([1, 1, 128, 1024], 3),     # 131K - AG sweep
        ([1, 1, 256, 2048], 3),     # 512K - AG sweep
        ([1, 1, 288, 3072], 3),     # 884K - AG sweep ([24, 3, 128, 96], 0),
        ([1, 1, 352, 2560], 3),     # 901K - SD3.5 prompt (tt_dit ff)
        ####([1, 1, 32, 28672], 3),     # 917K - Llama 70B Galaxy MLP w1/w3 RS
        ([1, 1, 256, 3584], 3),     # 917K - Llama 70B Galaxy MLP w1/w3 RS
        ([1, 1, 2048, 512], 3),     # 1M   - AG sweep ([16, 16, 8, 512], 3),
        ([1, 1, 4096, 512], 3),     # 2M   - AG sweep ([1, 8, 4096, 64], 1),
        ([1, 1, 1024, 2560], 3),    # 2.6M - SD3.5 spatial (tt_dit ff)
        ([1, 1, 1024, 5120], 3),    # 5M   - AG sweep
        ([1, 1, 4096, 2560], 3),    # 10M  - SD3.5 spatial full-res
        ([1, 1, 3072, 8192], 3),    # 25M  - AG sweep
        ([1, 2, 3072, 8192], 3),    # 50M
        ([2, 2, 3072, 8192], 3),    # 100M
        #([2, 8, 3072, 8192], 3),    # 400M
    ],
    ids=["8K", "32K", "65K", "131K", "512K", "884K",
         "901K", "917K", "1M", "2M",
         "2.6M", "5M", "10M", "25M", "50M", "100M"],
)
#@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT], ids=["tile"])
@pytest.mark.parametrize("use_blaze", [True, False], ids=["blaze", "minimal"])
@pytest.mark.parametrize("n_workers", [None], ids=lambda n: f"{n}workers")
@pytest.mark.parametrize("n_chunks", [None], ids=lambda n: f"{n}chunks")
@pytest.mark.parametrize(
    "mem_layout",
    [
        ttnn.TensorMemoryLayout.INTERLEAVED,
    ],
    #ids=["interleaved", "height_sharded", "width_sharded"],
    ids=["interleaved"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 110000, "fabric_router_config": create_fabric_router_config(15232)}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 110000, "fabric_router_config": create_fabric_router_config(15232)}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["ring", "linear"],
    #ids=["ring"],
)
def test_ag_tensor_size(mesh_device, ag_output_shape, dim, layout, mem_layout, all_gather_topology, n_workers, n_chunks, use_blaze):
    num_devices = mesh_device.get_num_devices()
    buffer_type = ttnn.BufferType.DRAM

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
        2,  # num_links
        ttnn.bfloat16,
        layout,
        mem_config_input,
        mem_config,
        all_gather_topology=all_gather_topology,
        enable_trace=True,
        num_iters=20,
        use_barrier=False,
        use_persistent_buffers=True,
        use_semaphore_free_all_gather_impl=False,
        skip_check=True,
        use_blaze=use_blaze,
        chunks_per_sync=n_chunks,
        num_workers_per_link=n_workers,
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim",
    [
        ([8, 64, 512, 32], 0),
        ([8, 32, 512, 64], 0),
        ([8, 16, 512, 128], 0),
        ([8, 8, 512, 256], 0),
        ([8, 4, 512, 512], 0),
        ([8, 2, 512, 1024], 0),
        ([8, 1, 512, 2048], 0),
        ([8, 1, 256, 4096], 0),
    ],
)
#@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT], ids=["rm"])
@pytest.mark.parametrize("use_blaze", [True, False], ids=["blaze", "minimal"])
@pytest.mark.parametrize("n_workers", [None], ids=lambda n: f"{n}workers")
@pytest.mark.parametrize("n_chunks", [None], ids=lambda n: f"{n}chunks")
@pytest.mark.parametrize(
    "mem_layout",
    [
        ttnn.TensorMemoryLayout.INTERLEAVED,
    ],
    #ids=["interleaved", "height_sharded", "width_sharded"],
    ids=["interleaved"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 110000, "fabric_router_config": create_fabric_router_config(15232)}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 110000, "fabric_router_config": create_fabric_router_config(15232)}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["ring", "linear"],
    #ids=["ring"],
)
def test_ag_page_size(mesh_device, ag_output_shape, dim, layout, mem_layout, all_gather_topology, n_workers, n_chunks, use_blaze):
    num_devices = mesh_device.get_num_devices()
    buffer_type = ttnn.BufferType.DRAM

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
        2,  # num_links
        ttnn.bfloat16,
        layout,
        mem_config_input,
        mem_config,
        all_gather_topology=all_gather_topology,
        enable_trace=True,
        num_iters=20,
        use_barrier=False,
        use_persistent_buffers=True,
        use_semaphore_free_all_gather_impl=False,
        skip_check=True,
        use_blaze=use_blaze,
        chunks_per_sync=n_chunks,
        num_workers_per_link=n_workers,
    )



####################################################################################
# TUNING SCAFFOLD (temporary): env-driven knob sweep, so one build + one pytest
# process covers many (shape, knob-tuple) cells.
#
#   AG_SHAPES  comma-separated ids from _SHAPES below         (default "512K,2M,25M")
#   AG_KNOBS   semicolon-separated 8-tuples, e.g. "0,0,0,0,0,0,0,1;2,0,0,0,0,0,0,1"
#              index: 0 wpd 1 pkts/cb 2 run_cap+1 3 mux_slots 4 cb_depth
#                     5 read_ahead 6 signals/stripe 7 factory(1=uni,2=mcast)
#   AG_TOPO    ring | linear | both                           (default "ring")
#   AG_ITERS   profiled iterations                            (default 20)
####################################################################################

import os as _os

# id -> (output_shape, dim, layout)
_SHAPES = {
    # tile, gather on last dim: page is always 2048 B
    "8K":    ([1, 1, 32, 256], 3, ttnn.TILE_LAYOUT),
    "32K":   ([1, 1, 32, 1024], 3, ttnn.TILE_LAYOUT),
    "65K":   ([1, 1, 128, 512], 3, ttnn.TILE_LAYOUT),
    "131K":  ([1, 1, 128, 1024], 3, ttnn.TILE_LAYOUT),
    "262K":  ([1, 1, 256, 1024], 3, ttnn.TILE_LAYOUT),
    "512K":  ([1, 1, 256, 2048], 3, ttnn.TILE_LAYOUT),
    "1M":    ([1, 1, 2048, 512], 3, ttnn.TILE_LAYOUT),
    "2M":    ([1, 1, 4096, 512], 3, ttnn.TILE_LAYOUT),
    "5M":    ([1, 1, 1024, 5120], 3, ttnn.TILE_LAYOUT),
    "10M":   ([1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT),
    "25M":   ([1, 1, 3072, 8192], 3, ttnn.TILE_LAYOUT),
    "50M":   ([1, 2, 3072, 8192], 3, ttnn.TILE_LAYOUT),
    "100M":  ([2, 2, 3072, 8192], 3, ttnn.TILE_LAYOUT),
    # row-major, gather on dim 0: page = last_dim * 2 B, volume held near 8.4 M
    "p64":   ([8, 64, 512, 32], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p128":  ([8, 32, 512, 64], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p256":  ([8, 16, 512, 128], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p512":  ([8, 8, 512, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p1K":   ([8, 4, 512, 512], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p2K":   ([8, 2, 512, 1024], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p4K":   ([8, 1, 512, 2048], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p8K":   ([8, 1, 256, 4096], 0, ttnn.ROW_MAJOR_LAYOUT),
    # Fixed volume, varying stripe length: the size ladder above confounds the two (its 1M/2M
    # entries are 512 wide, i.e. only 2 chunks per stripe, while its 5M+ entries have 10-32).
    # All of these are 10.5 MB of output => 5.2 MB per link; only chunks-per-stripe changes.
    "v5_s2":   ([1, 1, 10240, 512], 3, ttnn.TILE_LAYOUT),
    "v5_s4":   ([1, 1, 5120, 1024], 3, ttnn.TILE_LAYOUT),
    "v5_s8":   ([1, 1, 2560, 2048], 3, ttnn.TILE_LAYOUT),
    "v5_s16":  ([1, 1, 1280, 4096], 3, ttnn.TILE_LAYOUT),
    "v5_s32":  ([1, 1, 640, 8192], 3, ttnn.TILE_LAYOUT),
    "v5_s64":  ([1, 1, 320, 16384], 3, ttnn.TILE_LAYOUT),
    # Same, at 26 MB of output => 13 MB per link.
    "v25_s2":  ([1, 1, 25600, 512], 3, ttnn.TILE_LAYOUT),
    "v25_s8":  ([1, 1, 6400, 2048], 3, ttnn.TILE_LAYOUT),
    "v25_s32": ([1, 1, 1600, 8192], 3, ttnn.TILE_LAYOUT),
    "v25_s4":  ([1, 1, 12800, 1024], 3, ttnn.TILE_LAYOUT),
    "v25_s16": ([1, 1, 3200, 4096], 3, ttnn.TILE_LAYOUT),
    "v25_s64": ([1, 1, 800, 16384], 3, ttnn.TILE_LAYOUT),
    # Volume ladder at a fixed 32-chunk stripe, to locate the workers crossover without the
    # stripe length moving underneath it. Suffix is MB per link.
    "c065":  ([1, 1, 80, 8192], 3, ttnn.TILE_LAYOUT),
    "x079":  ([1, 1, 96, 8192], 3, ttnn.TILE_LAYOUT),
    # 0.5 MB per link at three stripe lengths, to see if the crossover moves with stripe.
    "y_s2":  ([1, 1, 1024, 512], 3, ttnn.TILE_LAYOUT),
    "y_s8":  ([1, 1, 256, 2048], 3, ttnn.TILE_LAYOUT),
    "y_s32": ([1, 1, 64, 8192], 3, ttnn.TILE_LAYOUT),
    # ~2 MB per link at three stripe lengths.
    "z_s2":  ([1, 1, 4096, 512], 3, ttnn.TILE_LAYOUT),
    "z_s8":  ([1, 1, 1024, 2048], 3, ttnn.TILE_LAYOUT),
    "z_s32": ([1, 1, 256, 8192], 3, ttnn.TILE_LAYOUT),
    "x092":  ([1, 1, 112, 8192], 3, ttnn.TILE_LAYOUT),
    "x105":  ([1, 1, 128, 8192], 3, ttnn.TILE_LAYOUT),
    "c13":   ([1, 1, 160, 8192], 3, ttnn.TILE_LAYOUT),
    "c26":   ([1, 1, 320, 8192], 3, ttnn.TILE_LAYOUT),
    "c39":   ([1, 1, 480, 8192], 3, ttnn.TILE_LAYOUT),
    "x34":   ([1, 1, 416, 8192], 3, ttnn.TILE_LAYOUT),
    "x46":   ([1, 1, 544, 8192], 3, ttnn.TILE_LAYOUT),
    "c52":   ([1, 1, 640, 8192], 3, ttnn.TILE_LAYOUT),
    "c79":   ([1, 1, 960, 8192], 3, ttnn.TILE_LAYOUT),
    "c105":  ([1, 1, 1280, 8192], 3, ttnn.TILE_LAYOUT),
    "c131":  ([1, 1, 1600, 8192], 3, ttnn.TILE_LAYOUT),
    "c183":  ([1, 1, 2240, 8192], 3, ttnn.TILE_LAYOUT),
    "c199":  ([1, 1, 2432, 8192], 3, ttnn.TILE_LAYOUT),
    "c220":  ([1, 1, 2688, 8192], 3, ttnn.TILE_LAYOUT),
    "c262":  ([1, 1, 3200, 8192], 3, ttnn.TILE_LAYOUT),
    # Even vs uneven (link*worker) slice splits at matched volume: for a dim-3 gather of a
    # W=8192 tile tensor, num_input_pages == H, so H divisible by 6 splits evenly across
    # 2 links x 3 workers and H = 1280 / 1408 does not.
    "e94":   ([1, 1, 1152, 8192], 3, ttnn.TILE_LAYOUT),
    "e110":  ([1, 1, 1344, 8192], 3, ttnn.TILE_LAYOUT),
    "u115":  ([1, 1, 1408, 8192], 3, ttnn.TILE_LAYOUT),
    "e47":   ([1, 1, 576, 8192], 3, ttnn.TILE_LAYOUT),
    "e63":   ([1, 1, 768, 8192], 3, ttnn.TILE_LAYOUT),
    # Same, at 2.6 MB of output => 1.3 MB per link (below the crossover).
    "v1_s2":   ([1, 1, 2560, 512], 3, ttnn.TILE_LAYOUT),
    "v1_s8":   ([1, 1, 640, 2048], 3, ttnn.TILE_LAYOUT),
    "v1_s32":  ([1, 1, 160, 8192], 3, ttnn.TILE_LAYOUT),
    # small-volume row-major, to separate the page axis from the volume axis
    "s_p512": ([8, 1, 128, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
    "s_p2K":  ([8, 1, 32, 1024], 0, ttnn.ROW_MAJOR_LAYOUT),
    "s_p8K":  ([8, 1, 8, 4096], 0, ttnn.ROW_MAJOR_LAYOUT),
}

_AG_SHAPE_IDS = [s for s in _os.environ.get("AG_SHAPES", "512K,2M,25M").split(",") if s]
_AG_KNOBS = [
    tuple(int(v) for v in spec.split(","))
    for spec in _os.environ.get("AG_KNOBS", "0,0,0,0,0,0,0,1").split(";")
    if spec
]
_AG_ITERS = int(_os.environ.get("AG_ITERS", "20"))
_AG_TOPO = _os.environ.get("AG_TOPO", "ring")

_TOPOS = {
    "ring": (
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200000,
         "fabric_router_config": create_fabric_router_config(15232)},
        ttnn.Topology.Ring,
    ),
    "linear": (
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200000,
         "fabric_router_config": create_fabric_router_config(15232)},
        ttnn.Topology.Linear,
    ),
}
_AG_TOPO_IDS = ["ring", "linear"] if _AG_TOPO == "both" else [_AG_TOPO]


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("shape_id", _AG_SHAPE_IDS)
@pytest.mark.parametrize("knobs", _AG_KNOBS, ids=lambda k: "k" + "_".join(str(v) for v in k))
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [_TOPOS[t] for t in _AG_TOPO_IDS],
    indirect=["device_params"],
    ids=_AG_TOPO_IDS,
)
def test_ag_knobs(mesh_device, shape_id, knobs, device_params, all_gather_topology):
    ag_output_shape, dim, layout = _SHAPES[shape_id]
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    # AG_CHECK=1 swaps the perf loop for the functional path (no trace, device skew injected,
    # PCC compared) so the same shape/knob matrix can be used to check correctness.
    check = _os.environ.get("AG_CHECK") == "1"

    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        ag_output_shape,
        dim,
        2,  # num_links (ignored by the blaze path; it auto-detects)
        ttnn.bfloat16,
        layout,
        mem_config,
        mem_config,
        all_gather_topology=all_gather_topology,
        enable_trace=not check,
        num_iters=1 if check else _AG_ITERS,
        use_barrier=False,
        use_persistent_buffers=True,
        use_semaphore_free_all_gather_impl=False,
        skip_check=not check,
        use_blaze=True,
        knobs=knobs,
    )


####################################################################################
# Page-size sweep that all three implementations can run natively.
#
# test_ag_page_size varies the page by using row-major with different row widths, but
# ttnn.experimental.all_gather_async has no native row-major path and falls back to a
# composite (AllBroadcast + Concat), so it cannot be compared there. A tile page is
# 32x32 elements, so its size follows the dtype instead: bfloat8_b 1088 B, bfloat16
# 2048 B, float32 4096 B. Only three points, but every implementation runs them
# natively. Shapes are paired with the dtype to hold the output near 16.8 MB, so the
# page is the only thing that moves.
####################################################################################

@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, ag_dtype",
    [
        ([1, 1, 2048, 8192], ttnn.bfloat8_b),  # 1088 B page, 17.8 MB
        ([1, 1, 2048, 4096], ttnn.bfloat16),   # 2048 B page, 16.8 MB
        ([1, 1, 2048, 2048], ttnn.float32),    # 4096 B page, 16.8 MB
    ],
    ids=["p1088", "p2048", "p4096"],
)
@pytest.mark.parametrize("use_blaze", [True, False], ids=["blaze", "minimal"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200000,
          "fabric_router_config": create_fabric_router_config(15232)}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200000,
          "fabric_router_config": create_fabric_router_config(15232)}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["ring", "linear"],
)
def test_ag_page_size_tile(mesh_device, ag_output_shape, ag_dtype, all_gather_topology, use_blaze):
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        ag_output_shape,
        3,  # gather on the last dim
        2,  # num_links
        ag_dtype,
        ttnn.TILE_LAYOUT,
        mem_config,
        mem_config,
        all_gather_topology=all_gather_topology,
        enable_trace=True,
        num_iters=20,
        use_barrier=False,
        use_persistent_buffers=True,
        use_semaphore_free_all_gather_impl=False,
        skip_check=True,
        use_blaze=use_blaze,
    )
