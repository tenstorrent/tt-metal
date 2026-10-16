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
    use_semaphore_free_all_gather_impl=False,
    sub_core_grids=None,
    use_broadcast=False,
    use_blaze=False,
    use_explicit_subdevice_id=True,
):
    use_sub_devices = False
    torch.manual_seed(0)

    tile = (32, 32)

    mesh_shape = tuple(mesh_device.shape)
    replicate = mesh_shape[cluster_axis] if cluster_axis is not None else num_devices

    # Skip unsupported cases
    is_known_failure, message = is_unsupported_case(
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
    # _, _, _, hidden_dim = ag_output_shape

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
                cluster_axis=cluster_axis,
            )
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
        # ([1, 1, 256, 96], 2),  # 3 tiles per device
        ([1, 1, 3072, 8192], 2),
    ],
    # ids=["small", "25M"],
)
# @pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT], ids=["tile"])
# @pytest.mark.parametrize("use_blaze", [True, False], ids=["blaze", "minimal"])
@pytest.mark.parametrize("use_blaze", [True], ids=["blaze"])
@pytest.mark.parametrize("use_trace", [True])
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
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 110000,
                "fabric_router_config": create_fabric_router_config(15232),
            },
            ttnn.Topology.Ring,
        ),
        # ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 110000, "fabric_router_config": create_fabric_router_config(15232)}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["ring"],  # , "linear"],
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
        ttnn.bfloat8_b,
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
        # num_workers_per_link=1,
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "ag_output_shape, dim",
    [
        ([1, 1, 32, 256], 3),  # 8K   - AG sweep baseline
        ([1, 1, 32, 1024], 3),  # 32K  - AG sweep
        # ([1, 1, 8, 7168], 3),       # 57K  - DeepSeek V3 decode (emb_dim=7168)
        ([1, 1, 128, 512], 3),  # 65K  - AG sweep ([16, 1, 8, 512], 3)
        ([1, 1, 128, 1024], 3),  # 131K - AG sweep
        ([1, 1, 256, 2048], 3),  # 512K - AG sweep
        ([1, 1, 288, 3072], 3),  # 884K - AG sweep ([24, 3, 128, 96], 0),
        ([1, 1, 352, 2560], 3),  # 901K - SD3.5 prompt (tt_dit ff)
        ####([1, 1, 32, 28672], 3),     # 917K - Llama 70B Galaxy MLP w1/w3 RS
        ([1, 1, 256, 3584], 3),  # 917K - Llama 70B Galaxy MLP w1/w3 RS
        ([1, 1, 2048, 512], 3),  # 1M   - AG sweep ([16, 16, 8, 512], 3),
        ([1, 1, 4096, 512], 3),  # 2M   - AG sweep ([1, 8, 4096, 64], 1),
        ([1, 1, 1024, 2560], 3),  # 2.6M - SD3.5 spatial (tt_dit ff)
        ([1, 1, 1024, 5120], 3),  # 5M   - AG sweep
        ([1, 1, 4096, 2560], 3),  # 10M  - SD3.5 spatial full-res
        ([1, 1, 3072, 8192], 3),  # 25M  - AG sweep
        ([1, 2, 3072, 8192], 3),  # 50M
        ([2, 2, 3072, 8192], 3),  # 100M
        # ([2, 8, 3072, 8192], 3),    # 400M
    ],
    ids=[
        "8K",
        "32K",
        "65K",
        "131K",
        "512K",
        "884K",
        "901K",
        "917K",
        "1M",
        "2M",
        "2.6M",
        "5M",
        "10M",
        "25M",
        "50M",
        "100M",
    ],
)
# @pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT], ids=["tile"])
@pytest.mark.parametrize("use_blaze", [True], ids=["blaze"])
@pytest.mark.parametrize("n_workers", [None], ids=lambda n: f"{n}workers")
@pytest.mark.parametrize("n_chunks", [None], ids=lambda n: f"{n}chunks")
@pytest.mark.parametrize(
    "mem_layout",
    [
        ttnn.TensorMemoryLayout.INTERLEAVED,
    ],
    # ids=["interleaved", "height_sharded", "width_sharded"],
    ids=["interleaved"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 110000,
                "fabric_router_config": create_fabric_router_config(15232),
            },
            ttnn.Topology.Ring,
        ),
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 110000,
                "fabric_router_config": create_fabric_router_config(15232),
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["ring", "linear"],
    # ids=["ring"],
)
def test_ag_tensor_size(
    mesh_device, ag_output_shape, dim, layout, mem_layout, all_gather_topology, n_workers, n_chunks, use_blaze
):
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
    ids=["p64", "p128", "p256", "p512", "p1K", "p2K", "p4K", "p8K"],
)
# @pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT], ids=["rm"])
@pytest.mark.parametrize("use_blaze", [True], ids=["blaze"])
@pytest.mark.parametrize("n_workers", [None], ids=lambda n: f"{n}workers")
@pytest.mark.parametrize("n_chunks", [None], ids=lambda n: f"{n}chunks")
@pytest.mark.parametrize(
    "mem_layout",
    [
        ttnn.TensorMemoryLayout.INTERLEAVED,
    ],
    # ids=["interleaved", "height_sharded", "width_sharded"],
    ids=["interleaved"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 110000,
                "fabric_router_config": create_fabric_router_config(15232),
            },
            ttnn.Topology.Ring,
        ),
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 110000,
                "fabric_router_config": create_fabric_router_config(15232),
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["ring", "linear"],
    # ids=["ring"],
)
def test_ag_page_size(
    mesh_device, ag_output_shape, dim, layout, mem_layout, all_gather_topology, n_workers, n_chunks, use_blaze
):
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
#   AG_SHAPES      comma-separated ids from _SHAPES below     (default "512K,2M,25M")
#   AG_KNOBS       ';'-separated TT_AG_TUNING strings, one test case each (default "" =
#                  one case with the op's defaults). An empty entry (or "default") is the
#                  defaults case. Each entry is "key=value,key=value" with the keys of
#                  ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_tuning.hpp:
#                    factory=multicast|unicast  k=auto|<power of 2>  xfer=legacy|segments
#                    rotation=none|absolute|absolute_dir_stagger
#                    entry=legacy|scatter_aware|one_packet  padded_slot=0|1  run_cap=<bytes>
#                    workers=<n>  signals=<n>  mcast_cb_depth=<n>  packet=<bytes>
#                  e.g. AG_KNOBS="factory=unicast;factory=unicast,rotation=absolute;"
#   AG_KNOBS_FILE  path to a file with one tuning string per line ('#' comments and blank
#                  lines ignored; write "default" for the defaults case). Its entries are
#                  appended to AG_KNOBS's when both are set.
#   AG_TOPO        ring | linear | both                       (default "ring")
#   AG_ITERS       profiled iterations                        (default 20)
#   AG_PACKET      fabric router max packet payload bytes     (default 15232)
#   AG_CHECK       1 = functional path (no trace, PCC checked) instead of perf
#
# The op reads TT_AG_TUNING on every call and the parsed struct is part of the program-cache
# hash, so test_ag_knobs sets it for the whole case (compile run, trace capture, replays,
# readback) and restores the previous value afterwards.
####################################################################################

import contextlib as _contextlib
import os as _os

# id -> (output_shape, dim, layout)
_SHAPES = {
    # tile, gather on last dim: page is always 2048 B
    "8K": ([1, 1, 32, 256], 3, ttnn.TILE_LAYOUT),
    "32K": ([1, 1, 32, 1024], 3, ttnn.TILE_LAYOUT),
    "65K": ([1, 1, 128, 512], 3, ttnn.TILE_LAYOUT),
    "131K": ([1, 1, 128, 1024], 3, ttnn.TILE_LAYOUT),
    "262K": ([1, 1, 256, 1024], 3, ttnn.TILE_LAYOUT),
    "512K": ([1, 1, 256, 2048], 3, ttnn.TILE_LAYOUT),
    "1M": ([1, 1, 2048, 512], 3, ttnn.TILE_LAYOUT),
    "2M": ([1, 1, 4096, 512], 3, ttnn.TILE_LAYOUT),
    "5M": ([1, 1, 1024, 5120], 3, ttnn.TILE_LAYOUT),
    "10M": ([1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT),
    "25M": ([1, 1, 3072, 8192], 3, ttnn.TILE_LAYOUT),
    "50M": ([1, 2, 3072, 8192], 3, ttnn.TILE_LAYOUT),
    "100M": ([2, 2, 3072, 8192], 3, ttnn.TILE_LAYOUT),
    # row-major, gather on dim 0: page = last_dim * 2 B, volume held near 8.4 M
    "p64": ([8, 64, 512, 32], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p128": ([8, 32, 512, 64], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p256": ([8, 16, 512, 128], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p512": ([8, 8, 512, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p1K": ([8, 4, 512, 512], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p2K": ([8, 2, 512, 1024], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p4K": ([8, 1, 512, 2048], 0, ttnn.ROW_MAJOR_LAYOUT),
    "p8K": ([8, 1, 256, 4096], 0, ttnn.ROW_MAJOR_LAYOUT),
    # Fixed volume, varying stripe length: the size ladder above confounds the two (its 1M/2M
    # entries are 512 wide, i.e. only 2 chunks per stripe, while its 5M+ entries have 10-32).
    # All of these are 10.5 MB of output => 5.2 MB per link; only chunks-per-stripe changes.
    "v5_s2": ([1, 1, 10240, 512], 3, ttnn.TILE_LAYOUT),
    "v5_s4": ([1, 1, 5120, 1024], 3, ttnn.TILE_LAYOUT),
    "v5_s8": ([1, 1, 2560, 2048], 3, ttnn.TILE_LAYOUT),
    "v5_s16": ([1, 1, 1280, 4096], 3, ttnn.TILE_LAYOUT),
    "v5_s32": ([1, 1, 640, 8192], 3, ttnn.TILE_LAYOUT),
    "v5_s64": ([1, 1, 320, 16384], 3, ttnn.TILE_LAYOUT),
    # Same, at 26 MB of output => 13 MB per link.
    "v25_s2": ([1, 1, 25600, 512], 3, ttnn.TILE_LAYOUT),
    "v25_s8": ([1, 1, 6400, 2048], 3, ttnn.TILE_LAYOUT),
    "v25_s32": ([1, 1, 1600, 8192], 3, ttnn.TILE_LAYOUT),
    "v25_s4": ([1, 1, 12800, 1024], 3, ttnn.TILE_LAYOUT),
    "v25_s16": ([1, 1, 3200, 4096], 3, ttnn.TILE_LAYOUT),
    "v25_s64": ([1, 1, 800, 16384], 3, ttnn.TILE_LAYOUT),
    # ~21 MB output, stripe = (W/8)/32 held at 5 / 10 / 12 / 20 to test whether the packets
    # preference tracks stripe length. The earlier ladder used powers of two only and so
    # skipped stripe 10, which is where the two remaining regressions live.
    "q_s5": ([1, 1, 8192, 1280], 3, ttnn.TILE_LAYOUT),
    "q_s10": ([1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT),
    "q_s12": ([1, 1, 3392, 3072], 3, ttnn.TILE_LAYOUT),
    "q_s20": ([1, 1, 2048, 5120], 3, ttnn.TILE_LAYOUT),
    "q_s8": ([1, 1, 5120, 2048], 3, ttnn.TILE_LAYOUT),
    "q_s14": ([1, 1, 2912, 3584], 3, ttnn.TILE_LAYOUT),
    "q_s16": ([1, 1, 2560, 4096], 3, ttnn.TILE_LAYOUT),
    "q_s24": ([1, 1, 1696, 6144], 3, ttnn.TILE_LAYOUT),
    # Volume ladder at a fixed 32-chunk stripe, to locate the workers crossover without the
    # stripe length moving underneath it. Suffix is MB per link.
    "c065": ([1, 1, 80, 8192], 3, ttnn.TILE_LAYOUT),
    "x079": ([1, 1, 96, 8192], 3, ttnn.TILE_LAYOUT),
    # 0.5 MB per link at three stripe lengths, to see if the crossover moves with stripe.
    "y_s2": ([1, 1, 1024, 512], 3, ttnn.TILE_LAYOUT),
    "y_s8": ([1, 1, 256, 2048], 3, ttnn.TILE_LAYOUT),
    "y_s32": ([1, 1, 64, 8192], 3, ttnn.TILE_LAYOUT),
    # ~2 MB per link at three stripe lengths.
    "z_s2": ([1, 1, 4096, 512], 3, ttnn.TILE_LAYOUT),
    "z_s8": ([1, 1, 1024, 2048], 3, ttnn.TILE_LAYOUT),
    "z_s32": ([1, 1, 256, 8192], 3, ttnn.TILE_LAYOUT),
    "x092": ([1, 1, 112, 8192], 3, ttnn.TILE_LAYOUT),
    "x105": ([1, 1, 128, 8192], 3, ttnn.TILE_LAYOUT),
    "c13": ([1, 1, 160, 8192], 3, ttnn.TILE_LAYOUT),
    "c26": ([1, 1, 320, 8192], 3, ttnn.TILE_LAYOUT),
    "c39": ([1, 1, 480, 8192], 3, ttnn.TILE_LAYOUT),
    "x34": ([1, 1, 416, 8192], 3, ttnn.TILE_LAYOUT),
    "x46": ([1, 1, 544, 8192], 3, ttnn.TILE_LAYOUT),
    "c52": ([1, 1, 640, 8192], 3, ttnn.TILE_LAYOUT),
    "c79": ([1, 1, 960, 8192], 3, ttnn.TILE_LAYOUT),
    "c105": ([1, 1, 1280, 8192], 3, ttnn.TILE_LAYOUT),
    "c131": ([1, 1, 1600, 8192], 3, ttnn.TILE_LAYOUT),
    "c183": ([1, 1, 2240, 8192], 3, ttnn.TILE_LAYOUT),
    "c199": ([1, 1, 2432, 8192], 3, ttnn.TILE_LAYOUT),
    "c220": ([1, 1, 2688, 8192], 3, ttnn.TILE_LAYOUT),
    "c262": ([1, 1, 3200, 8192], 3, ttnn.TILE_LAYOUT),
    # Even vs uneven (link*worker) slice splits at matched volume: for a dim-3 gather of a
    # W=8192 tile tensor, num_input_pages == H, so H divisible by 6 splits evenly across
    # 2 links x 3 workers and H = 1280 / 1408 does not.
    "e94": ([1, 1, 1152, 8192], 3, ttnn.TILE_LAYOUT),
    "e110": ([1, 1, 1344, 8192], 3, ttnn.TILE_LAYOUT),
    "u115": ([1, 1, 1408, 8192], 3, ttnn.TILE_LAYOUT),
    "e47": ([1, 1, 576, 8192], 3, ttnn.TILE_LAYOUT),
    "e63": ([1, 1, 768, 8192], 3, ttnn.TILE_LAYOUT),
    # Same, at 2.6 MB of output => 1.3 MB per link (below the crossover).
    "v1_s2": ([1, 1, 2560, 512], 3, ttnn.TILE_LAYOUT),
    "v1_s8": ([1, 1, 640, 2048], 3, ttnn.TILE_LAYOUT),
    "v1_s32": ([1, 1, 160, 8192], 3, ttnn.TILE_LAYOUT),
    # small-volume row-major, to separate the page axis from the volume axis
    "s_p512": ([8, 1, 128, 256], 0, ttnn.ROW_MAJOR_LAYOUT),
    "s_p2K": ([8, 1, 32, 1024], 0, ttnn.ROW_MAJOR_LAYOUT),
    "s_p8K": ([8, 1, 8, 4096], 0, ttnn.ROW_MAJOR_LAYOUT),
}

_AG_SHAPE_IDS = [s for s in _os.environ.get("AG_SHAPES", "512K,2M,25M").split(",") if s]
# Mirror of Tuning::parse (all_gather_tuning.cpp), so a typo fails at collection time instead of
# after the mesh is opened. Order = Tuning::attribute_names, so equal tunings get equal strings.
_AG_TUNING_KEYS = {
    "factory": ("multicast", "unicast"),
    "k": None,  # "auto" or a power of 2
    "xfer": ("legacy", "segments"),
    "rotation": ("none", "absolute", "absolute_dir_stagger"),
    "entry": ("legacy", "scatter_aware", "one_packet"),
    "padded_slot": ("0", "1", "true", "false"),
    "run_cap": None,
    "workers": None,
    "signals": None,
    "mcast_cb_depth": None,
    "packet": None,
}


def _canonical_ag_tuning(spec):
    """'rotation = absolute, factory=unicast' -> 'factory=unicast,rotation=absolute'; '' / 'default' -> ''."""
    spec = spec.strip()
    if spec.lower() == "default":
        return ""
    kv = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"AG_KNOBS: '{item}' in '{spec}' is not key=value")
        k, v = (s.strip() for s in item.split("=", 1))
        if k not in _AG_TUNING_KEYS:
            raise ValueError(f"AG_KNOBS: unknown key '{k}' in '{spec}' (known: {', '.join(_AG_TUNING_KEYS)})")
        allowed = _AG_TUNING_KEYS[k]
        if (allowed is not None and v not in allowed) or (
            allowed is None and not (v.isdigit() or (k == "k" and v == "auto"))
        ):
            want = "|".join(allowed) if allowed else ("auto|<uint>" if k == "k" else "<uint>")
            raise ValueError(f"AG_KNOBS: {k}={v} in '{spec}' (want {want})")
        if k in kv:
            raise ValueError(f"AG_KNOBS: key '{k}' repeated in '{spec}'")
        kv[k] = v
    return ",".join(f"{k}={kv[k]}" for k in _AG_TUNING_KEYS if k in kv)


def _ag_tuning_id(spec):
    """Test-id form of a canonical tuning string: 'factory=unicast,run_cap=0' -> 't_factory-unicast+run_cap-0'."""
    return "t_" + (spec.replace("=", "-").replace(",", "+") if spec else "default")


def _load_ag_knobs():
    entries = []
    if "AG_KNOBS" in _os.environ or "AG_KNOBS_FILE" not in _os.environ:
        entries += _os.environ.get("AG_KNOBS", "").split(";")
    knobs_file = _os.environ.get("AG_KNOBS_FILE")
    if knobs_file:
        with open(knobs_file) as fh:
            lines = [ln.split("#", 1)[0].strip() for ln in fh]
        entries += [ln for ln in lines if ln]
    out = []
    for e in entries:
        c = _canonical_ag_tuning(e)
        if c not in out:
            out.append(c)
    return out


@_contextlib.contextmanager
def _ag_tuning_env(spec):
    """Set TT_AG_TUNING to spec ('' -> unset) for the duration, then restore the previous value."""
    prev = _os.environ.get("TT_AG_TUNING")
    if spec:
        _os.environ["TT_AG_TUNING"] = spec
    else:
        _os.environ.pop("TT_AG_TUNING", None)
    try:
        yield
    finally:
        if prev is None:
            _os.environ.pop("TT_AG_TUNING", None)
        else:
            _os.environ["TT_AG_TUNING"] = prev


_AG_KNOBS = _load_ag_knobs()
_AG_ITERS = int(_os.environ.get("AG_ITERS", "20"))
_AG_TOPO = _os.environ.get("AG_TOPO", "ring")
_AG_PACKET = int(_os.environ.get("AG_PACKET", "15232"))

_TOPOS = {
    "ring": (
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 200000,
            "fabric_router_config": create_fabric_router_config(_AG_PACKET),
        },
        ttnn.Topology.Ring,
    ),
    "linear": (
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 200000,
            "fabric_router_config": create_fabric_router_config(_AG_PACKET),
        },
        ttnn.Topology.Linear,
    ),
}
_AG_TOPO_IDS = ["ring", "linear"] if _AG_TOPO == "both" else [_AG_TOPO]


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("shape_id", _AG_SHAPE_IDS)
@pytest.mark.parametrize("ag_tuning", _AG_KNOBS, ids=_ag_tuning_id)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [_TOPOS[t] for t in _AG_TOPO_IDS],
    indirect=["device_params"],
    ids=_AG_TOPO_IDS,
)
def test_ag_knobs(mesh_device, shape_id, ag_tuning, device_params, all_gather_topology):
    ag_output_shape, dim, layout = _SHAPES[shape_id]
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    # AG_CHECK=1 swaps the perf loop for the functional path (no trace, device skew injected,
    # PCC compared) so the same shape/knob matrix can be used to check correctness.
    check = _os.environ.get("AG_CHECK") == "1"

    logger.info(f"TT_AG_TUNING={ag_tuning!r} ({'defaults' if not ag_tuning else 'override'})")
    with _ag_tuning_env(ag_tuning):
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
        )


####################################################################################
# Device cases for the walk-v2 step (functional only; every case compares bit-exactly).
#   concat_relay_32B_*   concat: 32 B chunks into a packed 256 B DRAM output page on the 8-device line.
#                        A unicast relay would read 32 B chunks back from DRAM at 32 B offsets, which
#                        breaks the BH NoC read congruence ((l1 ^ dram) & 63 == 0). Defaults route it to
#                        multicast; forced unicast must TT_FATAL (relay_read_ok).
#   bfp8_16x32_*         16x32 bfloat8_b tiles (544 B page, 576 B DRAM aligned page) DRAM -> DRAM.
#   ring_ragged_*        even ring, row-major, 997 pages per device: per-slice chunk counts are not a
#                        multiple of xfer * stride, so slices end in a ragged unrotated tail.
#   nd_sharded_*         ND-sharded output (3 non-unit shard dims) on both factories.
#   persistent_twice_*   two calls with persistent outputs (and inputs) at different addresses: the
#                        second is a program-cache hit, so only override_runtime_arguments moves it.
####################################################################################

_LINE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "trace_region_size": 200000,
    "fabric_router_config": create_fabric_router_config(_AG_PACKET),
}
_RING_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "trace_region_size": 200000,
    "fabric_router_config": create_fabric_router_config(_AG_PACKET),
}
_L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
_DRAM_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def _nd_sharded(buffer_type, shard_shape, num_cores=8):
    # A DRAM bank id is the core's x coordinate, so a DRAM grid must be a single row.
    grid_extent = ttnn.CoreCoord(8, 1) if buffer_type == ttnn.BufferType.DRAM else ttnn.CoreCoord(8, 8)
    grid = ttnn.num_cores_to_corerangeset(num_cores, grid_extent, row_wise=True)
    return ttnn.MemoryConfig(
        buffer_type,
        ttnn.NdShardSpec(
            ttnn.Shape(shard_shape),
            grid,
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def _ag_case_input(mesh_device, full, dim, layout, dtype, mem_config, tile=None):
    kwargs = {"tile": tile} if tile is not None else {}
    tt_in = ttnn.from_torch(
        full,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=dim),
        **kwargs,
    )
    # What the devices actually hold (bfloat8_b is lossy on the way in); the gather must reproduce it exactly.
    expected = ttnn.to_torch(tt_in, mesh_composer=ConcatMeshToTensor(mesh_device, dim=dim))
    return tt_in, expected


def _ag_case_check(tt_out, expected, label):
    for d, dev_out in enumerate(ttnn.get_device_tensors(tt_out)):
        got = ttnn.to_torch(dev_out)
        _, pcc = comp_pcc(got, expected, 1.0)
        logger.info(f"{label}: device {d} {pcc}")
        assert got.shape == expected.shape and torch.equal(got, expected), f"{label}: device {d} mismatch ({pcc})"


def _ag_case_simple(mesh_device, tuning, full_shape, dim, layout, dtype, in_mc, out_mc, tile=None):
    torch.manual_seed(0)
    full = torch.randn(full_shape, dtype=torch.float32).to(torch.bfloat16)
    tt_in, expected = _ag_case_input(mesh_device, full, dim, layout, dtype, in_mc, tile)
    with _ag_tuning_env(tuning):
        tt_out = ttnn.all_gather(tt_in, dim=dim, memory_config=out_mc)
    ttnn.synchronize_device(mesh_device)
    _ag_case_check(tt_out, expected, f"TT_AG_TUNING={tuning!r}")


def _ag_case_persistent_twice(mesh_device, tuning):
    full_shape, dim = [1, 1, 512, 8 * 512], 3
    outs = []
    with _ag_tuning_env(tuning):
        for call in range(2):
            torch.manual_seed(call)
            full = torch.randn(full_shape, dtype=torch.float32).to(torch.bfloat16)
            tt_in, expected = _ag_case_input(mesh_device, full, dim, ttnn.TILE_LAYOUT, ttnn.bfloat16, _DRAM_INTERLEAVED)
            if call == 1:
                # Shift the allocator so the second input and output land at new addresses.
                spacer = ttnn.from_torch(
                    torch.zeros([1, 1, 96, 1024], dtype=torch.bfloat16),
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=_DRAM_INTERLEAVED,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            persistent = ttnn.from_torch(
                torch.zeros(full_shape, dtype=torch.bfloat16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=_DRAM_INTERLEAVED,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            entries_before = mesh_device.num_program_cache_entries()
            tt_out = ttnn.all_gather(tt_in, dim=dim, output_tensor=persistent)
            ttnn.synchronize_device(mesh_device)
            entries_after = mesh_device.num_program_cache_entries()
            logger.info(
                f"call {call}: output buffer 0x{persistent.buffer_address():x}, cache {entries_before}->{entries_after}"
            )
            outs.append((tt_out, persistent, expected, persistent.buffer_address(), entries_after - entries_before))
    assert outs[0][3] != outs[1][3], "the two persistent outputs must be at different addresses"
    assert outs[1][4] == 0, "the second call must be a program-cache hit (override_runtime_arguments)"
    for call, (tt_out, persistent, expected, _, _) in enumerate(outs):
        _ag_case_check(persistent, expected, f"TT_AG_TUNING={tuning!r} call {call}")


def _ag_case_expect_fatal(expect_error, message, runner, *args):
    with expect_error(RuntimeError, message):
        runner(*args)


# id -> (device_params, runner); runner(mesh_device, expect_error) gets the conftest expect_error fixture
_AG_CASES = {
    # Defaults: the relay rule (relay_read_ok) routes this to multicast, which never reads the output back.
    "concat_relay_32B_default": (
        _LINE_PARAMS,
        lambda m, ee: _ag_case_simple(
            m,
            "",
            [1, 1, 4096, 8 * 16],
            3,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            _L1_INTERLEAVED,
            _DRAM_INTERLEAVED,
        ),
    ),
    # Forced unicast cannot relay it: routing must TT_FATAL clearly instead of running and corrupting.
    "concat_relay_32B_unicast": (
        _LINE_PARAMS,
        lambda m, ee: _ag_case_expect_fatal(
            ee,
            "TT_AG_TUNING factory=unicast: all_gather unicast cannot relay this output",
            _ag_case_simple,
            m,
            "factory=unicast",
            [1, 1, 4096, 8 * 16],
            3,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.bfloat16,
            _L1_INTERLEAVED,
            _DRAM_INTERLEAVED,
        ),
    ),
    "bfp8_16x32_default": (
        _RING_PARAMS,
        lambda m, ee: _ag_case_simple(
            m,
            "",
            [1, 1, 256, 8 * 512],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat8_b,
            _DRAM_INTERLEAVED,
            _DRAM_INTERLEAVED,
            tile=ttnn.Tile((16, 32)),
        ),
    ),
    "bfp8_16x32_multicast_padded_slot": (
        _RING_PARAMS,
        lambda m, ee: _ag_case_simple(
            m,
            "factory=multicast,padded_slot=1",
            [1, 1, 256, 8 * 512],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat8_b,
            _DRAM_INTERLEAVED,
            _DRAM_INTERLEAVED,
            tile=ttnn.Tile((16, 32)),
        ),
    ),
}
for _page in (1024, 2048):  # 2 KB and 4 KB row-major pages (k=auto gives k=2 at 4 KB on BH)
    for _tuning in (
        "factory=unicast",
        "factory=unicast,xfer=segments",
        "factory=unicast,k=auto,rotation=absolute_dir_stagger,entry=one_packet,xfer=segments",
    ):
        _AG_CASES[f"ring_ragged_p{_page * 2 // 1024}K_{_ag_tuning_id(_tuning)}"] = (
            _RING_PARAMS,
            lambda m, ee, p=_page, t=_tuning: _ag_case_simple(
                m, t, [8, 1, 997, p], 0, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, _DRAM_INTERLEAVED, _DRAM_INTERLEAVED
            ),
        )
for _bt, _bname in ((ttnn.BufferType.L1, "l1"), (ttnn.BufferType.DRAM, "dram")):
    for _tuning in ("factory=unicast", "factory=multicast"):
        _AG_CASES[f"nd_sharded_{_bname}_{_ag_tuning_id(_tuning)}"] = (
            _LINE_PARAMS,
            lambda m, ee, bt=_bt, t=_tuning: _ag_case_simple(
                m,
                t,
                [2, 8, 128, 256],
                1,
                ttnn.TILE_LAYOUT,
                ttnn.bfloat16,
                _DRAM_INTERLEAVED,
                _nd_sharded(bt, [1, 2, 64, 128]),
            ),
        )
for _tuning in ("factory=unicast", "factory=multicast"):
    _AG_CASES[f"persistent_twice_{_ag_tuning_id(_tuning)}"] = (
        _RING_PARAMS,
        lambda m, ee, t=_tuning: _ag_case_persistent_twice(m, t),
    )

_AG_CASE_XFAIL = {}


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params, case_id",
    [pytest.param(p, cid, id=cid, marks=_AG_CASE_XFAIL.get(cid, ())) for cid, (p, _) in _AG_CASES.items()],
    indirect=["device_params"],
)
def test_ag_cases(mesh_device, device_params, case_id, expect_error):
    _AG_CASES[case_id][1](mesh_device, expect_error)


####################################################################################
# Unicast relay routing (route_all_gather + relay_read_ok), on 3- and 4-device sub-lines of the
# 8-device line. Blackhole alignments: DRAM 64 B, L1 16 B (on Wormhole, 32 B DRAM, concat_32B is
# relay-legal and runs unicast).
#   matched_32B_l1_to_dram       one chunk per output page, padded page: relayed through the padded CB slot
#   split_32B_out_page           likewise, split into a DRAM width-sharded output
#   concat_48B_page              concat into a padded, shared output page: no relay possible
#   concat_32B_l1_to_128B_dram   packed (128 == 4 x 32) but chunk 1 starts at +32, not a multiple of 64
# TT_AG_TUNING is read every time the op's params are built, so _ag_tuning_env applies per call.
####################################################################################


def _relay_cases():
    from tests.nightly.t3000.ccl.test_all_gather import _dram_width_sharded

    # name -> (num_devices, gathered shape, dim, input memory config, output memory config)
    return {
        "matched_32B_l1_to_dram": (4, [4, 1, 32768, 16], 0, _L1_INTERLEAVED, _DRAM_INTERLEAVED),
        "split_32B_out_page": (4, [1, 1, 65536, 32], 2, _L1_INTERLEAVED, _dram_width_sharded(65536, 16, 2)),
        "concat_48B_page": (3, [1, 1, 32768, 24], 3, _L1_INTERLEAVED, _DRAM_INTERLEAVED),
        "concat_32B_l1_to_128B_dram": (4, [1, 1, 32768, 64], 3, _L1_INTERLEAVED, _DRAM_INTERLEAVED),
    }


_RELAY_FATAL = "TT_AG_TUNING factory=unicast: all_gather unicast cannot relay this output"
# (case, TT_AG_TUNING, expected error substring or None)
_RELAY_MATRIX = [
    # heuristic: unicast where relay_ok (now including the padded-slot cases), else multicast
    ("matched_32B_l1_to_dram", "", None),
    ("split_32B_out_page", "", None),
    ("concat_48B_page", "", None),
    ("concat_32B_l1_to_128B_dram", "", None),
    # forced unicast: legal for the padded one-chunk-per-page outputs; a clear TT_FATAL otherwise
    ("matched_32B_l1_to_dram", "factory=unicast", None),
    ("split_32B_out_page", "factory=unicast", None),
    ("concat_48B_page", "factory=unicast", _RELAY_FATAL),
    ("concat_32B_l1_to_128B_dram", "factory=unicast", _RELAY_FATAL),
    # forced multicast: always legal on a straight 1D line
    ("matched_32B_l1_to_dram", "factory=multicast", None),
    ("concat_32B_l1_to_128B_dram", "factory=multicast", None),
    # padded slot + other walk knobs on unicast
    ("matched_32B_l1_to_dram", "factory=unicast,xfer=segments", None),
    ("matched_32B_l1_to_dram", "factory=unicast,entry=one_packet", None),
    ("split_32B_out_page", "factory=unicast,rotation=absolute_dir_stagger", None),
]


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [_LINE_PARAMS], indirect=True, ids=["line"])
@pytest.mark.parametrize(
    "case, ag_tuning, error",
    _RELAY_MATRIX,
    ids=[f"{c}-{_ag_tuning_id(t)}" for c, t, _ in _RELAY_MATRIX],
)
def test_ag_relay_routing(mesh_device, device_params, case, ag_tuning, error, expect_error):
    if ttnn.get_arch_name() != "blackhole":
        pytest.skip("expectations assume Blackhole's 64 B DRAM read alignment")
    num_devices, full_shape, dim, in_mc, out_mc = _relay_cases()[case]
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, num_devices))
    args = (submesh, ag_tuning, full_shape, dim, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16, in_mc, out_mc)
    if error is None:
        _ag_case_simple(*args)
    else:
        _ag_case_expect_fatal(expect_error, error, _ag_case_simple, *args)


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
        ([1, 1, 2048, 4096], ttnn.bfloat16),  # 2048 B page, 16.8 MB
        ([1, 1, 2048, 2048], ttnn.float32),  # 4096 B page, 16.8 MB
    ],
    ids=["p1088", "p2048", "p4096"],
)
@pytest.mark.parametrize("use_blaze", [True, False], ids=["blaze", "minimal"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200000,
                "fabric_router_config": create_fabric_router_config(15232),
            },
            ttnn.Topology.Ring,
        ),
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 200000,
                "fabric_router_config": create_fabric_router_config(15232),
            },
            ttnn.Topology.Linear,
        ),
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
