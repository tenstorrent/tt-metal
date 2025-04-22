import pytest
import ttnn
import torch
from loguru import logger
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_global_semaphore_with_same_address,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc


def test_allgather_pre_qkv(t3k_mesh_device, use_program_cache):
    t3k_mesh_device.enable_async(True)
    shape = (1, 1, 44544, 3072)
    x = torch.randn(shape)
    x_tt = ttnn.from_torch(
        x,
        device=t3k_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(t3k_mesh_device, dim=2),
    )

    def run_op(x_tt):
        for _ in range(2):
            x_tt_all_to_all = ttnn.all_gather(x_tt, dim=2)

    print("compiling op")
    run_op(x_tt)


def test_allgather_pre_attention(t3k_mesh_device, use_program_cache):
    t3k_mesh_device.enable_async(True)
    shape = (1, 1, 44544, 3072 * 3)
    x = torch.randn(shape)
    x_tt = ttnn.from_torch(
        x,
        device=t3k_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(t3k_mesh_device, dim=2),
    )

    def run_op(x_tt):
        for _ in range(2):
            x_tt_all_to_all = ttnn.all_gather(x_tt, dim=2)

    print("compiling op")
    run_op(x_tt)


def test_allgather_post_attention(t3k_mesh_device, use_program_cache):
    t3k_mesh_device.enable_async(True)
    shape = (1, 1, 44544, 3072)
    x = torch.randn(shape)
    x_tt = ttnn.from_torch(
        x,
        device=t3k_mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(t3k_mesh_device, dim=3),
    )

    def run_op(x_tt):
        for _ in range(2):
            x_tt_all_to_all = ttnn.all_gather(x_tt, dim=3)

    print("compiling op")
    run_op(x_tt)


def run_all_gather_impl(
    mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    use_program_cache,
    function_level_defaults,
    all_gather_topology,
    num_iters=1,
    enable_async=False,
    trace_mode=False,
    rand_tensor=True,
    mem_config=None,
    input_shard_shape=None,
    input_shard_grid=None,
    output_shard_shape=None,
    output_shard_grid=None,
    tensor_mem_layout=None,
    use_cluster_axis_api=False,
    cluster_axis=None,
):
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")
    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)

    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")

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
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)
    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    logger.info(f"Output shape: {output_shape}")
    logger.info(f"dim: {dim}")
    logger.info(f"input_shard_shape: {input_shard_shape}")
    logger.info(f"input_shard_grid: {input_shard_grid}")

    ### For sharded all gather only
    if bool(input_shard_shape) != bool(input_shard_grid) and bool(tensor_mem_layout) != bool(input_shard_grid):
        pytest.fail(
            "Both input_shard_shape, shard_grid, and tensor_mem_layout must be provided together or all must be None"
        )
    if input_shard_shape and input_shard_grid:
        input_shard_spec = ttnn.ShardSpec(
            input_shard_grid,
            input_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_mem_config = ttnn.MemoryConfig(
            tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec
        )
        if output_shard_shape is None:
            assert (
                output_shard_grid is None
            ), "output_shard_grid must not be provided if output_shard_shape is not provided"
            output_shard_shape = list(input_shard_shape)
            if dim == len(output_shape) - 1:
                output_shard_shape[1] *= num_devices
            else:
                output_shard_shape[0] *= num_devices
            output_shard_spec = ttnn.ShardSpec(
                input_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
        else:
            assert output_shard_grid is not None, "output_shard_grid must be provided if output_shard_shape is provided"
            output_shard_spec = ttnn.ShardSpec(
                output_shard_grid,
                output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            output_mem_config = ttnn.MemoryConfig(
                tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
            )
    else:
        assert mem_config is not None
        input_mem_config = mem_config
        output_mem_config = mem_config
    ###

    input_tensor_mesh_list = []
    output_tensor_goldens_list = []

    for i in range(num_iters):
        if rand_tensor:
            output_tensor = torch.rand(output_shape).bfloat16()
        else:
            output_tensor = torch.zeros(output_shape)
            tile_id = 1
            for w in range(output_shape[0]):
                for z in range(output_shape[1]):
                    for y in range(0, output_shape[2], 32):
                        for x in range(0, output_shape[3], 32):
                            output_tensor[w, z, y : y + 32, x : x + 32] = tile_id
                            tile_id += 1

        output_tensor_goldens_list.append(output_tensor)
        input_tensors = torch.chunk(output_tensor, num_devices, dim)
        tt_input_tensors = []
        for i, t in enumerate(input_tensors):
            tt_input_tensors.append(
                ttnn.Tensor(t, input_dtype).to(layout).to(mesh_device.get_devices()[i], input_mem_config)
            )
            logger.info(f"using device {mesh_device.get_devices()[i].id()}")

        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

        input_tensor_mesh_list.append(input_tensor_mesh)

    tt_out_tensor_list = []
    if trace_mode:
        pass
        # tt_out_tensor = run_with_trace(
        #     mesh_device,
        #     all_gather_topology,
        #     input_tensor_mesh_list[0],
        #     dim,
        #     num_links,
        #     output_mem_config,
        #     multi_device_global_semaphore=ccl_semaphore_handles[0],
        #     num_iter=num_iters,
        #     subdevice_id=worker_sub_device_id,
        # )
        # tt_out_tensor_list.append(tt_out_tensor)
    else:
        for i in range(num_iters):
            if use_cluster_axis_api:
                tt_out_tensor = ttnn.experimental.all_gather_async(
                    input_tensor_mesh_list[i],
                    dim,
                    cluster_axis=cluster_axis,
                    mesh_device=mesh_device,
                    memory_config=output_mem_config,
                    topology=all_gather_topology,
                    multi_device_global_semaphore=ccl_semaphore_handles[i],
                    subdevice_id=worker_sub_device_id,
                    num_preferred_links=num_links,
                )

            else:
                tt_out_tensor = ttnn.experimental.all_gather_async(
                    input_tensor_mesh_list[i],
                    dim,
                    multi_device_global_semaphore=ccl_semaphore_handles[i],
                    num_links=num_links,
                    memory_config=output_mem_config,
                    topology=all_gather_topology,
                    subdevice_id=worker_sub_device_id,
                )
            tt_out_tensor_list.append(tt_out_tensor)

        logger.info(f"Waiting for op")
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done op")

    # passed = True
    # for tensor_index in range(len(tt_out_tensor_list)):
    #     tt_out_tensor = tt_out_tensor_list[tensor_index]
    #     output_tensor = output_tensor_goldens_list[tensor_index]
    #     for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
    #         tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    #         logger.info(f"Checking for device {t.device().id()}")

    #         if input_dtype == ttnn.bfloat16:
    #             eq, output = comp_equal(tt_output_tensor, output_tensor)
    #         else:
    #             eq, output = comp_pcc(tt_output_tensor, output_tensor)
    #         if not eq:
    #             logger.error(f"output mismatch for tensor {i}")
    #             passed = False

    # for i in range(num_devices):
    #     assert (
    #         mesh_device.get_devices()[i].num_program_cache_entries() == 1
    #         or mesh_device.get_devices()[i].num_program_cache_entries() == num_iters
    #     ), f"Device {i} has {mesh_device.get_devices()[i].num_program_cache_entries()} program cache entries"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    # if not passed:
    #     assert eq, f"{i} FAILED: {output}"


@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, dim, layout",
    [
        # (8, 1, [1, 1, 3072, 3072], 3, ttnn.TILE_LAYOUT),  # QKV weight
        # (8, 1, [1, 1, 3072, 9216], 3, ttnn.TILE_LAYOUT), # QKV weight
        (8, 1, [1, 1, 44544, 3072], 2, ttnn.TILE_LAYOUT),  # Pre-qkv all-to-all
        (8, 1, [1, 1, 44544, 3072 * 3], 2, ttnn.TILE_LAYOUT),  # Pre-attn all-to-all
        (8, 1, [1, 1, 44544, 3072], 3, ttnn.TILE_LAYOUT),  # Post-attn all-to-all
    ],
    ids=["pre-qkv", "pre-attn", "post-attn"],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("enable_async", [True])
def test_all_gather_minimal(
    t3k_mesh_device,
    num_devices,
    output_shape,
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        output_shape,
        dim,
        num_links,
        input_dtype,
        layout,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        num_iters=num_iters,
        enable_async=enable_async,
        rand_tensor=True,
        mem_config=mem_config,
    )


def run_with_trace(
    mesh_device,
    all_gather_topology,
    input_tensor_mesh,
    in_dim,
    out_dim,
    num_links,
    output_mem_config,
    multi_device_global_semaphore,
    num_iter=20,
    subdevice_id=None,
):
    # Compile Run
    logger.info("Compiling model")
    tt_out_tensor = ttnn.experimental.all_to_all_async(
        input_tensor_mesh,
        in_dim,
        out_dim,
        multi_device_global_semaphore=multi_device_global_semaphore,
        num_links=num_links,
        memory_config=output_mem_config,
        topology=all_gather_topology,
        subdevice_id=subdevice_id,
    )
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    output_tensors = []
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        tt_out_tensor = ttnn.experimental.all_to_all_async(
            input_tensor_mesh,
            in_dim,
            out_dim,
            multi_device_global_semaphore=multi_device_global_semaphore,
            num_links=num_links,
            memory_config=output_mem_config,
            topology=all_gather_topology,
            subdevice_id=subdevice_id,
        )
        output_tensors.append(tt_out_tensor)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Run the op
    logger.info("Starting Trace perf test...")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)

    return output_tensors


def run_all_to_all_impl(
    mesh_device,
    num_devices,
    logical_shape,
    in_dim,
    out_dim,
    num_links,
    input_dtype,
    layout,
    use_program_cache,
    function_level_defaults,
    all_gather_topology,
    num_iters=1,
    enable_async=False,
    trace_mode=False,
    mem_config=None,
    do_check=True,
):
    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")
    # Use Async mode based on test input config
    mesh_device.enable_async(enable_async)

    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")

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
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)
    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    logger.info(f"Logical shape: {logical_shape}")
    logger.info(f"in_dim: {in_dim}")
    logger.info(f"out_dim: {out_dim}")

    input_mem_config = mem_config
    output_mem_config = mem_config
    ###

    input_tensor_mesh_list = []
    output_tensor_goldens_list = []

    for i in range(num_iters):
        output_tensor = torch.rand(logical_shape).bfloat16()
        # shard_shape = list(logical_shape)
        # shard_shape[in_dim] = shard_shape[in_dim] // num_devices
        # tts = []
        # for i in range(num_devices):
        #     tts.append(torch.full(shard_shape, i))
        # output_tensor = torch.cat(tts, dim=in_dim)

        output_tensor_goldens_list.append(torch.chunk(output_tensor, num_devices, out_dim))
        input_tensors = torch.chunk(output_tensor, num_devices, in_dim)
        tt_input_tensors = []
        for i, t in enumerate(input_tensors):
            tt_input_tensors.append(
                ttnn.Tensor(t, input_dtype).to(layout).to(mesh_device.get_devices()[i], input_mem_config)
            )
            logger.info(f"using device {mesh_device.get_devices()[i].id()}")

        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

        input_tensor_mesh_list.append(input_tensor_mesh)

    tt_out_tensor_list = []
    if trace_mode:
        tt_out_tensor_list = run_with_trace(
            mesh_device,
            all_gather_topology,
            input_tensor_mesh_list[0],
            in_dim,
            out_dim,
            num_links,
            output_mem_config,
            multi_device_global_semaphore=ccl_semaphore_handles[0],
            num_iter=num_iters,
            subdevice_id=worker_sub_device_id,
        )
    else:
        for i in range(num_iters):
            tt_out_tensor = ttnn.experimental.all_to_all_async(
                input_tensor_mesh_list[i],
                in_dim,
                out_dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                memory_config=output_mem_config,
                topology=all_gather_topology,
                subdevice_id=worker_sub_device_id,
            )
            tt_out_tensor_list.append(tt_out_tensor)

        logger.info(f"Waiting for op")
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done op")

    passed = True
    if do_check:
        for tensor_index in range(len(tt_out_tensor_list)):
            tt_out_tensor = tt_out_tensor_list[tensor_index]
            output_tensors = output_tensor_goldens_list[tensor_index]
            for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
                tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
                output_tensor = output_tensors[i]
                logger.info(f"Checking for device {t.device().id()}")
                if input_dtype == ttnn.bfloat16:
                    eq, output = comp_equal(tt_output_tensor, output_tensor)
                else:
                    eq, output = comp_pcc(tt_output_tensor, output_tensor)
                if not eq:
                    logger.error(f"output mismatch for tensor {i}")
                    passed = False

    for i in range(num_devices):
        assert (
            mesh_device.get_devices()[i].num_program_cache_entries() == 1
            or mesh_device.get_devices()[i].num_program_cache_entries() == num_iters
        ), f"Device {i} has {mesh_device.get_devices()[i].num_program_cache_entries()} program cache entries"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    if do_check and not passed:
        assert eq, f"{i} FAILED: {output}"


@pytest.mark.parametrize(
    "num_devices, num_links, logical_shape, in_dim, out_dim, layout",
    [
        (8, 1, [1, 1, 44544, 3072 * 3], 2, 3, ttnn.TILE_LAYOUT),  # Pre-attn all-to-all
        (8, 1, [1, 1, 44544, 3072], 3, 2, ttnn.TILE_LAYOUT),  # Post-attn all-to-all
    ],
    ids=["pre-attn", "post-attn"],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "mem_config",
    [
        ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
    ],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("num_iters, do_check", [(2, True), (6, False)], ids=["check", "perf"])
@pytest.mark.parametrize("enable_async", [True])
def test_all_to_all(
    t3k_mesh_device,
    num_devices,
    logical_shape,
    in_dim,
    out_dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    num_iters,
    use_program_cache,
    function_level_defaults,
    enable_async,
    do_check,
):
    run_all_to_all_impl(
        t3k_mesh_device,
        num_devices,
        logical_shape,
        in_dim,
        out_dim,
        num_links,
        input_dtype,
        layout,
        use_program_cache,
        function_level_defaults,
        all_gather_topology=ttnn.Topology.Ring,
        num_iters=num_iters,
        enable_async=enable_async,
        mem_config=mem_config,
        do_check=do_check,
        trace_mode=False,
    )
