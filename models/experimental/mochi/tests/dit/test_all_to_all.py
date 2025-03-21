import pytest
import ttnn
import os
import torch
from loguru import logger
from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_all_to_all_pre_attention(mesh_device):
    shape = (1, 1, 44544, 3072 * 3)
    x = torch.randn(shape)
    x_tt = ttnn.from_torch(
        x,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )

    def run_op(x_tt):
        for _ in range(10):
            x_tt_all_to_all = ttnn.all_gather(x_tt, dim=2)
            # ttnn.deallocate(x_tt_all_to_all)

    print("compiling op")
    run_op(x_tt)


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_all_to_all_post_attention(mesh_device):
    shape = (1, 1, 44544, 3072)
    x = torch.randn(shape)
    x_tt = ttnn.from_torch(
        x,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )

    def run_op(x_tt):
        for _ in range(10):
            x_tt_all_to_all = ttnn.all_gather(x_tt, dim=3)
            # ttnn.deallocate(x_tt_all_to_all)

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
    rand_tensor=True,
    mem_config=None,
    create_persistent_fabric=True,
    teardown_persistent_fabric=True,
    wrap_fabric_around_mesh=False,
):
    enable_persistent_fabric = True
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
    if create_persistent_fabric:
        mesh_sub_device_manager_id = create_and_load_sub_device_manager_with_fabric_interface(
            mesh_device,
            [worker_sub_device],
            0,
            0,
            enable_persistent_fabric,
            wrap_fabric_around_mesh=wrap_fabric_around_mesh,
            topology=all_gather_topology,
        )
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphore_with_same_address(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    logger.info(f"Output shape: {output_shape}")
    logger.info(f"dim: {dim}")

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

    for i in range(num_iters):
        tt_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor_mesh_list[i],
            dim,
            multi_device_global_semaphore=ccl_semaphore_handles[i],
            num_links=num_links,
            memory_config=output_mem_config,
            topology=all_gather_topology,
            subdevice_id=worker_sub_device_id,
            enable_persistent_fabric_mode=enable_persistent_fabric,
        )
        tt_out_tensor_list.append(tt_out_tensor)

    logger.info(f"Waiting for op")
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
    logger.info(f"Done op")

    passed = True
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

    for i in range(num_devices):
        assert (
            mesh_device.get_devices()[i].num_program_cache_entries() == 1
            or mesh_device.get_devices()[i].num_program_cache_entries() == num_iters
        ), f"Device {i} has {mesh_device.get_devices()[i].num_program_cache_entries()} program cache entries"

    if enable_persistent_fabric and teardown_persistent_fabric:
        mesh_device.reset_sub_device_stall_group()
        teardown_fabric_interface(
            mesh_device, wrap_fabric_around_mesh=wrap_fabric_around_mesh, topology=all_gather_topology
        )

    # if not passed:
    #     assert eq, f"{i} FAILED: {output}"


@pytest.mark.parametrize(
    "num_devices, num_links, output_shape, dim, layout",
    [
        (8, 1, [1, 1, 3072, 3072], 3, ttnn.TILE_LAYOUT),  # QKV weight
        # (8, 1, [1, 1, 3072, 9216], 3, ttnn.TILE_LAYOUT), # QKV weight
        # (8, 1, [1, 1, 44544, 9216], 2, ttnn.TILE_LAYOUT), # Pre-attn all-to-all
        # (8, 1, [1, 1, 44544, 3072], 3, ttnn.TILE_LAYOUT), # Post-attn all-to-all
    ],
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
@pytest.mark.parametrize("num_iters", [10])
@pytest.mark.parametrize("enable_async", [True])
def test_all_gather_minimal(
    t3k_mesh_device,
    # pcie_mesh_device,
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
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
        mem_config=mem_config,
        wrap_fabric_around_mesh=True,
    )
