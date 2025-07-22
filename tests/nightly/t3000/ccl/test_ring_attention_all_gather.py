# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc


def create_global_semaphores(mesh_device, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def create_persistent_buffers(ag_output_shape, num_inputs, mesh_device, ag_input_dtype, mem_config_ag, rp_axis):
    dims = [None, None]
    up_axis = 1 - rp_axis
    dims[up_axis] = 1
    persistent_buffers = [
        ttnn.from_torch(
            torch.zeros(ag_output_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ag_input_dtype,
            memory_config=mem_config_ag,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )
        for _ in range(num_inputs)
    ]

    return persistent_buffers


def create_ring_attention_submesh(mesh_device, rp_axis, rp_factor, up_factor):
    submesh_shape = [0, 0]
    submesh_shape[rp_axis] = rp_factor
    submesh_shape[1 - rp_axis] = up_factor
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape[0], submesh_shape[1]))
    return submesh_device


def run_ring_attention_all_gather_impl(
    mesh_device,
    ag_output_shape,
    ag_num_inputs,
    rp_axis,
    rp_factor,
    up_factor,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    all_gather_topology,
    num_iters=1,
    enable_trace=True,
):
    torch.manual_seed(0)

    sequence_index = 2
    head_index = 1

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

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
    ccl_semaphore_handles = [create_global_semaphores(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    persistent_output_buffers = [
        create_persistent_buffers(
            ag_output_shape,
            ag_num_inputs,
            mesh_device,
            ag_input_dtype,
            mem_config_ag,
            rp_axis,
        )
        for _ in range(num_iters)
    ]
    logger.info("Done creating persistent buffers")

    ##### All gather input setup #####
    logger.info(f"All gather output shape: {ag_output_shape}")

    ag_input_tensor_mesh_list = []
    ag_output_tensor_list = []
    _, _, _, hidden_dim = ag_output_shape

    input_dims = [None, None]
    input_dims[rp_axis] = 2 if rp_factor > 1 else None
    input_dims[1 - rp_axis] = 1 if up_factor > 1 else None
    for i in range(num_iters):
        ag_output_tensor_list_per_iteration = []
        ag_input_tensor_mesh_list_per_iteration = []
        for j in range(ag_num_inputs):
            ag_output_tensor = torch.rand(ag_output_shape).bfloat16()
            ag_output_tensor_list_per_iteration.append(ag_output_tensor)
            input_tensor_mesh = ttnn.from_torch(
                ag_output_tensor,
                device=mesh_device,
                layout=layout,
                dtype=ag_input_dtype,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_dims),
                memory_config=mem_config_input,
            )
            ag_input_tensor_mesh_list_per_iteration.append(input_tensor_mesh)
        ag_output_tensor_list.append(ag_output_tensor_list_per_iteration)
        ag_input_tensor_mesh_list.append(ag_input_tensor_mesh_list_per_iteration)

    ##### Perform the TT ops #####
    tt_all_gather_out_tensor_list = []

    def run_op(i):
        tt_all_gather_out_tensors = ttnn.experimental.ring_attention_all_gather_async(
            ag_input_tensor_mesh_list[i],
            persistent_output_buffer=persistent_output_buffers[i],
            dim=sequence_index,
            multi_device_global_semaphore=ccl_semaphore_handles[i],
            cluster_axis=rp_axis,
            mesh_device=mesh_device,
            num_links=num_links,
            memory_config=mem_config_ag,
            topology=all_gather_topology,
            subdevice_id=worker_sub_device_id,
        )

        return tt_all_gather_out_tensors

    if enable_trace:
        # Compile the op
        tt_all_gather_out_tensors = run_op(0)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_all_gather_out_tensors = run_op(0)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done capturing trace")

        # Execute trace
        for i in range(num_iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensors)
        logger.info(f"Done executing trace")
    else:
        for i in range(num_iters):
            tt_all_gather_out_tensors = run_op(i)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensors)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    output_dims = [None, None]
    output_dims[rp_axis] = sequence_index
    output_dims[1 - rp_axis] = head_index
    for i in range(num_iters):
        tt_ag_out_tensors = tt_all_gather_out_tensor_list[i]
        torch_ag_out_tensors = ag_output_tensor_list[i if not enable_trace else 0]

        for j in range(ag_num_inputs):
            tt_ag_out = ttnn.to_torch(
                tt_ag_out_tensors[j],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device,
                    mesh_shape=tuple(mesh_device.shape),
                    dims=output_dims,
                ),
            )
            tt_ag_out = torch.narrow(tt_ag_out, sequence_index, 0, torch_ag_out_tensors[j].shape[sequence_index])
            if ag_input_dtype == ttnn.bfloat16:
                eq, output = comp_equal(tt_ag_out, torch_ag_out_tensors[j])
            else:
                eq, output = comp_pcc(tt_ag_out, torch_ag_out_tensors[j])
            logger.info(f"{output}, iteration {i}, tensor {j}")
            assert eq, f"{i}{j} FAILED ag: {output}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("layout, ag_input_dtype", [(ttnn.TILE_LAYOUT, ttnn.bfloat16)])
@pytest.mark.parametrize(
    "ag_output_shape, ag_num_inputs, rp_axis, rp_factor, up_factor",
    [
        ([1, 1, 4096, 2560], 1, 1, 4, 1),
        ([1, 5, 4096, 64], 1, 1, 4, 1),
        ([1, 5, 4096, 64], 2, 0, 2, 1),
        ([1, 10, 4096, 64], 2, 1, 4, 1),
        ([1, 40, 4096, 64], 2, 1, 4, 2),
    ],
    ids=[
        "shape1_1input_rp4",
        "shape2_1input_rp4",
        "shape2_2input_rp2",
        "shape3_2input_rp4",
        "shape4_2input_rp4_up2",
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
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    ids=["ring ", "line"],
    indirect=["device_params"],
)
def test_ring_attention_all_gather(
    mesh_device,
    ag_output_shape,
    ag_num_inputs,
    rp_axis,
    rp_factor,
    up_factor,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    all_gather_topology,
):
    if all_gather_topology == ttnn.Topology.Ring:
        pytest.skip("Ring topology not supported on T3K - requires 2D torus")

    submesh_device = create_ring_attention_submesh(mesh_device, rp_axis, rp_factor, up_factor)

    run_ring_attention_all_gather_impl(
        submesh_device,
        ag_output_shape,
        ag_num_inputs,
        rp_axis,
        rp_factor,
        up_factor,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("layout, ag_input_dtype", [(ttnn.TILE_LAYOUT, ttnn.bfloat16)])
@pytest.mark.parametrize(
    "ag_output_shape, ag_num_inputs, rp_axis, rp_factor, up_factor",
    [
        ([1, 5, 4096, 64], 2, 1, 4, 1),
        ([1, 5, 4096, 64], 2, 0, 2, 1),
    ],
    ids=[
        "shape2_2input_rp4",
        "shape2_2input_rp2",
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
        (False, 1),
    ],
    ids=["check"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    ids=[
        "line",
    ],
    indirect=["device_params"],
)
def test_ring_attention_all_gather_program_cache(
    mesh_device,
    ag_output_shape,
    ag_num_inputs,
    rp_axis,
    rp_factor,
    up_factor,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    all_gather_topology,
):
    submesh_device = create_ring_attention_submesh(mesh_device, rp_axis, rp_factor, up_factor)

    dummy_tensors = []
    for i in range(3):
        dummy_tensors.append(
            ttnn.from_torch(
                torch.rand(ag_output_shape),
                device=submesh_device,
                layout=layout,
                dtype=ag_input_dtype,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    submesh_device, mesh_shape=tuple(submesh_device.shape), dims=[None, None]
                ),
            )
        )
        run_ring_attention_all_gather_impl(
            submesh_device,
            ag_output_shape,
            ag_num_inputs,
            rp_axis,
            rp_factor,
            up_factor,
            num_links,
            ag_input_dtype,
            layout,
            mem_config_input,
            mem_config_ag,
            all_gather_topology=all_gather_topology,
            enable_trace=enable_trace,
            num_iters=num_iters,
        )
        ttnn.synchronize_device(submesh_device)

    assert submesh_device.num_program_cache_entries() == 1
