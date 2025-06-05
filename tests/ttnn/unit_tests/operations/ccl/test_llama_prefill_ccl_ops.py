# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch
import pytest
from loguru import logger
from tests.ttnn.unit_tests.operations.ccl.test_new_reduce_scatter import run_reduce_scatter_impl
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def padded_shape(output_shape, tile):
    output_tiles_shape = (math.ceil(output_shape[2] / tile[0]), math.ceil(output_shape[3] / tile[1]))
    output_tile_num = output_tiles_shape[0] * output_tiles_shape[1]
    padded_output_tile_num = math.ceil(output_tile_num / 48) * 48

    padded_shape = [
        output_shape[0],
        output_shape[1],
        output_shape[2],
        math.ceil(padded_output_tile_num / output_tiles_shape[0]) * tile[1],
    ]
    return padded_shape


def run_all_gather_impl(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    all_gather_topology,
    use_program_cache,
    num_iters=1,
    enable_trace=True,
):
    torch.manual_seed(0)

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    tile = (32, 32)

    # Skip unsupported cases
    (is_known_failure, message) = is_unsupported_case(
        ag_output_shape, dim, mem_config, num_devices, num_links, ag_input_dtype, layout, tile
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
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

    sub_device_manager = t3k_mesh_device.create_sub_device_manager([worker_sub_device], 0)
    t3k_mesh_device.load_sub_device_manager(sub_device_manager)
    t3k_mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphores(t3k_mesh_device, num_devices, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    persistent_intermediate_buffers = [
        ttnn.from_torch(
            torch.zeros(padded_shape(ag_output_shape, tile)),
            device=t3k_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ag_input_dtype,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
        )
        for _ in range(num_iters)
    ]
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(ag_output_shape),
            device=t3k_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ag_input_dtype,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
        )
        for _ in range(num_iters)
    ]

    logger.info("Done creating persistent buffers")

    ##### All gather input setup #####
    input_tensor_mesh_list = []
    ag_output_tensor_goldens_list = []

    for i in range(num_iters):
        ag_output_tensor = torch.randn(ag_output_shape).bfloat16()
        ag_output_tensor_goldens_list.append(ag_output_tensor)
        input_tensors = torch.chunk(ag_output_tensor, num_devices, dim)
        tt_input_tensors = []
        for i, t in enumerate(input_tensors):
            tt_input_tensors.append(ttnn.Tensor(t, ag_input_dtype).to(layout))
        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors).to(t3k_mesh_device, mem_config)

        input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform the TT ops #####
    tt_all_gather_out_tensor_list = []

    def run_op(i):
        tt_all_gather_out_tensor = ttnn.experimental.all_gather_async(
            input_tensor_mesh_list[i],
            persistent_intermediate_buffer=persistent_intermediate_buffers[i],
            persistent_output_buffer=persistent_output_buffers[i],
            dim=dim,
            multi_device_global_semaphore=ccl_semaphore_handles[i],
            num_links=num_links,
            memory_config=mem_config,
            topology=all_gather_topology,
            subdevice_id=worker_sub_device_id,
        )

        return tt_all_gather_out_tensor

    if enable_trace:
        assert num_iters == 1, "When running in trace, use num_iters = 1"

        # Compile the op
        tt_all_gather_out_tensor = run_op(0)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        tt_all_gather_out_tensor = run_op(0)
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")

        # Synchronize the devices
        ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)

        tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)
    else:
        for i in range(num_iters):
            tt_all_gather_out_tensor = run_op(i)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_ag_out_tensor = tt_all_gather_out_tensor_list[i]
        torch_ag_out_tensor = ag_output_tensor_goldens_list[i]

        tt_ag_out = ttnn.from_device(tt_ag_out_tensor)
        tt_ag_out = ttnn.to_torch(tt_ag_out, mesh_composer=ttnn.ConcatMeshToTensor(t3k_mesh_device, dim=3))[
            :, :, :, 0 : torch_ag_out_tensor.shape[3]
        ]
        eq, output = comp_pcc(tt_ag_out, torch_ag_out_tensor)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED ag: {output}"

    t3k_mesh_device.reset_sub_device_stall_group()
    t3k_mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize(
    "num_devices, num_links, ag_output_shape, dim, layout, ag_input_dtype",
    [
        # 4k shapes
        (8, 1, [1, 1, 4096, 320 * 8], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [1, 1, 4096, 256 * 8], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [1, 1, 4096, 32 * 8], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 1, [1, 1, 4096, 896 * 8], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        # 8k shapes
        (8, 1, [1, 1, 8192, 320 * 8], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [1, 1, 8192, 256 * 8], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [1, 1, 8192, 32 * 8], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        (8, 1, [1, 1, 8192, 896 * 8], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 1),
        (False, 10),
    ],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
)
def test_all_gather_async(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    enable_trace,
    num_iters,
    use_program_cache,
    all_gather_topology,
):
    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        use_program_cache=use_program_cache,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )


@pytest.mark.parametrize(
    "num_devices, num_links, rs_input_shape, dim, layout, rs_input_dtype",
    [
        # 4k shapes
        (8, 1, [1, 1, 4096, 1280], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [1, 1, 4096, 2048], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [1, 1, 4096, 3584], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        # 8k shapes
        (8, 1, [1, 1, 8192, 1280], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [1, 1, 8192, 2048], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [1, 1, 8192, 3584], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
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
    "enable_trace, num_iters",
    [
        (True, 1),
        (False, 10),
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
)
def test_reduce_scatter_async(
    t3k_mesh_device,
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
    use_program_cache,
    rs_topology,
):
    run_reduce_scatter_impl(
        t3k_mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        use_program_cache=use_program_cache,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )
