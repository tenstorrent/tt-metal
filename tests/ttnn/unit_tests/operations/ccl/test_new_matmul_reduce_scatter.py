# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case

from ttnn import ShardTensorToMesh, ConcatMeshToTensor


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(3)]
    return ccl_semaphore_handles


def run_reduce_scatter_impl(
    t3k_mesh_device,
    num_devices,
    rs_input_shape,
    mm_shard_dim,
    rs_scatter_dim,
    num_links,
    mm_weights_shape,
    rs_input_dtype,
    layout,
    matmul_weights_dtype,
    max_in0_block_w,
    use_bias,
    mem_config_input,
    mem_config_rs,
    mem_config_mm,
    rs_topology,
    use_non_fused,
    use_program_cache,
    mem_config_weights=None,
    num_iters=1,
    enable_trace=True,
):
    torch.manual_seed(0)

    tile = (32, 32)

    # Set the default config
    if mem_config_weights is None:
        mem_config_weights = mem_config_rs

    ##### Fabric setup #####
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
    rs_num_batches = rs_input_shape[0]
    single_batch_input_shape = rs_input_shape[:]
    single_batch_input_shape[2] //= rs_num_batches
    persistent_intermediate_buffers = [
        ttnn.from_torch(
            torch.zeros(single_batch_input_shape),
            device=t3k_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=rs_input_dtype,
            memory_config=mem_config_rs,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
        )
        for _ in range(num_iters)
    ]
    rs_output_shape = rs_input_shape[:]
    rs_output_shape[3] //= num_devices
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(rs_output_shape),
            device=t3k_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=rs_input_dtype,
            memory_config=mem_config_rs,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
        )
        for _ in range(num_iters)
    ]

    logger.info("Done creating persistent buffers")

    ##### Matmul weight setup #####
    weights_tensor = torch.randn(mm_weights_shape).bfloat16()
    weights_tensor_padded = weights_tensor
    weight_tt = ttnn.from_torch(
        weights_tensor_padded,
        dtype=matmul_weights_dtype,
        layout=layout,
        device=t3k_mesh_device,
        memory_config=mem_config_weights,
        mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=mm_shard_dim),
    )

    if use_bias:
        bias_tensor_padded = torch.randn([1, 1, 1, rs_input_shape[3]]).float()
        bias_tensor_scaled = bias_tensor_padded * (1 / 8.0)
        bias_tt = ttnn.from_torch(
            bias_tensor_scaled,
            dtype=matmul_weights_dtype,
            layout=layout,
            device=t3k_mesh_device,
            memory_config=mem_config_weights,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
            tile=ttnn.Tile(tile),
        )
    else:
        bias_tt = None

    ##### Configs for ttnn.matmul #####
    core_grid = (8, 6)
    in0_block_w = min(max_in0_block_w, mm_weights_shape[2] // num_devices // 32 // core_grid[0])
    per_core_M = max(1, math.ceil(rs_input_shape[2] / 32 / core_grid[1]))  # M / TILE_HEIGHT / Grid_Size
    per_core_N = max(1, math.ceil(rs_input_shape[3] / 32 / core_grid[0]))  # N / TILE_WIDTH / Grid_Size
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_block_w=per_core_N // 2,
        transpose_mcast=False,
        fused_activation=None,  # ttnn.UnaryOpType.SILU,
        fuse_batch=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    ##### MM input setup #####
    logger.info(f"Reduce scatter shape: {rs_input_shape}")
    logger.info(f"Reduce scatter dim: {rs_scatter_dim}")

    tt_input_tensor_mesh_list = []
    torch_input_tensor_list = []

    for i in range(num_iters):
        mm_input_shape = [rs_input_shape[0], 1, rs_input_shape[2], mm_weights_shape[2]]
        mm_input_tensor = torch.rand(mm_input_shape).bfloat16()
        input_tensors = torch.chunk(mm_input_tensor, num_devices, 3)
        torch_input_tensor_list.append(input_tensors)
        tt_input_tensors = []
        for j, t in enumerate(input_tensors):
            tt_input_tensors.append(ttnn.Tensor(t, rs_input_dtype).to(layout))
        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors).to(t3k_mesh_device, mem_config_input)

        tt_input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform torch ops #####
    torch_reduce_scatter_output_list = []
    torch_matmul_output_list = []
    for i in range(num_iters):
        matmul_input = torch.cat(torch_input_tensor_list[i], dim=3)
        if use_bias:
            matmul_output = torch.matmul(matmul_input, weights_tensor) + bias_tensor_padded
        else:
            matmul_output = torch.matmul(matmul_input, weights_tensor)
        scatter_output = torch.chunk(matmul_output, num_devices, rs_scatter_dim)
        torch_reduce_scatter_output_list.append(scatter_output)
        torch_matmul_output_list.append(matmul_output)

    ##### Perform the TT ops #####
    tt_reduce_scatter_output_list = []
    tt_matmul_output_list = []

    def run_op(i):
        if use_non_fused:
            tt_matmul_out_tensor = ttnn.linear(
                tt_input_tensor_mesh_list[i],
                weight_tt,
                bias=bias_tt,
                memory_config=mem_config_mm,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
            tt_reduce_scatter_output_tensor = ttnn.experimental.reduce_scatter_minimal_async(
                tt_matmul_out_tensor,
                persistent_intermediate_buffer=persistent_intermediate_buffers[i],
                persistent_output_buffer=persistent_output_buffers[i],
                dim=rs_scatter_dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                memory_config=mem_config_rs,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
            )
        else:
            tt_matmul_out_tensor, tt_reduce_scatter_output_tensor = ttnn.experimental.matmul_reduce_scatter_async(
                tt_input_tensor_mesh_list[i],
                weight_tt,
                persistent_intermediate_buffer=persistent_intermediate_buffers[i],
                persistent_output_buffer=persistent_output_buffers[i],
                dim=rs_scatter_dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                reduce_scatter_core_grid_offset=(0, 6),
                bias=bias_tt,
                num_links=num_links,
                memory_config_rs=mem_config_rs,
                topology=rs_topology,
                subdevice_id=worker_sub_device_id,
                memory_config_mm=mem_config_mm,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )

        return tt_matmul_out_tensor, tt_reduce_scatter_output_tensor

    if enable_trace:
        # Compile the op
        for i in range(num_iters):
            tt_matmul_out_tensor, tt_reduce_scatter_output_tensor = run_op(i)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_matmul_out_tensor, tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_list.append(tt_reduce_scatter_output_tensor)
            tt_matmul_output_list.append(tt_matmul_out_tensor)
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")

        # Synchronize the devices
        ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
    else:
        for i in range(num_iters):
            tt_matmul_out_tensor, tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_list.append(tt_reduce_scatter_output_tensor)
            tt_matmul_output_list.append(tt_matmul_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_mm_out_tensor = tt_matmul_output_list[i]
        torch_mm_out_tensor = torch_matmul_output_list[i]

        tt_mm_out = ttnn.from_device(tt_mm_out_tensor)
        tt_mm_out = ttnn.to_torch(tt_mm_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=3))
        tt_mm_out = torch.sum(torch.stack(torch.chunk(tt_mm_out, num_devices, 3)), dim=0)
        eq, output = comp_pcc(tt_mm_out, torch_mm_out_tensor)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED mm: {output}"
        tt_rs_out_tensor = tt_reduce_scatter_output_list[i]
        torch_rs_out_tensor = torch_reduce_scatter_output_list[i]

        torch_rs_out = torch.cat(torch_rs_out_tensor, 3)

        tt_rs_out = ttnn.from_device(tt_rs_out_tensor)
        tt_rs_out = ttnn.to_torch(tt_rs_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=3))
        eq, output = comp_pcc(tt_rs_out, torch_rs_out)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED ag: {output}"

        # print(f"RS TORCH TENSOR {torch_rs_out}")
        # print(f"RS TT TENSOR {tt_rs_out}")

    t3k_mesh_device.reset_sub_device_stall_group()
    t3k_mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize(
    "num_devices, num_links, mm_weights_shape, rs_input_shape, mm_shard_dim, rs_scatter_dim, layout, max_in0_block_w, matmul_weights_dtype, rs_input_dtype, use_bias",
    [
        (
            8,
            1,
            [1, 1, 10240, 2560],
            [8, 1, 512, 2560],
            2,
            3,
            ttnn.TILE_LAYOUT,
            5,
            ttnn.bfloat16,
            ttnn.bfloat16,
            True,
        ),  # use batching when fused
        (
            8,
            1,
            [1, 1, 10240, 2560],
            [4, 1, 1024, 2560],
            2,
            3,
            ttnn.TILE_LAYOUT,
            5,
            ttnn.bfloat16,
            ttnn.bfloat16,
            True,
        ),  # use batching when fused
        (
            8,
            1,
            [1, 1, 10240, 2560],
            [2, 1, 2048, 2560],
            2,
            3,
            ttnn.TILE_LAYOUT,
            5,
            ttnn.bfloat16,
            ttnn.bfloat16,
            True,
        ),  # use batching when fused
        (
            8,
            1,
            [1, 1, 10240, 2560],
            [1, 1, 4096, 2560],
            2,
            3,
            ttnn.TILE_LAYOUT,
            5,
            ttnn.bfloat16,
            ttnn.bfloat16,
            True,
        ),  # use batching when fused
    ],
    ids=["batch_8", "batch_4", "batch_2", "batch_1"],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_mm, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
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
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 266240}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
)
def test_reduce_scatter_async(
    t3k_mesh_device,
    num_devices,
    num_links,
    mm_weights_shape,
    rs_input_shape,
    mm_shard_dim,
    rs_scatter_dim,
    layout,
    use_bias,
    matmul_weights_dtype,
    max_in0_block_w,
    rs_input_dtype,
    mem_config_mm,
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
        mm_shard_dim,
        rs_scatter_dim,
        num_links,
        mm_weights_shape,
        rs_input_dtype,
        layout,
        matmul_weights_dtype,
        max_in0_block_w,
        use_bias,
        mem_config_input,
        mem_config_rs,
        mem_config_mm,
        use_program_cache=use_program_cache,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        use_non_fused=False,
    )
