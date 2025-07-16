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
    matmul_output_dim,
    matmul_weights_dtype,
    max_in0_block_w,
    use_bias,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    all_gather_topology,
    use_non_fused,
    use_legacy_allgather,
    mem_config_weights=None,
    num_iters=1,
    enable_trace=True,
):
    torch.manual_seed(0)

    tile = (32, 32)

    # Set the default config
    if mem_config_weights is None:
        mem_config_weights = mem_config_ag

    # Skip unsupported cases
    (is_known_failure, message) = is_unsupported_case(
        ag_output_shape, dim, mem_config_ag, num_devices, num_links, ag_input_dtype, layout, tile
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

    if not use_legacy_allgather:
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

        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)

        # create global semaphore handles
        ccl_semaphore_handles = [
            create_global_semaphores(mesh_device, num_devices, ccl_sub_device_crs, 0) for _ in range(num_iters)
        ]

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
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

    ##### Matmul weight setup #####
    if use_bias:
        weights_tensor = torch.randn([hidden_dim, matmul_output_dim * num_devices]).bfloat16()
        weights_tensor_padded = weights_tensor.unsqueeze(0).unsqueeze(0)
    else:
        weights_tensor = torch.randn([1, 1, hidden_dim, matmul_output_dim * num_devices]).bfloat16()
        weights_tensor_padded = weights_tensor
    weight_tt = ttnn.from_torch(
        weights_tensor_padded,
        dtype=matmul_weights_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=mem_config_weights,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=dim),
    )

    if use_bias:
        bias_tensor = torch.randn([1, matmul_output_dim * num_devices]).float()
        bias_tensor_padded = bias_tensor.unsqueeze(0).unsqueeze(0)
        bias_tt = ttnn.from_torch(
            bias_tensor_padded,
            dtype=matmul_weights_dtype,
            layout=layout,
            device=mesh_device,
            memory_config=mem_config_weights,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=dim),
            tile=ttnn.Tile(tile),
        )
    else:
        bias_tt = None

    ##### Configs for ttnn.matmul #####
    core_grid = (8, 6)
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=min(max_in0_block_w, hidden_dim // 32 // core_grid[0]),  # how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max(1, math.ceil(ag_output_shape[2] / 32 / core_grid[1])),  # M / TILE_HEIGHT / Grid_Size
        per_core_N=max(1, math.ceil(matmul_output_dim / 32 / core_grid[0])),  # N / TILE_WIDTH / Grid_Size
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

    ##### Perform torch ops #####
    torch_matmul_output_list = []
    for i in range(num_iters):
        if use_bias:
            matmul_output = torch.nn.functional.linear(
                ag_output_tensor_goldens_list[i], weights_tensor.T.contiguous(), bias_tensor
            )
        else:
            matmul_output = torch.matmul(ag_output_tensor_goldens_list[i], weights_tensor)
        torch_matmul_output_list.append(matmul_output)

    ##### Perform the TT ops #####
    tt_matmul_out_tensor_list = []
    tt_all_gather_out_tensor_list = []

    def run_op(i):
        if use_non_fused:
            if use_legacy_allgather:
                tt_all_gather_out_tensor = ttnn.all_gather(
                    input_tensor_mesh_list[i],
                    dim,
                    num_links=num_links,
                    memory_config=mem_config_ag,
                )
            else:
                tt_all_gather_out_tensor = ttnn.experimental.all_gather_async(
                    input_tensor_mesh_list[i],
                    persistent_output_buffer=persistent_output_buffers[i],
                    dim=dim,
                    multi_device_global_semaphore=ccl_semaphore_handles[i],
                    num_links=num_links,
                    memory_config=mem_config_ag,
                    topology=all_gather_topology,
                    subdevice_id=worker_sub_device_id,
                )

            tt_matmul_out_tensor = ttnn.linear(
                tt_all_gather_out_tensor,
                weight_tt,
                bias=bias_tt,
                memory_config=mem_config_mm,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
        else:
            if use_legacy_allgather:
                tt_all_gather_out_tensor, tt_matmul_out_tensor, _ = ttnn.experimental.all_gather_matmul(
                    input_tensor_mesh_list[i],
                    weight_tt,
                    dim,
                    (0, 6),
                    bias=bias_tt,
                    num_links=num_links,
                    memory_config_ag=mem_config_ag,
                    memory_config_mm=mem_config_mm,
                    transpose_a=False,
                    transpose_b=False,
                    program_config=program_config,
                    compute_kernel_config=compute_kernel_config,
                )
            else:
                tt_all_gather_out_tensor, tt_matmul_out_tensor = ttnn.experimental.all_gather_matmul_async(
                    input_tensor_mesh_list[i],
                    weight_tt,
                    persistent_output_buffer=persistent_output_buffers[i],
                    dim=dim,
                    multi_device_global_semaphore=ccl_semaphore_handles[i],
                    all_gather_core_grid_offset=(0, 6),
                    bias=bias_tt,
                    num_links=num_links,
                    memory_config_ag=mem_config_ag,
                    topology=all_gather_topology,
                    subdevice_id=worker_sub_device_id,
                    memory_config_mm=mem_config_mm,
                    program_config=program_config,
                    compute_kernel_config=compute_kernel_config,
                )

        return tt_all_gather_out_tensor, tt_matmul_out_tensor

    if enable_trace:
        # Compile the op
        tt_all_gather_out_tensor, tt_matmul_out_tensor = run_op(0)
        if not use_legacy_allgather:
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_all_gather_out_tensor, tt_matmul_out_tensor = run_op(0)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        if not use_legacy_allgather:
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done capturing trace")

        # Execute trace
        for i in range(num_iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            if not use_legacy_allgather:
                ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)
            tt_matmul_out_tensor_list.append(tt_matmul_out_tensor)
        logger.info(f"Done executing trace")
    else:
        for i in range(num_iters):
            tt_all_gather_out_tensor, tt_matmul_out_tensor = run_op(i)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)
            tt_matmul_out_tensor_list.append(tt_matmul_out_tensor)

            if not use_legacy_allgather:
                logger.info(f"Waiting for op")
                ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
                logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_mm_out_tensor = tt_matmul_out_tensor_list[i]
        torch_mm_out_tensor = torch_matmul_output_list[i if not enable_trace else 0]

        tt_mm_out = ttnn.from_device(tt_mm_out_tensor)
        tt_mm_out = ttnn.to_torch(tt_mm_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
        eq, output = comp_pcc(tt_mm_out, torch_mm_out_tensor)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED mm: {output}"

        tt_ag_out_tensor = tt_all_gather_out_tensor_list[i]
        torch_ag_out_tensor = ag_output_tensor_goldens_list[i if not enable_trace else 0]

        tt_ag_out = ttnn.from_device(tt_ag_out_tensor)
        tt_ag_out = ttnn.to_torch(tt_ag_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
        tt_ag_out = tt_ag_out[:, :, :, 0 : torch_ag_out_tensor.shape[3]]
        eq, output = comp_pcc(tt_ag_out, torch_ag_out_tensor, 1)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED ag: {output}"

    if not use_legacy_allgather:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize(
    "num_devices, num_links, ag_output_shape, dim, layout, matmul_output_dim, max_in0_block_w, matmul_weights_dtype, ag_input_dtype, use_bias",
    [
        (4, 1, [1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT, 960, 2, ttnn.bfloat16, ttnn.bfloat16, True),
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag, mem_config_mm",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [
        # (True, 10),
        (False, 50),
    ],
    # ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "use_non_fused",
    [
        True,
        # False,
    ],
    # ids=["separate", "fused"],
)
@pytest.mark.parametrize(
    "device_params, use_legacy_allgather, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, False, ttnn.Topology.Ring),
        # ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, False, ttnn.Topology.Linear),
        # (
        #     {"trace_region_size": 90112},
        #     True,
        #     ttnn.Topology.Ring,
        # ),
    ],
    indirect=["device_params"],
    # ids=["fabric_ring", "fabric_linear", "legacy_ring"],
)
def test_all_gather_matmul_async(
    mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    matmul_output_dim,
    matmul_weights_dtype,
    max_in0_block_w,
    use_bias,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    enable_trace,
    use_non_fused,
    use_legacy_allgather,
    all_gather_topology,
    num_iters,
):
    if use_non_fused == False and all_gather_topology == ttnn.Topology.Linear:
        pytest.skip("linear is not supported when using fused for all-gather")

    run_all_gather_impl(
        mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        matmul_output_dim,
        matmul_weights_dtype,
        max_in0_block_w,
        use_bias,
        mem_config_input,
        mem_config_ag,
        mem_config_mm,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        use_non_fused=use_non_fused,
        use_legacy_allgather=use_legacy_allgather,
        num_iters=num_iters,
    )
