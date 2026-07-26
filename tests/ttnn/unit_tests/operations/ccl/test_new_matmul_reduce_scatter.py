# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from models.experimental.kimi_delta_attention.tt.sp_layer import _socket_config
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

from ttnn import ShardTensorToMesh, ConcatMeshToTensor


def create_global_semaphores(mesh_device, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(3)]
    return ccl_semaphore_handles


def run_reduce_scatter_impl(
    mesh_device,
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
    mem_config_weights=None,
    num_iters=1,
    enable_trace=True,
    check_first_output_tile=False,
    input_rank3_then_reshape=False,
    populate_allowed_worker_cores=False,
    use_kda_compute_config=False,
    input_from_device_producer=False,
    input_from_kda_gated_rms=False,
    use_sub_device_manager=True,
    clone_reduce_scatter_output=False,
    use_barrier_semaphore=False,
):
    torch.manual_seed(0)
    if input_from_device_producer and input_from_kda_gated_rms:
        raise ValueError("choose either the multiply or KDA gated-RMS input producer")

    tile = (32, 32)

    # Set the default config
    if mem_config_weights is None:
        mem_config_weights = mem_config_rs

    ##### Fabric setup #####
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    if use_sub_device_manager:
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
    else:
        worker_sub_device_id = None
        sub_device_stall_group = []

    # create global semaphore handles
    ccl_semaphore_handles = [create_global_semaphores(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]
    barrier_semaphore_handles = (
        [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]
        if use_barrier_semaphore
        else [None] * num_iters
    )

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    rs_num_batches = rs_input_shape[0]
    single_batch_input_shape = rs_input_shape[:]
    single_batch_input_shape[2] //= rs_num_batches
    # Line reduce-scatter carries forward and backward partials separately.
    # Its intermediate is consequently two input-sized halves; Ring uses the
    # original single-input staging shape.
    intermediate_shape = single_batch_input_shape[:]
    if rs_topology == ttnn.Topology.Linear:
        intermediate_shape[0] *= 2
    persistent_intermediate_buffers = [
        ttnn.from_torch(
            torch.zeros(intermediate_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=rs_input_dtype,
            memory_config=mem_config_rs,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(num_iters)
    ]
    rs_output_shape = rs_input_shape[:]
    rs_output_shape[3] //= num_devices
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(rs_output_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=rs_input_dtype,
            memory_config=mem_config_rs,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
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
        device=mesh_device,
        memory_config=mem_config_weights,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=mm_shard_dim),
    )

    if use_bias:
        bias_tensor_padded = torch.randn([1, 1, 1, rs_input_shape[3]]).float()
        bias_tensor_scaled = bias_tensor_padded * (1 / 8.0)
        bias_tt = ttnn.from_torch(
            bias_tensor_scaled,
            dtype=matmul_weights_dtype,
            layout=layout,
            device=mesh_device,
            memory_config=mem_config_weights,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            tile=ttnn.Tile(tile),
        )
    else:
        bias_tt = None

    ##### Configs for ttnn.matmul #####
    core_grid = (8, 6)
    in0_block_w = min(max_in0_block_w, mm_weights_shape[2] // num_devices // 32 // core_grid[0])
    per_core_M = max(1, math.ceil(rs_input_shape[2] / 32 / core_grid[1]))  # M / TILE_HEIGHT / Grid_Size
    per_core_N = max(1, math.ceil(rs_input_shape[3] / 32 / core_grid[0]))  # N / TILE_WIDTH / Grid_Size
    program_config_kwargs = {}
    if populate_allowed_worker_cores:
        program_config_kwargs["allowed_worker_cores"] = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid[0] - 1, core_grid[1] - 1))}
        )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=core_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_block_w=max(divisor for divisor in range(per_core_N // 2, 0, -1) if per_core_N % divisor == 0),
        transpose_mcast=False,
        fused_activation=None,  # ttnn.UnaryOpType.SILU,
        fuse_batch=False,
        **program_config_kwargs,
    )
    compute_kernel_config = (
        ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        if use_kda_compute_config
        else ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    )

    ##### MM input setup #####
    logger.info(f"Reduce scatter shape: {rs_input_shape}")
    logger.info(f"Reduce scatter dim: {rs_scatter_dim}")

    tt_input_tensor_mesh_list = []
    torch_input_tensor_list = []

    for i in range(num_iters):
        if input_from_kda_gated_rms:
            # This is the exact producer immediately before KDA's TP output
            # projection: each rank owns eight FP32 [T, 128] value heads and
            # a BF16 1,024-wide gate, then flattens them to its K-shard.
            heads_per_device, value_dim = 8, 128
            assert mm_weights_shape[2] == num_devices * heads_per_device * value_dim
            rms_input = torch.rand(num_devices * heads_per_device, rs_input_shape[2], value_dim)
            gate = torch.rand(1, rs_input_shape[2], mm_weights_shape[2]).bfloat16()
            rms_weight = torch.rand(1, 1, 1, value_dim).bfloat16()
            gate_by_device = torch.chunk(gate, num_devices, 2)
            input_tensors = []
            for device_index, device_heads in enumerate(torch.chunk(rms_input, num_devices, 0)):
                normalized = device_heads * torch.rsqrt(torch.mean(device_heads.square(), dim=-1, keepdim=True) + 1e-5)
                normalized = normalized * rms_weight.float().reshape(value_dim)
                gated = normalized * torch.sigmoid(
                    gate_by_device[device_index]
                    .float()
                    .reshape(rs_input_shape[2], heads_per_device, value_dim)
                    .permute(1, 0, 2)
                )
                input_tensors.append(gated.permute(1, 0, 2).reshape(1, 1, rs_input_shape[2], -1))
            torch_input_tensor_list.append(input_tensors)

            rms_input_tensor = ttnn.from_torch(
                rms_input,
                device=mesh_device,
                layout=layout,
                dtype=ttnn.float32,
                memory_config=mem_config_input,
                mesh_mapper=ttnn.create_mesh_mapper(
                    mesh_device,
                    ttnn.MeshMapperConfig(
                        [ttnn.PlacementReplicate(), ttnn.PlacementShard(0)],
                        ttnn.MeshShape(1, num_devices),
                    ),
                ),
            )
            gate_tensor = ttnn.from_torch(
                gate,
                device=mesh_device,
                layout=layout,
                dtype=ttnn.bfloat16,
                memory_config=mem_config_input,
                mesh_mapper=ttnn.create_mesh_mapper(
                    mesh_device,
                    ttnn.MeshMapperConfig(
                        [ttnn.PlacementReplicate(), ttnn.PlacementShard(2)],
                        ttnn.MeshShape(1, num_devices),
                    ),
                ),
            )
            rms_weight_tensor = ttnn.from_torch(
                rms_weight,
                device=mesh_device,
                layout=layout,
                dtype=ttnn.bfloat16,
                memory_config=mem_config_input,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            input_tensor_mesh = ttnn.transformer.kda_gated_rms_norm(
                rms_input_tensor,
                gate_tensor,
                rms_weight_tensor,
                heads_per_device,
                epsilon=1e-5,
                memory_config=mem_config_input,
                compute_kernel_config=compute_kernel_config,
            )
            input_tensor_mesh = ttnn.reshape(
                input_tensor_mesh,
                (1, 1, rs_input_shape[2], mm_weights_shape[2] // num_devices),
            )
        else:
            mm_input_shape = [rs_input_shape[0], 1, rs_input_shape[2], mm_weights_shape[2]]
            mm_input_tensor = torch.rand(mm_input_shape).bfloat16()
            input_tensors = torch.chunk(mm_input_tensor, num_devices, 3)
            torch_input_tensor_list.append(input_tensors)

            if input_rank3_then_reshape:
                input_tensor_mesh = ttnn.from_torch(
                    mm_input_tensor.squeeze(1),
                    device=mesh_device,
                    layout=layout,
                    dtype=rs_input_dtype,
                    memory_config=mem_config_input,
                    mesh_mapper=ttnn.create_mesh_mapper(
                        mesh_device,
                        ttnn.MeshMapperConfig(
                            [ttnn.PlacementReplicate(), ttnn.PlacementShard(2)], ttnn.MeshShape(1, num_devices)
                        ),
                    ),
                )
                input_tensor_mesh = ttnn.reshape(
                    input_tensor_mesh,
                    (1, 1, rs_input_shape[2], mm_weights_shape[2] // num_devices),
                )
            else:
                input_tensor_mesh = ttnn.from_torch(
                    mm_input_tensor,
                    device=mesh_device,
                    layout=layout,
                    dtype=rs_input_dtype,
                    memory_config=mem_config_input,
                    mesh_mapper=ttnn.create_mesh_mapper(
                        mesh_device,
                        ttnn.MeshMapperConfig(
                            [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], ttnn.MeshShape(1, num_devices)
                        ),
                    ),
                )

        tt_input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform torch ops #####
    torch_reduce_scatter_output_list = []
    torch_matmul_output_list = []
    for i in range(num_iters):
        matmul_input = torch.cat(torch_input_tensor_list[i], dim=3)
        if input_from_device_producer:
            matmul_input = matmul_input * matmul_input
        matmul_weight = weights_tensor.float() if matmul_input.dtype == torch.float32 else weights_tensor
        if use_bias:
            matmul_output = torch.matmul(matmul_input, matmul_weight) + bias_tensor_padded
        else:
            matmul_output = torch.matmul(matmul_input, matmul_weight)
        scatter_output = torch.chunk(matmul_output, num_devices, rs_scatter_dim)
        torch_reduce_scatter_output_list.append(scatter_output)
        torch_matmul_output_list.append(matmul_output)

    ##### Perform the TT ops #####
    tt_reduce_scatter_output_list = []
    tt_matmul_output_list = []

    def run_op(i):
        matmul_input_tensor = tt_input_tensor_mesh_list[i]
        if input_from_device_producer:
            matmul_input_tensor = ttnn.multiply(
                matmul_input_tensor,
                matmul_input_tensor,
                dtype=rs_input_dtype,
                memory_config=mem_config_input,
            )
        if use_non_fused:
            tt_matmul_out_tensor = ttnn.linear(
                matmul_input_tensor,
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
                matmul_input_tensor,
                weight_tt,
                persistent_intermediate_buffer=persistent_intermediate_buffers[i],
                persistent_output_buffer=persistent_output_buffers[i],
                dim=rs_scatter_dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                barrier_semaphore=barrier_semaphore_handles[i],
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

        if clone_reduce_scatter_output:
            tt_reduce_scatter_output_tensor = ttnn.clone(
                tt_reduce_scatter_output_tensor,
                memory_config=mem_config_rs,
            )
        return tt_matmul_out_tensor, tt_reduce_scatter_output_tensor

    if enable_trace:
        # Compile the op
        for i in range(num_iters):
            tt_matmul_out_tensor, tt_reduce_scatter_output_tensor = run_op(i)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_matmul_out_tensor, tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_list.append(tt_reduce_scatter_output_tensor)
            tt_matmul_output_list.append(tt_matmul_out_tensor)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")

        # Synchronize the devices
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
    else:
        for i in range(num_iters):
            tt_matmul_out_tensor, tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_list.append(tt_reduce_scatter_output_tensor)
            tt_matmul_output_list.append(tt_matmul_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_mm_out_tensor = tt_matmul_output_list[i]
        torch_mm_out_tensor = torch_matmul_output_list[i]

        tt_mm_out = ttnn.from_device(tt_mm_out_tensor)
        tt_mm_out = ttnn.to_torch(tt_mm_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
        tt_mm_out = torch.sum(torch.stack(torch.chunk(tt_mm_out, num_devices, 3)), dim=0)
        eq, output = comp_pcc(tt_mm_out, torch_mm_out_tensor)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED mm: {output}"
        tt_rs_out_tensor = tt_reduce_scatter_output_list[i]
        torch_rs_out_tensor = torch_reduce_scatter_output_list[i]

        torch_rs_out = torch.cat(torch_rs_out_tensor, 3)

        tt_rs_out = ttnn.from_device(tt_rs_out_tensor)
        tt_rs_out = ttnn.to_torch(tt_rs_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
        assert torch.isfinite(tt_rs_out).all(), f"{i} fused reduce-scatter produced non-finite output"
        eq, output = comp_pcc(tt_rs_out, torch_rs_out)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED ag: {output}"
        if check_first_output_tile:
            first_tile_eq, first_tile_pcc = comp_pcc(tt_rs_out[..., :32, :], torch_rs_out[..., :32, :])
            assert first_tile_eq, f"{i} FAILED first output tile: {first_tile_pcc}"

        # print(f"RS TORCH TENSOR {torch_rs_out}")
        # print(f"RS TT TENSOR {tt_rs_out}")

    if use_sub_device_manager:
        mesh_device.reset_sub_device_stall_group()
        mesh_device.clear_loaded_sub_device_manager()


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
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 330000}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 500000}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_reduce_scatter_async(
    mesh_device,
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
    rs_topology,
):
    run_reduce_scatter_impl(
        mesh_device,
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
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        use_non_fused=False,
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("enable_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize("child_start", [0, 4], ids=["child_0", "child_4"])
def test_linear_matmul_reduce_scatter_tp4_child_mesh(mesh_device, child_start, enable_trace):
    """Exercise the KDA output-projection MRS shape on one LoudBox child mesh.

    KDA TP=4 is row parallel: every chip owns 1,024 of the 4,096 input
    features, produces all 2,304 output features, then reduces/scatters the
    output width.  Keep this minimal repro independent of KDA's recurrent
    scheduling so a first-tile failure is attributable to fused Line MRS.
    """
    child_mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, child_start))
    run_reduce_scatter_impl(
        child_mesh,
        num_devices=4,
        rs_input_shape=[1, 1, 640, 2304],
        mm_shard_dim=2,
        rs_scatter_dim=3,
        num_links=2,
        mm_weights_shape=[1, 1, 4096, 2304],
        rs_input_dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        matmul_weights_dtype=ttnn.bfloat16,
        max_in0_block_w=4,
        use_bias=False,
        mem_config_input=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        rs_topology=ttnn.Topology.Linear,
        use_non_fused=False,
        num_iters=1,
        enable_trace=enable_trace,
        check_first_output_tile=True,
        # Match the KDA epilogue's local-width reshape, explicit worker grid,
        # Blackhole compute config, on-device producer, no subdevice manager,
        # and clone of the persistent RS output.
        input_rank3_then_reshape=True,
        populate_allowed_worker_cores=True,
        use_kda_compute_config=True,
        input_from_device_producer=True,
        use_sub_device_manager=False,
        clone_reduce_scatter_output=True,
        use_barrier_semaphore=True,
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("enable_trace", [False, True], ids=["eager", "trace"])
@pytest.mark.parametrize("child_start", [0, 4], ids=["child_0", "child_4"])
def test_linear_matmul_reduce_scatter_tp4_kda_gated_rms_producer(mesh_device, child_start, enable_trace):
    """Exercise Line MRS directly after KDA's FP32 gated-RMS producer."""
    child_mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, child_start))
    run_reduce_scatter_impl(
        child_mesh,
        num_devices=4,
        rs_input_shape=[1, 1, 640, 2304],
        mm_shard_dim=2,
        rs_scatter_dim=3,
        num_links=2,
        mm_weights_shape=[1, 1, 4096, 2304],
        rs_input_dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        matmul_weights_dtype=ttnn.bfloat16,
        max_in0_block_w=4,
        use_bias=False,
        mem_config_input=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        rs_topology=ttnn.Topology.Linear,
        use_non_fused=False,
        num_iters=1,
        enable_trace=enable_trace,
        check_first_output_tile=True,
        populate_allowed_worker_cores=True,
        use_kda_compute_config=True,
        input_from_kda_gated_rms=True,
        use_sub_device_manager=False,
        clone_reduce_scatter_output=True,
        use_barrier_semaphore=True,
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 1024 * 1024}],
    indirect=True,
)
def test_linear_matmul_reduce_scatter_tp4_after_sp_socket_handoff(mesh_device):
    """Reproduce the two TP4 output consumers after an SP cache handoff.

    The integrated KDA failure affects the second child after it has received
    the first span's FP32 recurrent cache.  Keep the exact gated-RMS producer,
    Line MRS parameters, child placement, and socket ordering here, while
    removing the unrelated KDA recurrence schedule.
    """
    source = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    destination = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 4))
    send_socket, recv_socket = ttnn.create_socket_pair(source, destination, _socket_config(source.shape))
    recurrent_state = torch.rand(1, 8, 128, 128)
    ttnn.experimental.send_async(
        ttnn.from_torch(
            recurrent_state,
            device=source,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(source),
        ),
        send_socket,
    )
    ttnn.experimental.recv_async(
        ttnn.from_torch(
            torch.zeros_like(recurrent_state),
            device=destination,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(destination),
        ),
        recv_socket,
    )
    common = dict(
        num_devices=4,
        rs_input_shape=[1, 1, 640, 2304],
        mm_shard_dim=2,
        rs_scatter_dim=3,
        num_links=2,
        mm_weights_shape=[1, 1, 4096, 2304],
        rs_input_dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        matmul_weights_dtype=ttnn.bfloat16,
        max_in0_block_w=4,
        use_bias=False,
        mem_config_input=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        mem_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        rs_topology=ttnn.Topology.Linear,
        use_non_fused=False,
        num_iters=1,
        enable_trace=False,
        check_first_output_tile=True,
        populate_allowed_worker_cores=True,
        use_kda_compute_config=True,
        input_from_kda_gated_rms=True,
        use_sub_device_manager=False,
        clone_reduce_scatter_output=True,
        use_barrier_semaphore=True,
    )
    run_reduce_scatter_impl(source, **common)
    run_reduce_scatter_impl(destination, **common)
