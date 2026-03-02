# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def assert_quality(torch_output, tt_output):
    pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    relative_rmse_val = torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / torch_output.std().item()
    logger.info(f"PCC: {pcc_val:.7f}, Relative RMSE: {relative_rmse_val:.4f}")
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
    }


def run_test_linear_impl(
    device,
    torch_input,
    weight_input,
    bias_input,
    tt_input,
    tt_weight,
    tt_bias,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    num_devices,
    num_links,
    topology,
    cluster_axis,
    input_dtype,
    core_grid,
    num_workers_per_link,
    activation=None,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_acc=True,
    num_iters=1,
    enable_trace=False,
    use_persistent_buffers=True,
    use_non_fused=False,
    force_transpose=True,
    sp_axis=0,
    tp_axis=1,
    torch_dtype=torch.float32,
    fuse_addcmul=False,
    torch_addcmul_a=None,
    torch_addcmul_b=None,
    addcmul_scalar=1.0,
):
    ccl_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
    )

    # create global semaphore handles
    ccl_semaphore_handles = [create_global_semaphores(device, num_devices, ccl_cores, 0) for _ in range(num_iters)]

    barrier_semaphore_handles = [ttnn.create_global_semaphore(device, ccl_cores, 0) for _ in range(num_iters)]

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    M = torch_input.shape[2] if use_non_fused else torch_input.shape[0]
    K = torch_input.shape[3] if use_non_fused else torch_input.shape[1]
    N = weight_input.shape[3] if use_non_fused else weight_input.shape[1]
    per_device_M = M // device.shape[sp_axis]
    if use_persistent_buffers:
        persistent_output_buffers = [
            ttnn.from_torch(
                torch.zeros((1, 1, per_device_M, K), dtype=torch_dtype)
                if use_non_fused
                else torch.zeros((per_device_M, K), dtype=torch_dtype),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=input_dtype,
                memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=[None, None]),
            )
            for _ in range(num_iters)
        ]
    else:
        persistent_output_buffers = []

    activation_fn = None
    if activation == "gelu":
        activation_fn = (ttnn.UnaryOpType.GELU, False)
    else:
        assert activation is None, f"Unsupported activation: {activation}"

    if fuse_addcmul:
        if sp_axis == 1:
            if use_non_fused:
                shard_dims = [None, tp_axis + 2]
            else:
                shard_dims = [None, tp_axis]
        else:
            if use_non_fused:
                shard_dims = [sp_axis + 2, None]
            else:
                shard_dims = [sp_axis, None]
        tt_addcmul_a = ttnn.from_torch(
            torch_addcmul_a,
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=shard_dims),
        )
        tt_addcmul_b = ttnn.from_torch(torch_addcmul_b, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        tt_addcmul_a = None
        tt_addcmul_b = None
        addcmul_scalar = None

    with torch.no_grad():
        torch_output = torch_input @ weight_input
        if bias_input is not None:
            torch_output = torch_output + bias_input
        if fuse_addcmul:
            torch_output = torch.addcmul(torch_addcmul_a, torch_output, torch_addcmul_b, value=addcmul_scalar)

        if activation == "gelu":
            torch_output = torch.nn.functional.gelu(torch_output)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )

    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=core_grid,
    )

    tt_output_tensor_list = []

    def run_op(i):
        if use_non_fused:
            tt_all_gather_out_tensor = ttnn.experimental.all_gather_async(
                tt_input,
                persistent_output_buffer=persistent_output_buffers[i],
                dim=3,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                topology=topology,
                cluster_axis=cluster_axis,
                chunks_per_sync=16,
                num_workers_per_link=3,
                num_buffers_per_channel=2,
            )

            if fuse_addcmul:
                tt_output = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                    tt_all_gather_out_tensor,
                    tt_weight,
                    addcmul_scalar,
                    tt_addcmul_a,
                    tt_addcmul_b,
                    bias_tensor=tt_bias,
                    fused_activation=activation_fn,
                    config=matmul_config,
                    compute_kernel_config=compute_config,
                )
            else:
                tt_output = ttnn.experimental.minimal_matmul(
                    tt_all_gather_out_tensor,
                    tt_weight,
                    bias_tensor=tt_bias,
                    fused_activation=activation_fn,
                    compute_kernel_config=compute_config,
                    config=matmul_config,
                )

        else:
            tt_output = ttnn.experimental.all_gather_minimal_matmul_async(
                tt_input,
                tt_weight,
                bias_tensor=tt_bias,
                fused_activation=activation_fn,
                compute_kernel_config=compute_config,
                config=matmul_config,
                persistent_output_buffer=persistent_output_buffers[i],
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                topology=topology,
                cluster_axis=cluster_axis,
                barrier_semaphore=barrier_semaphore_handles[0] if not use_persistent_buffers else None,
                force_transpose=force_transpose,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=48,
                scalar=addcmul_scalar,
                addcmul_input_tensor1=tt_addcmul_a,
                addcmul_input_tensor2=tt_addcmul_b,
            )

        return tt_output

    if enable_trace:
        # Compile the op
        run_op(0)
        ttnn.synchronize_device(device)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        tt_out_tensor = run_op(0)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)
        logger.info(f"Done capturing trace")

        # Execute trace
        for i in range(num_iters):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            tt_output_tensor_list.append(tt_out_tensor)
        logger.info(f"Done executing trace")
    else:
        for i in range(num_iters):
            ttnn.synchronize_device(device)
            tt_out_tensor = run_op(i)
            tt_output_tensor_list.append(tt_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(device)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    # Check results
    check_result_list = []
    for n in range(num_iters):
        print(f"iteration {n}:")
        tt_output = tt_output_tensor_list[n]

        if use_non_fused:
            if cluster_axis == 0:
                concat_dims = [sp_axis + 2, tp_axis + 2]
            else:
                concat_dims = [tp_axis + 2, sp_axis + 2]
        else:
            if cluster_axis == 0:
                concat_dims = [sp_axis, tp_axis]
            else:
                concat_dims = [tp_axis, sp_axis]

        tt_output = ttnn.from_device(tt_output)
        tt_output = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(device.shape), dims=concat_dims),
        )
        check_result = []

        for i in range(device.shape[sp_axis]):
            for j in range(device.shape[tp_axis]):
                m_slice = slice(i * per_device_M, (i + 1) * per_device_M)
                n_slice = slice(j * N, (j + 1) * N)

                if use_non_fused:
                    idx = (slice(None), slice(None), m_slice, n_slice)
                else:
                    idx = (m_slice, n_slice)

                tt_device_output = tt_output[idx]

                check_result.append(
                    assert_quality(
                        torch_output[:, :, i * per_device_M : (i + 1) * per_device_M, :]
                        if use_non_fused
                        else torch_output[i * per_device_M : (i + 1) * per_device_M, :],
                        tt_device_output,
                    )
                )
        check_result_list.append(check_result)

    return check_result_list


def run_test_linear(
    device,
    M,
    K,
    N,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    topology,
    core_grid,
    num_workers_per_link,
    num_links,
    use_bias=True,
    activation=None,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_acc=True,
    dtype=ttnn.bfloat16,
    weight_dtype=None,
    bias_dtype=None,
    use_non_fused=False,
    force_transpose=True,
    sp_axis=0,
    tp_axis=1,
    num_iters=1,
    enable_trace=False,
    cluster_axis=1,
    fuse_addcmul=False,
    addcmul_scalar=1.0,
):
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_dtype = torch.float32

    if use_non_fused:
        torch_input = torch.randn((1, 1, M, K), dtype=torch_dtype)
        weight_input = torch.randn((1, 1, K, N), dtype=torch_dtype)
    else:
        torch_input = torch.randn((M, K), dtype=torch_dtype)
        weight_input = torch.randn((K, N), dtype=torch_dtype)
    bias_input = None
    if use_bias:
        if use_non_fused:
            bias_input = torch.randn((1, 1, 1, N), dtype=torch_dtype)
        else:
            bias_input = torch.randn((1, N), dtype=torch_dtype)

    if fuse_addcmul:
        if use_non_fused:
            torch_addcmul_a = torch.randn(1, 1, M, N, dtype=torch.bfloat16)  # base value (full shape)
            torch_addcmul_b = torch.randn(1, 1, 1, N, dtype=torch.bfloat16)  # gate (broadcast like bias)
        else:
            torch_addcmul_a = torch.randn(M, N, dtype=torch.bfloat16)  # base value (full shape)
            torch_addcmul_b = torch.randn(1, N, dtype=torch.bfloat16)  # gate (broadcast like bias)
    else:
        torch_addcmul_a = None
        torch_addcmul_b = None

    # Prepare TT tensors
    if sp_axis == 1:
        if use_non_fused:
            shard_dims = [sp_axis + 2, tp_axis + 2]
        else:
            shard_dims = [sp_axis, tp_axis]
    else:
        if use_non_fused:
            shard_dims = [tp_axis + 2, sp_axis + 2]
        else:
            shard_dims = [tp_axis, sp_axis]
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=shard_dims),
    )

    tt_weight = ttnn.from_torch(weight_input, dtype=weight_dtype or dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = None
    if use_bias:
        tt_bias = ttnn.from_torch(bias_input, dtype=bias_dtype or dtype, device=device, layout=ttnn.TILE_LAYOUT)

    return run_test_linear_impl(
        device=device,
        torch_input=torch_input,
        weight_input=weight_input,
        bias_input=bias_input,
        tt_input=tt_input,
        tt_weight=tt_weight,
        tt_bias=tt_bias,
        M_block_size=M_block_size,
        K_block_size=K_block_size,
        N_block_size=N_block_size,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        activation=activation,
        math_fidelity=math_fidelity,
        fp32_acc=fp32_acc,
        core_grid=core_grid,
        input_dtype=dtype,
        num_devices=device.get_num_devices(),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        num_workers_per_link=num_workers_per_link,
        use_non_fused=use_non_fused,
        force_transpose=force_transpose,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        torch_dtype=torch_dtype,
        num_iters=num_iters,
        enable_trace=enable_trace,
        fuse_addcmul=fuse_addcmul,
        torch_addcmul_a=torch_addcmul_a,
        torch_addcmul_b=torch_addcmul_b,
        addcmul_scalar=addcmul_scalar,
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
    [
        [
            (2, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
            ttnn.Topology.Ring,
            1,
            4,
            0,
            1,
            4,
            4,
            1,
        ],
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            1,
            8,
            0,
            1,
            8,
            8,
            1,
        ],
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            2,
            4,
            0,
            1,
            8,
            8,
            1,
        ],
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            4,
            2,
            0,
            1,
            8,
            8,
            1,
        ],
        [
            (4, 8),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            2,
            6,
            1,
            0,
            12,
            9,
            0,
        ],
    ],
    ids=[
        "2x4links1",
        "wh8x4links1",
        "wh8x4links2",
        "wh8x4links4",
        "bh8x4links2",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N, force_transpose, use_bias, activation",
    [
        (32768, 4096, 4096, True, False, None),
        (75776, 5120, 3840, True, True, None),
        (75776, 5120, 1280, True, True, None),
        (75776, 5120, 3456, True, True, "gelu"),
        (3072, 5120, 3456, True, True, "gelu"),
    ],
    ids=[
        "4k4k4k",
        "qkv",
        "denseout",
        "ff1",
        "unit",
    ],
)
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(10, 8, 8, 2, 1)],
)
@pytest.mark.parametrize(
    "use_non_fused",
    [
        True,
        False,
    ],
    ids=["separate", "fused"],
)
@pytest.mark.parametrize(
    "fuse_addcmul",
    [
        True,
        False,
    ],
    ids=["addcmul", "noternary"],
)
@pytest.mark.parametrize(
    "enable_trace,num_iters",
    [
        (False, 1),
        (True, 2),
    ],
    ids=["check", "perf"],
)
def test_linear(
    mesh_device,
    M,
    K,
    N,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    topology,
    core_grid_x,
    core_grid_y,
    num_workers_per_link,
    num_links,
    use_non_fused,
    force_transpose,
    sp_axis,
    tp_axis,
    use_bias,
    activation,
    enable_trace,
    num_iters,
    cluster_axis,
    fuse_addcmul,
):
    check_result = run_test_linear(
        mesh_device,
        M,
        K,
        N,
        M_block_size,
        K_block_size,
        N_block_size,
        subblock_h,
        subblock_w,
        topology,
        core_grid=ttnn.CoreCoord(core_grid_x, core_grid_y),
        num_workers_per_link=num_workers_per_link,
        num_links=num_links,
        use_non_fused=use_non_fused,
        force_transpose=force_transpose,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        use_bias=use_bias,
        activation=activation,
        enable_trace=enable_trace,
        num_iters=num_iters,
        cluster_axis=cluster_axis,
        fuse_addcmul=fuse_addcmul,
    )

    for n in range(num_iters):
        for i in range(mesh_device.get_num_devices()):
            assert check_result[n][i]["pcc"] > 0.999_500
            assert check_result[n][i]["relative_rmse"] < 0.02
