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
    fp32_acc=False,
    num_iters=1,
    enable_trace=False,
    use_persistent_buffers=True,
    use_non_fused=False,
    force_transpose=True,
    sp_axis=0,
    tp_axis=1,
    torch_dtype=torch.float32,
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
    per_device_M_buffer = per_device_M
    if use_persistent_buffers:
        persistent_output_buffers = [
            ttnn.from_torch(
                torch.zeros((1, 1, per_device_M_buffer, K), dtype=torch_dtype)
                if use_non_fused
                else torch.zeros((per_device_M_buffer, K), dtype=torch_dtype),
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

    with torch.no_grad():
        torch_output = torch_input @ weight_input
        if bias_input is not None:
            torch_output = torch_output + bias_input

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
            # Fused: we shard (M, K) with M on sp_axis, K on tp_axis; output (per_device_M, N) per device -> reassemble as (M, N*mesh_cols)
            concat_dims = [sp_axis, tp_axis]

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
    fp32_acc=False,
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

    # Non-fused all_gather_async(dim=3) gathers along the ring (cluster_axis); so K must be sharded on the ring to get full K after one gather.
    # Use [sp_axis+2, tp_axis+2] so K (dim 3) is on tp_axis = ring; then per device (1,1, M/8, K/4), gather gives (1,1, M/8, K).
    if sp_axis == 1:
        if use_non_fused:
            shard_dims = [sp_axis + 2, tp_axis + 2]
        else:
            shard_dims = [sp_axis, tp_axis]
    else:
        if use_non_fused:
            shard_dims = [sp_axis + 2, tp_axis + 2]  # M on mesh 0, K on mesh 1 (ring) so gather dim=3 gives full K
        else:
            # Fused: shard (M, K) with M on sp_axis, K on tp_axis so op's K = per_device_K * ring_size matches K_w
            shard_dims = [sp_axis, tp_axis]
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
        # Llama 70B config: 8x4 mesh with 4 workers (less L1 usage)
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            1,
            4,  # num_workers_per_link=4 like Llama
            0,
            1,
            8,
            8,
            1,
        ],
        # 4x8 core grid (32 cores) - current working config in Llama
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            1,
            4,  # num_workers_per_link=4 like Llama
            0,
            1,
            4,  # core_grid_x = 4
            8,  # core_grid_y = 8
            1,
        ],
        # 7x8 core grid (56 cores) - test if grid_x=7 works
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            1,
            4,
            0,
            1,
            7,  # core_grid_x = 7
            8,  # core_grid_y = 8
            1,
        ],
        # 8x4 core grid (32 cores) - swap x and y
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            1,
            4,
            0,
            1,
            8,  # core_grid_x = 8
            4,  # core_grid_y = 4
            1,
        ],
        # 2x8 core grid (16 cores) - test grid_x=2
        [
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            1,
            4,
            0,
            1,
            2,  # core_grid_x = 2
            8,  # core_grid_y = 8
            1,
        ],
    ],
    ids=["2x4links1", "wh8x4links1", "llama_8x4", "llama_4x8grid", "llama_7x8grid", "llama_8x4grid", "llama_2x8grid"],
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
        # Llama 70B W2 sizes for 8x4 mesh: M=65536 (8192*8), K=3584, N=2048
        # Per device: M=8192, K=896 (before gather), N=2048
        (65536, 3584, 2048, True, True, None),
        # Llama 70B W2 with K padded to 4096 (power of 2)
        (65536, 4096, 2048, True, True, None),
    ],
    ids=[
        "4k4k4k",
        "qkv",
        "denseout",
        "ff1",
        "unit",
        "llama70b_w2_8x4",
        "llama70b_w2_k4096",
    ],
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
):
    # 8x4 fused on wormhole exceeds fabric L1; separate path is supported
    # COMMENTED OUT to test if Llama 70B sizes work with 8x8 grid
    # if (
    #     not use_non_fused
    #     and tuple(mesh_device.shape) == (8, 4)
    #     and mesh_device.arch() == ttnn.device.Arch.WORMHOLE_B0
    # ):
    #     pytest.skip(
    #         "8x4 fused exceeds fabric L1 on wormhole (memory_map_end_address > l1_end_address); run separate path"
    #     )

    print(f"M,K,N,h,w: {M_block_size}, {K_block_size}, {N_block_size}, {subblock_h},  {subblock_w}\n")

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
    )

    for n in range(num_iters):
        for i in range(mesh_device.get_num_devices()):
            assert check_result[n][i]["pcc"] > 0.999_500
            assert check_result[n][i]["relative_rmse"] < 0.02
