# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu


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
    chunks=1,
    broadcast_gate=True,
    fuse_swiglu=False,
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
    if fuse_swiglu:
        # weight is the packed [gate|up] of width 2*out_N; the fused op emits out_N.
        N = N // 2
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
        if broadcast_gate:
            tt_addcmul_b = ttnn.from_torch(torch_addcmul_b, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        else:
            tt_addcmul_b = ttnn.from_torch(
                torch_addcmul_b,
                dtype=input_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=shard_dims),
            )
    else:
        tt_addcmul_a = None
        tt_addcmul_b = None
        addcmul_scalar = None

    with torch.no_grad():
        torch_output = torch_input @ weight_input
        if bias_input is not None:
            torch_output = torch_output + bias_input
        if fuse_swiglu:
            first, second = torch.chunk(torch_output, 2, dim=-1)
            torch_output = first * torch.nn.functional.silu(second)
        if fuse_addcmul:
            torch_output = torch.addcmul(torch_addcmul_a, torch_output, torch_addcmul_b, value=addcmul_scalar)

        if activation == "gelu":
            torch_output = torch.nn.functional.gelu(torch_output)

        torch_output = torch.chunk(torch_output, chunks, dim=-1)

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
                num_workers_per_link=4,
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
                if chunks > 1:
                    tt_output = ttnn.experimental.minimal_matmul_split(
                        tt_all_gather_out_tensor,
                        tt_weight,
                        chunks=chunks,
                        dim=-1,
                        bias_tensor=tt_bias,
                        fused_activation=activation_fn,
                        compute_kernel_config=compute_config,
                        config=matmul_config,
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
            if chunks == 1:
                tt_output = [tt_output]

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
                chunks=chunks,
                fuse_swiglu=fuse_swiglu,
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
            # Fused AGMM output: M on non-cluster axis, N on cluster axis
            concat_dims = [0, 0]
            concat_dims[1 - cluster_axis] = 0  # M gathered on non-cluster axis
            concat_dims[cluster_axis] = 1  # N on cluster axis

        check_result = []
        for c in range(chunks):
            tt_output_chunk = ttnn.from_device(tt_output[c])
            tt_output_chunk = ttnn.to_torch(
                tt_output_chunk,
                mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(device.shape), dims=concat_dims),
            )
            check_result_chunk = []

            for i in range(device.shape[sp_axis]):
                for j in range(device.shape[tp_axis]):
                    m_slice = slice(i * per_device_M, (i + 1) * per_device_M)
                    n_slice = slice(j * (N // chunks), (j + 1) * (N // chunks))

                    if use_non_fused:
                        idx = (slice(None), slice(None), m_slice, n_slice)
                    else:
                        idx = (m_slice, n_slice)

                    tt_device_output = tt_output_chunk[idx]

                    check_result_chunk.append(
                        assert_quality(
                            torch_output[c][:, :, i * per_device_M : (i + 1) * per_device_M, :]
                            if use_non_fused
                            else torch_output[c][i * per_device_M : (i + 1) * per_device_M, :],
                            tt_device_output,
                        )
                    )
            check_result.append(check_result_chunk)
        check_result_list.append(check_result)

    return check_result_list


def _create_cluster_submesh(mesh_device, cluster_axis):
    """Create a 1xN (or Nx1) submesh sized to the cluster axis ring.

    The op only operates along cluster_axis, so the non-cluster axis is just
    redundant compute. Sub-meshing keeps the test focused on a single ring.
    """
    submesh_shape = [1, 1]
    submesh_shape[cluster_axis] = mesh_device.shape[cluster_axis]
    return mesh_device.create_submesh(ttnn.MeshShape(tuple(submesh_shape)))


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
    chunks=1,
    broadcast_gate=True,
    fuse_swiglu=False,
):
    logger.info(f"Running test_linear with M={M}, K={K}, N={N}")
    torch_dtype = torch.float32

    # For fused SwiGLU the weight packs [gate|up] -> width 2N; the op emits out width N.
    weight_N = 2 * N if fuse_swiglu else N

    if use_non_fused:
        torch_input = torch.randn((1, 1, M, K), dtype=torch_dtype)
        weight_input = torch.randn((1, 1, K, weight_N), dtype=torch_dtype)
    else:
        torch_input = torch.randn((M, K), dtype=torch_dtype)
        weight_input = torch.randn((K, weight_N), dtype=torch_dtype)
    bias_input = None
    if use_bias:
        if use_non_fused:
            bias_input = torch.randn((1, 1, 1, weight_N), dtype=torch_dtype)
        else:
            bias_input = torch.randn((1, weight_N), dtype=torch_dtype)

    if fuse_addcmul:
        if use_non_fused:
            torch_addcmul_a = torch.randn(1, 1, M, N, dtype=torch.bfloat16)  # base value (full shape)
            if broadcast_gate:
                torch_addcmul_b = torch.randn(1, 1, 1, N, dtype=torch.bfloat16)  # gate (broadcast like bias)
            else:
                torch_addcmul_b = torch.randn(1, 1, M, N, dtype=torch.bfloat16)  # gate (full, no broadcast)
        else:
            torch_addcmul_a = torch.randn(M, N, dtype=torch.bfloat16)  # base value (full shape)
            if broadcast_gate:
                torch_addcmul_b = torch.randn(1, N, dtype=torch.bfloat16)  # gate (broadcast like bias)
            else:
                torch_addcmul_b = torch.randn(M, N, dtype=torch.bfloat16)  # gate (full, no broadcast)
    else:
        torch_addcmul_a = None
        torch_addcmul_b = None

    # Prepare TT tensors
    if use_non_fused:
        if sp_axis == 1:
            shard_dims = [sp_axis + 2, tp_axis + 2]
        else:
            shard_dims = [tp_axis + 2, sp_axis + 2]
    else:
        # Fused AGMM gathers K (last dim) across cluster_axis
        shard_dims = [0, 0]
        shard_dims[cluster_axis] = 1  # K on cluster_axis
        shard_dims[1 - cluster_axis] = 0  # M on the other axis
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=shard_dims),
    )

    # Fused SwiGLU: tile-pair interleave the (replicated) weight/bias so each ring device's
    # N-slice holds whole [second(silu'd), first] pairs. ndev = ring size (cluster_axis).
    weight_to_load = weight_input
    bias_to_load = bias_input
    if fuse_swiglu:
        ring_size = device.shape[cluster_axis]
        weight_2d = weight_input.reshape(K, weight_input.shape[-1])
        weight_to_load = prepare_for_fused_swiglu(weight_2d, ndev=ring_size).reshape(weight_input.shape)
        if use_bias:
            bias_to_load = prepare_for_fused_swiglu(bias_input.reshape(1, -1), ndev=ring_size).reshape(bias_input.shape)

    tt_weight = ttnn.from_torch(weight_to_load, dtype=weight_dtype or dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = None
    if use_bias:
        tt_bias = ttnn.from_torch(bias_to_load, dtype=bias_dtype or dtype, device=device, layout=ttnn.TILE_LAYOUT)

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
        chunks=chunks,
        broadcast_gate=broadcast_gate,
        fuse_swiglu=fuse_swiglu,
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
            (4, 8),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            1,
            8,
            1,
            0,
            8,
            8,
            0,
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
            4,
            1,
            0,
            8,
            8,
            0,
        ],
        [
            (4, 8),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Ring,
            4,
            2,
            1,
            0,
            8,
            8,
            0,
        ],
        [
            (4, 8),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Linear,
            4,
            2,
            1,
            0,
            8,
            8,
            0,
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
        "wh4x8links1",
        "wh4x8links2",
        "wh4x8links4_ring",
        "wh4x8links4_linear",
        "bh4x8links2",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N, force_transpose, use_bias, activation, chunks, fuse_addcmul, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [
        (32768, 4096, 4096, True, False, None, 1, False, 8, 8, 8, 2, 2),
        (75776, 5120, 3840, True, True, None, 3, False, 7, 5, 16, 1, 2),
        (75776, 5120, 1280, True, True, None, 1, True, 10, 8, 8, 2, 1),
        (75776, 5120, 1280, True, True, None, 1, False, 10, 8, 8, 2, 1),
        (75776, 5120, 3456, True, True, "gelu", 1, False, 9, 5, 12, 1, 2),
        (3072, 5120, 3456, True, True, "gelu", 1, False, 8, 8, 8, 2, 2),
        (18944, 5120, 3840, True, True, None, 3, False, 7, 5, 16, 1, 2),
        (18944, 5120, 1280, True, True, None, 1, True, 10, 8, 6, 2, 1),
        (18944, 5120, 1280, True, True, None, 1, False, 10, 8, 6, 2, 1),
        (18944, 5120, 3456, True, True, "gelu", 1, False, 7, 5, 12, 1, 2),
        (49920, 5120, 3840, True, True, None, 3, False, 7, 5, 16, 1, 2),
        (49920, 5120, 1280, True, True, None, 1, True, 10, 8, 6, 2, 1),
        (49920, 5120, 1280, True, True, None, 1, False, 10, 8, 6, 2, 1),
        (49920, 5120, 3456, True, True, "gelu", 1, False, 7, 5, 12, 1, 2),
        (115200, 5120, 3840, True, True, None, 3, False, 7, 5, 16, 1, 2),
        (115200, 5120, 1280, True, True, None, 1, True, 10, 8, 6, 2, 1),
        (115200, 5120, 1280, True, True, None, 1, False, 10, 8, 6, 2, 1),
        (115200, 5120, 3456, True, True, "gelu", 1, False, 7, 5, 12, 1, 2),
        # K-fractured-across-4-devices shapes (K_block_size chosen to evenly
        # divide K-tiles per device: 40 for K=5120, 27 for K=3456).
        (3072, 5120, 3840, True, True, None, 1, False, 8, 8, 8, 2, 2),
        (3072, 5120, 1280, True, True, None, 1, False, 8, 8, 8, 2, 2),
        (3072, 5120, 3456, True, True, "gelu", 1, False, 8, 8, 8, 2, 2),
        (3072, 3456, 5120, True, True, None, 1, False, 8, 9, 8, 2, 2),
    ],
    ids=[
        "4k4k4k",
        "1xqkv",
        "1xdenseattn1",
        "1xdenseattn2",
        "1xff1",
        "unit",
        "4xqkv",
        "4xdenseattn1",
        "4xdenseattn2",
        "4xff1",
        "1xssg480pqkv",
        "1xssg480pdenseattn1",
        "1xssg480pdenseattn2",
        "1xssg480pff1",
        "1xssg720pqkv",
        "1xssg720pdenseattn1",
        "1xssg720pdenseattn2",
        "1xssg720pff1",
        "3072x5120x7680",
        "3072x5120x2560",
        "3072x5120x6912",
        "3072x3456x5120",
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
    fuse_addcmul,
    chunks,
):
    submesh = _create_cluster_submesh(mesh_device, cluster_axis)
    check_result = run_test_linear(
        submesh,
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
        chunks=chunks,
    )

    for n in range(num_iters):
        for c in range(chunks):
            for i in range(submesh.get_num_devices()):
                assert check_result[n][c][i]["pcc"] > 0.999_500
                assert check_result[n][c][i]["relative_rmse"] < 0.02


def run_test_linear_fsdp(
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
    *,
    tp_axis,
    fsdp_axis,
    fsdp_topology,
    fuse_fsdp=True,
    use_bias=True,
    activation=None,
    chunks=1,
    num_iters=1,
    enable_trace=False,
    dtype=ttnn.bfloat16,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_acc=True,
):
    """
    FSDP-fused variant. fsdp_axis is also the sp_axis from the model's perspective, so
    activation M is sharded on it. Layout:
      - x       : [M, K]        M sharded on fsdp_axis (= sp), K sharded on tp_axis
      - weight  : [K, N]        K sharded on fsdp_axis,        N sharded on tp_axis
      - bias    : [1, N]        N sharded on tp_axis (replicated on fsdp_axis)
      - output  : [M, N]        M sharded on fsdp_axis,        N sharded on tp_axis
    """
    logger.info(f"Running test_linear fsdp with M={M}, K={K}, N={N}")
    assert tp_axis != fsdp_axis, "tp_axis and fsdp_axis must be distinct"
    tp_size = device.shape[tp_axis]
    fsdp_size = device.shape[fsdp_axis]
    assert tp_size > 1 and fsdp_size > 1, "FSDP fusion test requires both axes > 1"

    torch_dtype = torch.float32
    torch_input = torch.randn((M, K), dtype=torch_dtype)
    weight_input = torch.randn((K, N), dtype=torch_dtype)
    bias_input = torch.randn((1, N), dtype=torch_dtype) if use_bias else None

    with torch.no_grad():
        torch_output = torch_input @ weight_input
        if bias_input is not None:
            torch_output = torch_output + bias_input
        if activation == "gelu":
            torch_output = torch.nn.functional.gelu(torch_output)
        torch_output_chunks = torch.chunk(torch_output, chunks, dim=-1)

    # --- K-sharding ---
    # FUSED: the weight is gathered in lockstep by AGMM's fsdp ring, so in0's tp ring and in1's
    # fsdp ring must consume the same global K-block at each step. We enforce that with a skewed
    # (a+b) K-sharding: device (tp=a, fsdp=b) holds global K-stripe (a+b) for BOTH operands, via a
    # per-block cyclic roll of the K dim so a *uniform* 2D shard of the rolled tensor lands stripe
    # (a+b) on device (a,b). The skew is purely on the contracted K dim, so torch_output (computed
    # above from the unrolled tensors) is unchanged.
    # SEPARATE: the weight is fully gathered (full K) by a standalone all-gather before the matmul,
    # so AGMM indexes the weight by global K-offset (like run_test_linear) and no skew is needed —
    # use the natural, contiguously-sharded tensors.
    if fuse_fsdp:
        assert tp_size == fsdp_size, "skewed sharding requires tp_size == fsdp_size"
        N_ring = tp_size
        assert K % N_ring == 0, "K must be divisible by the ring size for skewed sharding"
        K_per_stripe = K // N_ring

        x_to_load = torch_input.clone()
        M_per_fsdp = M // fsdp_size
        for b in range(fsdp_size):
            rows = slice(b * M_per_fsdp, (b + 1) * M_per_fsdp)
            x_to_load[rows, :] = torch.roll(torch_input[rows, :], shifts=-(b * K_per_stripe), dims=1)

        w_to_load = weight_input.clone()
        N_per_tp = N // tp_size
        for a in range(tp_size):
            cols = slice(a * N_per_tp, (a + 1) * N_per_tp)
            w_to_load[:, cols] = torch.roll(weight_input[:, cols], shifts=-(a * K_per_stripe), dims=0)

        # Self-consistency: after a uniform shard, device (tp=a, fsdp=b) must hold original K-stripe (a+b).
        for a in range(tp_size):
            for b in range(fsdp_size):
                s = (a + b) % N_ring
                x_local = x_to_load[b * M_per_fsdp : (b + 1) * M_per_fsdp, a * K_per_stripe : (a + 1) * K_per_stripe]
                x_ref = torch_input[b * M_per_fsdp : (b + 1) * M_per_fsdp, s * K_per_stripe : (s + 1) * K_per_stripe]
                assert torch.equal(x_local, x_ref), f"x skew mismatch at (a={a}, b={b})"
                w_local = w_to_load[b * K_per_stripe : (b + 1) * K_per_stripe, a * N_per_tp : (a + 1) * N_per_tp]
                w_ref = weight_input[s * K_per_stripe : (s + 1) * K_per_stripe, a * N_per_tp : (a + 1) * N_per_tp]
                assert torch.equal(w_local, w_ref), f"W skew mismatch at (a={a}, b={b})"
    else:
        x_to_load = torch_input
        w_to_load = weight_input

    # x: M (dim 0) sharded on fsdp_axis (= sp_axis), K (dim 1) sharded on tp_axis
    x_shard_dims = [None, None]
    x_shard_dims[fsdp_axis] = 0
    x_shard_dims[tp_axis] = 1
    tt_input = ttnn.from_torch(
        x_to_load,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=x_shard_dims),
    )

    # W: K (dim 0) sharded on fsdp_axis, N (dim 1) sharded on tp_axis
    w_shard_dims = [None, None]
    w_shard_dims[fsdp_axis] = 0
    w_shard_dims[tp_axis] = 1
    tt_weight = ttnn.from_torch(
        w_to_load,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=w_shard_dims),
    )

    # bias: N (dim 1) sharded on tp_axis only
    tt_bias = None
    if use_bias:
        b_shard_dims = [None, None]
        b_shard_dims[tp_axis] = 1
        tt_bias = ttnn.from_torch(
            bias_input,
            dtype=dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=b_shard_dims),
        )

    activation_fn = None
    if activation == "gelu":
        activation_fn = (ttnn.UnaryOpType.GELU, False)
    else:
        assert activation is None, f"Unsupported activation: {activation}"

    ccl_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
    )

    # Semaphores: TP-axis activation gather + FSDP-axis weight gather (both ping-pong pairs)
    tp_sems = [create_global_semaphores(device, device.get_num_devices(), ccl_cores, 0) for _ in range(num_iters)]
    fsdp_sems = [create_global_semaphores(device, device.get_num_devices(), ccl_cores, 0) for _ in range(num_iters)]

    logger.info("Creating persistent buffers")
    # Persistent activation-gather buffer holds the per-device gathered activation
    # [M/fsdp_size, K] (M is sharded on fsdp_axis, K becomes full after the TP all-gather).
    per_device_M = M // fsdp_size
    ag_persistent_buffers = [
        ttnn.from_torch(
            torch.zeros((per_device_M, K), dtype=torch_dtype),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=[None, None]),
        )
        for _ in range(num_iters)
    ]

    # Persistent gathered-weight buffer: [K, N/tp_size] per device — N is TP-sharded.
    per_device_N = N // tp_size
    pwb_shard_dims = [None, None]
    pwb_shard_dims[tp_axis] = 1
    pwb_persistent_buffers = [
        ttnn.from_torch(
            torch.zeros((K, per_device_N), dtype=torch_dtype),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=[None, None]),
        )
        for _ in range(num_iters)
    ]

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

    def run_op(i):
        if fuse_fsdp:
            # Fused: AGMM gathers the weight across fsdp (into the PWB) and the activation across
            # tp internally, then matmuls.
            return ttnn.experimental.all_gather_minimal_matmul_async(
                tt_input,
                tt_weight,
                bias_tensor=tt_bias,
                fused_activation=activation_fn,
                compute_kernel_config=compute_config,
                config=matmul_config,
                persistent_output_buffer=ag_persistent_buffers[i],
                multi_device_global_semaphore=tp_sems[i],
                num_links=num_links,
                topology=topology,
                cluster_axis=tp_axis,
                force_transpose=True,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=48,
                chunks=chunks,
                fsdp_cluster_axis=fsdp_axis,
                fsdp_multi_device_global_semaphore=fsdp_sems[i],
                persistent_weight_buffer=pwb_persistent_buffers[i],
                fsdp_topology=fsdp_topology,
            )

        # Separate: standalone all-gather of the weight across fsdp (K = dim 0) -> [K, N/tp] full-K
        # weight (reusing the [K, N/tp] PWB buffer as the gather output), then plain (non-fsdp) AGMM
        # gathers the activation across tp and matmuls against that full-K weight.
        gathered_weight = ttnn.experimental.all_gather_async(
            tt_weight,
            persistent_output_buffer=pwb_persistent_buffers[i],
            dim=0,
            multi_device_global_semaphore=fsdp_sems[i],
            num_links=num_links,
            topology=fsdp_topology,
            cluster_axis=fsdp_axis,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=2,
        )
        ttnn.synchronize_device(device)
        return ttnn.experimental.all_gather_minimal_matmul_async(
            tt_input,
            gathered_weight,
            bias_tensor=tt_bias,
            fused_activation=activation_fn,
            compute_kernel_config=compute_config,
            config=matmul_config,
            persistent_output_buffer=ag_persistent_buffers[i],
            multi_device_global_semaphore=tp_sems[i],
            num_links=num_links,
            topology=topology,
            cluster_axis=tp_axis,
            force_transpose=True,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=48,
            chunks=chunks,
        )

    tt_output_tensor_list = []
    if enable_trace:
        run_op(0)
        ttnn.synchronize_device(device)
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        tt_out_tensor = run_op(0)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)
        for _ in range(num_iters):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            tt_output_tensor_list.append(tt_out_tensor)
    else:
        for i in range(num_iters):
            ttnn.synchronize_device(device)
            tt_out_tensor = run_op(i)
            tt_output_tensor_list.append(tt_out_tensor)
            logger.info(f"Waiting for op")
            ttnn.synchronize_device(device)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    # Output is [M/fsdp, N/tp] per device — M sharded on fsdp_axis, N sharded on tp_axis.
    # After ConcatMesh2dToTensor the tensor recovers global [M, N/chunks].
    concat_dims = [0, 0]
    concat_dims[fsdp_axis] = 0  # M
    concat_dims[tp_axis] = 1  # N (after chunk split: N/chunks)
    chunk_n = N // chunks

    check_result_list = []
    for n in range(num_iters):
        tt_output = tt_output_tensor_list[n]
        check_result = []
        for c in range(chunks):
            tt_output_chunk = ttnn.from_device(tt_output[c])
            tt_output_chunk = ttnn.to_torch(
                tt_output_chunk,
                mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(device.shape), dims=concat_dims),
            )
            # PCC every (fsdp_i, tp_i) device's slice against the matching slab of torch.
            check_result_chunk = []
            for fsdp_i in range(fsdp_size):
                m_slice = slice(fsdp_i * per_device_M, (fsdp_i + 1) * per_device_M)
                for tp_i in range(tp_size):
                    n_per_dev = chunk_n // tp_size
                    n_slice = slice(tp_i * n_per_dev, (tp_i + 1) * n_per_dev)
                    tt_device_output = tt_output_chunk[m_slice, n_slice]
                    torch_slice = torch_output_chunks[c][m_slice, n_slice]
                    check_result_chunk.append(assert_quality(torch_slice, tt_device_output))
            check_result.append(check_result_chunk)
        check_result_list.append(check_result)

    return check_result_list


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology, fsdp_topology, num_links, num_workers_per_link, tp_axis, fsdp_axis, core_grid_x, core_grid_y",
    [
        [
            (4, 8),
            (4, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Linear,  # in0: forced Linear so it matches the fsdp uni-ring K-block order
            ttnn.Topology.Linear,
            4,
            2,
            0,
            1,
            8,
            8,
        ],
    ],
    ids=["wh_sweep_4x4"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N, use_bias, activation, chunks, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [
        # 4x4 shapes
        (6912, 5120, 15360, True, None, 1, 8, 8, 6, 2, 2),
        (6912, 5120, 5120, True, None, 1, 10, 5, 6, 2, 2),
        (6912, 5120, 13824, True, None, 1, 7, 10, 5, 1, 1),
        (9216, 5120, 15360, True, None, 1, 10, 5, 10, 2, 2),
        (9216, 5120, 5120, True, None, 1, 14, 8, 6, 2, 2),
        (9216, 5120, 13824, True, None, 1, 9, 5, 14, 1, 2),
        (10584, 5120, 15360, True, None, 1, 12, 5, 10, 2, 2),
        (10584, 5120, 5120, True, None, 1, 12, 4, 6, 2, 2),
        (10584, 5120, 13824, True, None, 1, 11, 4, 14, 1, 2),
        (12288, 5120, 15360, True, None, 1, 12, 5, 10, 2, 2),
        (12288, 5120, 5120, True, None, 1, 12, 5, 6, 2, 2),
        (12288, 5120, 13824, True, None, 1, 12, 3, 14, 2, 2),
        (16384, 5120, 15360, True, None, 1, 12, 5, 10, 2, 2),
        (16384, 5120, 5120, True, None, 1, 8, 8, 5, 2, 1),
        (16384, 5120, 13824, True, None, 1, 8, 4, 14, 2, 2),
        (18432, 5120, 15360, True, None, 1, 12, 5, 8, 2, 2),
        (18432, 5120, 5120, True, None, 1, 10, 8, 10, 2, 2),
        (18432, 5120, 13824, True, None, 1, 10, 4, 14, 2, 2),
        (24576, 5120, 15360, True, None, 1, 14, 5, 10, 2, 2),
        (24576, 5120, 5120, True, None, 1, 8, 8, 6, 2, 2),
        (24576, 5120, 13824, True, None, 1, 12, 3, 14, 2, 2),
        (27648, 5120, 15360, True, None, 1, 14, 5, 10, 2, 2),
        (27648, 5120, 5120, True, None, 1, 10, 8, 6, 2, 2),
        (27648, 5120, 13824, True, None, 1, 10, 4, 14, 2, 2),
        (32768, 5120, 15360, True, None, 1, 12, 8, 8, 2, 2),
        (32768, 5120, 5120, True, None, 1, 8, 8, 6, 2, 2),
        (32768, 5120, 13824, True, None, 1, 8, 4, 14, 2, 2),
        (36864, 5120, 15360, True, None, 1, 12, 8, 8, 2, 2),
        (36864, 5120, 5120, True, None, 1, 10, 8, 6, 2, 2),
        (36864, 5120, 13824, True, None, 1, 10, 5, 14, 2, 2),
        (42336, 5120, 15360, True, None, 1, 14, 5, 10, 2, 2),
        (42336, 5120, 5120, True, None, 1, 14, 8, 6, 2, 1),
        (42336, 5120, 13824, True, None, 1, 14, 5, 8, 2, 2),
        (49152, 5120, 15360, True, None, 1, 12, 8, 8, 2, 2),
        (49152, 5120, 5120, True, None, 1, 10, 8, 6, 2, 2),
        (49152, 5120, 13824, True, None, 1, 10, 5, 14, 2, 2),
        (65536, 5120, 15360, True, None, 1, 14, 5, 10, 2, 2),
        (65536, 5120, 5120, True, None, 1, 14, 8, 6, 2, 2),
        (65536, 5120, 13824, True, None, 1, 10, 5, 14, 2, 2),
    ],
    ids=[
        "6912qkv",
        "6912denseout",
        "6912ff1",
        "9216qkv",
        "9216denseout",
        "9216ff1",
        "10584qkv",
        "10584denseout",
        "10584ff1",
        "12288qkv",
        "12288denseout",
        "12288ff1",
        "16384qkv",
        "16384denseout",
        "16384ff1",
        "18432qkv",
        "18432denseout",
        "18432ff1",
        "24576qkv",
        "24576denseout",
        "24576ff1",
        "27648qkv",
        "27648denseout",
        "27648ff1",
        "32768qkv",
        "32768denseout",
        "32768ff1",
        "36864qkv",
        "36864denseout",
        "36864ff1",
        "42336qkv",
        "42336denseout",
        "42336ff1",
        "49152qkv",
        "49152denseout",
        "49152ff1",
        "65536qkv",
        "65536denseout",
        "65536ff1",
    ],
)
@pytest.mark.parametrize(
    "fuse_fsdp",
    [True, False],
    ids=["fused", "separate"],
)
def test_linear_fsdp(
    mesh_device,
    mesh_shape,
    M,
    K,
    N,
    M_block_size,
    K_block_size,
    N_block_size,
    subblock_h,
    subblock_w,
    topology,
    fsdp_topology,
    core_grid_x,
    core_grid_y,
    num_workers_per_link,
    num_links,
    tp_axis,
    fsdp_axis,
    use_bias,
    activation,
    chunks,
    fuse_fsdp,
):
    """
    Exercises all_gather_minimal_matmul_async with the FSDP weight gather fused in.

    Layout:
      - x        : [M, K]         replicated on fsdp_axis, K-sharded on tp_axis
      - weight   : [K, N]         K-sharded on fsdp_axis, N-sharded on tp_axis
      - bias     : [1, N]         N-sharded on tp_axis (replicated on fsdp_axis)
      - output   : [M, N/tp]      replicated on fsdp_axis, N-sharded on tp_axis
    """
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    # Scheme-4 single-row muxes: the fused path now uses the full 8x8 grid (64 worker cores). All 8
    # muxes (4 in0 + 4 fsdp) share the single row below the grid, interleaved by column parity, so the
    # matmul reclaims both the old fsdp column and the second mux row. The separate path is the plain
    # (non-fsdp) AGMM, which already uses the full 8x8 grid (matching test_linear).
    if not fuse_fsdp:
        core_grid_x = 8
        core_grid_y = 8

    check_result = run_test_linear_fsdp(
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
        tp_axis=tp_axis,
        fsdp_axis=fsdp_axis,
        fsdp_topology=fsdp_topology,
        fuse_fsdp=fuse_fsdp,
        use_bias=use_bias,
        activation=activation,
        chunks=chunks,
    )

    fsdp_size = mesh_device.shape[fsdp_axis]
    tp_size = mesh_device.shape[tp_axis]
    for n in range(len(check_result)):
        for c in range(chunks):
            assert len(check_result[n][c]) == fsdp_size * tp_size
            for entry in check_result[n][c]:
                assert entry["pcc"] > 0.999_500
                assert entry["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
    [
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
    ids=["bh4x8links2"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("broadcast_gate", [True, False], ids=["broadcast_gate", "full_gate"])
def test_linear_addcmul_gate(
    mesh_device,
    topology,
    num_links,
    num_workers_per_link,
    sp_axis,
    tp_axis,
    core_grid_x,
    core_grid_y,
    cluster_axis,
    broadcast_gate,
):
    """Test fused addcmul with both broadcast and non-broadcast (full) gate."""
    submesh = _create_cluster_submesh(mesh_device, cluster_axis)
    check_result = run_test_linear(
        submesh,
        M=3072,
        K=5120,
        N=1280,
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=2,
        subblock_w=1,
        topology=topology,
        core_grid=ttnn.CoreCoord(core_grid_x, core_grid_y),
        num_workers_per_link=num_workers_per_link,
        num_links=num_links,
        use_bias=True,
        fuse_addcmul=True,
        addcmul_scalar=1.0,
        broadcast_gate=broadcast_gate,
        use_non_fused=False,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        cluster_axis=cluster_axis,
    )
    for c in range(1):
        for i in range(submesh.get_num_devices()):
            assert check_result[0][c][i]["pcc"] > 0.999_500
            assert check_result[0][c][i]["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
    [
        [
            (4, 8),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(4096),
                "trace_region_size": 90112,
            },
            ttnn.Topology.Linear,
            4,
            2,
            1,
            0,
            8,
            8,
            0,
        ],
    ],
    ids=["wh4x8links4_linear"],
    indirect=["mesh_device", "device_params"],
)
# K=5120 → K_tiles_per_device = 40 (on a 4-device cluster axis).
# K_block values chosen so they do NOT evenly divide 40, exercising the tail-block path:
#   K_block=6  →  7 blocks/device, tail = 40 - 6*6 = 4
#   K_block=9  →  5 blocks/device, tail = 40 - 4*9 = 4
#   K_block=7  →  6 blocks/device, tail = 40 - 5*7 = 5
@pytest.mark.parametrize(
    "M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [
        (3072, 5120, 3840, 8, 6, 8, 2, 2),
        (3072, 5120, 1280, 8, 9, 8, 2, 2),
        (3072, 5120, 3840, 8, 7, 8, 2, 2),
    ],
    ids=["kblk6_tail4", "kblk9_tail4", "kblk7_tail5"],
)
def test_linear_k_tail(
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
    sp_axis,
    tp_axis,
    cluster_axis,
):
    """Linear-only tests where K_block_size does not evenly divide K_tiles_per_device.

    Exercises the tail-block code path: each device's last K-block has fewer real tiles
    than K_block_size, with the remainder zero-padded in L1 to preserve the K_block_size
    row stride. Ring still requires divisibility and would reject these configs.
    """
    submesh = _create_cluster_submesh(mesh_device, cluster_axis)
    check_result = run_test_linear(
        submesh,
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
        force_transpose=True,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        use_bias=True,
        cluster_axis=cluster_axis,
    )
    for c in range(1):
        for i in range(submesh.get_num_devices()):
            assert check_result[0][c][i]["pcc"] > 0.999_500
            assert check_result[0][c][i]["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
    [
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
    ids=["bh4x8links2"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("use_bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize(
    "M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [
        (3072, 5120, 1280, 8, 8, 8, 2, 1),
        (3072, 5120, 3840, 8, 8, 8, 2, 2),
    ],
    ids=["3072x5120x1280", "3072x5120x3840"],
)
def test_linear_swiglu(
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
    sp_axis,
    tp_axis,
    cluster_axis,
    use_bias,
):
    """Ring-fused all_gather_minimal_matmul_async with FUSE_SWIGLU.

    The (replicated) weight is the packed [gate|up] of width 2N, tile-pair interleaved so each
    ring device's N-slice holds whole pairs; the op emits silu(gate)*up of width N in one matmul.
    N here is the OUTPUT width (weight width is 2N).
    """
    submesh = _create_cluster_submesh(mesh_device, cluster_axis)
    check_result = run_test_linear(
        submesh,
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
        force_transpose=True,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        use_bias=use_bias,
        cluster_axis=cluster_axis,
        fuse_swiglu=True,
    )
    for c in range(1):
        for i in range(submesh.get_num_devices()):
            assert check_result[0][c][i]["pcc"] > 0.999_000
            assert check_result[0][c][i]["relative_rmse"] < 0.02


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y, cluster_axis",
    [
        [
            (2, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112},
            ttnn.Topology.Linear,
            1,
            4,
            0,
            1,
            4,
            4,
            1,
        ],
    ],
    ids=["t3k_linear"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "M, K, N, M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [
        (3072, 5120, 1280, 8, 8, 8, 2, 2),
        (3072, 5120, 3840, 8, 8, 8, 2, 2),
    ],
    ids=["3072x5120x2560", "3072x5120x7680"],
)
def test_linear_t3k(
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
    sp_axis,
    tp_axis,
    cluster_axis,
):
    """Linear topology PCC on a 2x4 (T3K) mesh.

    Same uni-ring algorithm as `wh4x8links4_linear` in `test_linear`, but on the smaller
    T3K mesh shape and with `num_links=1` / `num_workers_per_link=4` to exercise the mux
    placement and worker_idx assignment at a different link/worker configuration than the
    one TG uses (links=4, workers_per_link=2). cluster_axis=1 keeps the K-fractured layout
    (4 devices along the AGMM ring axis).
    """
    submesh = _create_cluster_submesh(mesh_device, cluster_axis)
    check_result = run_test_linear(
        submesh,
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
        force_transpose=True,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        use_bias=True,
        cluster_axis=cluster_axis,
    )
    for c in range(1):
        for i in range(submesh.get_num_devices()):
            assert check_result[0][c][i]["pcc"] > 0.999_500
            assert check_result[0][c][i]["relative_rmse"] < 0.02
