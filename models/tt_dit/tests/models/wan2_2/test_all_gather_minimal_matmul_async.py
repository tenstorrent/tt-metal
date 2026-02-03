# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


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

    if use_non_fused:
        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_block_size,
            K_block_size=K_block_size,
            N_block_size=N_block_size,
            subblock_h=subblock_h,
            subblock_w=subblock_w,
            compute_with_storage_grid_size=core_grid,
        )
    else:
        matmul_config = ttnn.AllGatherMinimalMatmulAsyncConfig(
            M_block_size=M_block_size,
            K_block_size=K_block_size,
            N_block_size=N_block_size,
            subblock_h=subblock_h,
            subblock_w=subblock_w,
            compute_with_storage_grid_size=core_grid,
        )

    if use_non_fused:
        tt_all_gather_out_tensor = ttnn.experimental.strided_all_gather_async(
            tt_input,
            persistent_output_buffer=persistent_output_buffers[0],
            dim=3,
            multi_device_global_semaphore=ccl_semaphore_handles[0],
            num_links=num_links,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            topology=topology,
            cluster_axis=cluster_axis,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=2,
            mm_cores_y=core_grid.y,
            mm_block_ht=8,
            mm_block_wt=8,
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
            persistent_output_buffer=persistent_output_buffers[0],
            multi_device_global_semaphore=ccl_semaphore_handles[0],
            num_links=num_links,
            topology=topology,
            cluster_axis=cluster_axis,
            barrier_semaphore=barrier_semaphore_handles[0] if not use_persistent_buffers else None,
            force_transpose=force_transpose,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=2,
        )

    ttnn.synchronize_device(device)

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            device,
            mesh_shape=tuple(device.shape),
            dims=[sp_axis + 2, tp_axis + 2] if use_non_fused else [sp_axis, tp_axis],
        ),
    )
    check_result = []

    for i in range(device.shape[0]):
        for j in range(device.shape[1]):
            if use_non_fused:
                tt_device_output = tt_output[
                    :,
                    :,
                    i * per_device_M : (i + 1) * per_device_M,
                    j * N : (j + 1) * N,
                ]
            else:
                tt_device_output = tt_output[
                    i * per_device_M : (i + 1) * per_device_M,
                    j * N : (j + 1) * N,
                ]
            check_result.append(
                assert_quality(
                    torch_output[:, :, i * per_device_M : (i + 1) * per_device_M, :]
                    if use_non_fused
                    else torch_output[i * per_device_M : (i + 1) * per_device_M, :],
                    tt_device_output,
                )
            )

    return check_result


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
    use_bias=False,
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

    # Prepare TT tensors
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            device,
            mesh_shape=tuple(device.shape),
            dims=[sp_axis + 2, tp_axis + 2] if use_non_fused else [sp_axis, tp_axis],
        ),
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
        cluster_axis=1,
        num_workers_per_link=num_workers_per_link,
        use_non_fused=use_non_fused,
        force_transpose=force_transpose,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        torch_dtype=torch_dtype,
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, num_links, num_workers_per_link, sp_axis, tp_axis, core_grid_x, core_grid_y",
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
        ],
        [
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
            ttnn.Topology.Ring,
            1,
            8,
            0,
            1,
            8,
            8,
        ],
        [
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
            ttnn.Topology.Ring,
            2,
            4,
            0,
            1,
            8,
            8,
        ],
        [
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
            ttnn.Topology.Ring,
            4,
            2,
            0,
            1,
            8,
            8,
        ],
        [
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112},
            ttnn.Topology.Ring,
            2,
            2,
            0,
            1,
            12,
            10,
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
    "M, K, N, force_transpose",
    [
        (32768, 4096, 4096, True),
        (75776, 5120, 3840, True),
        (75776, 5120, 1280, True),
        (75776, 5120, 3456, True),
    ],
    ids=[
        "4k4k4k",
        "qkv",
        "denseout",
        "ff1",
    ],
)
@pytest.mark.parametrize(
    "M_block_size, K_block_size, N_block_size, subblock_h, subblock_w",
    [(8, 8, 8, 2, 2)],
)
@pytest.mark.parametrize(
    "use_non_fused",
    [
        True,
        False,
    ],
    ids=["separate", "fused"],
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
):
    compute_grid_size = mesh_device.compute_with_storage_grid_size()

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
    )
    for i in range(mesh_device.get_num_devices()):
        assert check_result[i]["pcc"] > 0.999_500
        assert check_result[i]["relative_rmse"] < 0.02
