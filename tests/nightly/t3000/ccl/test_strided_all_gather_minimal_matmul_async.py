# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import is_unsupported_case
from models.common.utility_functions import skip_for_blackhole

from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from tracy import signpost


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def run_strided_all_gather_impl(
    mesh_device,
    num_devices,
    M,
    K,
    N,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    all_gather_topology,
    mm_cores_y,
    mm_block_m,
    mm_block_k,
    mm_block_n,
    subblock_h,
    subblock_w,
    num_iters=1,
    enable_trace=True,
    cluster_axis=None,
    tiles_per_chunk=None,
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    allowed_pcc=1,
    skip_check=False,
    num_l1_banks=64,
    use_bias=False,
    activation=None,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_acc=True,
    mm_core_grid=None,
    use_non_fused=True,
):
    torch.manual_seed(0)

    tile = (32, 32)

    ag_output_shape = (1, 1, M, K)

    # Skip unsupported cases
    (is_known_failure, message) = is_unsupported_case(
        ag_output_shape,
        dim,
        mem_config_ag,
        num_devices,
        num_links,
        ag_input_dtype,
        layout,
        tile,
        num_l1_banks,
        mem_config_input,
    )
    if is_known_failure:
        pytest.skip(f"Skipping unsupported case {message}.")

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

    barrier_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    ##### All gather input setup #####
    logger.info(f"All gather output shape: {ag_output_shape}")
    logger.info(f"All gather dim: {dim}")

    input_tensor_mesh_list = []
    weight_tensor_mesh_list = []
    bias_tensor_mesh_list = []
    ag_output_tensor_goldens_list = []
    torch_matmul_output_list = []

    for i in range(num_iters):
        torch_dtype = torch.float32
        ag_output_tensor = torch.randn(ag_output_shape, dtype=torch_dtype)
        ag_output_tensor_goldens_list.append(ag_output_tensor)
        weight_input = torch.randn((1, 1, K, N), dtype=torch_dtype)
        if use_bias:
            bias_input = torch.randn((1, N), dtype=torch_dtype)
        activation_fn = None
        if activation == "gelu":
            activation_fn = (ttnn.UnaryOpType.GELU, False)
        else:
            assert activation is None, f"Unsupported activation: {activation}"

        input_tensor_mesh = ttnn.from_torch(
            ag_output_tensor,
            device=mesh_device,
            layout=layout,
            dtype=ag_input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        )
        weight_tensor_mesh = ttnn.from_torch(
            weight_input,
            device=mesh_device,
            layout=layout,
            dtype=ag_input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        )
        if use_bias:
            bias_tensor_mesh = ttnn.from_torch(
                bias_input,
                device=mesh_device,
                layout=layout,
                dtype=ag_input_dtype,
                memory_config=mem_config_input,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
            )
        else:
            bias_tensor_mesh = None

        input_tensor_mesh_list.append(input_tensor_mesh)
        weight_tensor_mesh_list.append(weight_tensor_mesh)
        bias_tensor_mesh_list.append(bias_tensor_mesh)

        if use_bias:
            matmul_output = torch.nn.functional.linear(
                ag_output_tensor_goldens_list[i], weight_input.T.contiguous(), bias_input
            )
        else:
            matmul_output = torch.matmul(ag_output_tensor_goldens_list[i], weight_input)
        torch_matmul_output_list.append(matmul_output)

    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=mm_block_m // 32,
        K_block_size=mm_block_k // 32,
        N_block_size=mm_block_n // 32,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=mm_core_grid,
    )

    ##### Perform the TT ops #####
    tt_all_gather_out_tensor_list = []
    tt_matmul_out_tensor_list = []

    def run_op(i):
        if use_non_fused:
            tt_all_gather_out_tensor = ttnn.experimental.strided_all_gather_async(
                input_tensor_mesh_list[i],
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                memory_config=mem_config_ag,
                topology=all_gather_topology,
                subdevice_id=worker_sub_device_id,
                cluster_axis=cluster_axis,
                barrier_semaphore=barrier_semaphore_handles[i],
                tiles_per_chunk=tiles_per_chunk,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
                mm_cores_y=mm_cores_y,
                mm_block_h=mm_block_m // 32,
                mm_block_w=mm_block_k // 32,
            )

            tt_matmul_out_tensor = ttnn.experimental.minimal_matmul(
                tt_all_gather_out_tensor,
                weight_tensor_mesh_list[i],
                bias_tensor=bias_tensor_mesh_list[i] if use_bias else None,
                fused_activation=activation_fn,
                compute_kernel_config=compute_config,
                config=matmul_config,
            )
        else:
            tt_all_gather_out_tensor, tt_matmul_out_tensor = ttnn.experimental.strided_all_gather_minimal_matmul_async(
                input_tensor_mesh_list[i],
                weight_tensor_mesh_list[i],
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                strided_all_gather_core_grid_offset=(0, 6),
                num_links=num_links,
                memory_config_ag=mem_config_ag,
                topology=all_gather_topology,
                barrier_semaphore=barrier_semaphore_handles[i],
                subdevice_id=worker_sub_device_id,
                bias=bias_tensor_mesh_list[i] if use_bias else None,
                fused_activation=activation_fn,
                config=matmul_config,
                memory_config_mm=mem_config_mm,
                compute_kernel_config=compute_config,
                tiles_per_chunk=tiles_per_chunk,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
            )
        return tt_all_gather_out_tensor, tt_matmul_out_tensor

    if enable_trace:
        # Compile the op
        tt_all_gather_out_tensor = run_op(0)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_all_gather_out_tensor = run_op(0)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
        logger.info(f"Done capturing trace")

        # Execute trace
        signpost("start")
        for i in range(num_iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)
        logger.info(f"Done executing trace")
        signpost("stop")
    else:
        for i in range(num_iters):
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            tt_all_gather_out_tensor, tt_matmul_out_tensor = run_op(i)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)
            tt_matmul_out_tensor_list.append(tt_matmul_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    if not skip_check:
        for i in range(num_iters):
            tt_ag_out_tensor = tt_all_gather_out_tensor_list[i]
            torch_ag_out_tensor = ag_output_tensor_goldens_list[i if not enable_trace else 0]
            expected_tensor = torch_ag_out_tensor

            tt_ag_out = ttnn.from_device(tt_ag_out_tensor)
            tt_ag_out = ttnn.to_torch(tt_ag_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

            tt_ag_out = tt_ag_out[:, :, :, 0 : expected_tensor.shape[3]]
            eq, output = comp_pcc(tt_ag_out, expected_tensor, allowed_pcc)

            logger.info(f"{output}, iteration {i}")
            assert eq, f"{i} AG FAILED ag: {output}"

            tt_mm_out_tensor = tt_matmul_out_tensor_list[i]
            torch_mm_out_tensor = torch_matmul_output_list[i if not enable_trace else 0]

            tt_mm_out = ttnn.from_device(tt_mm_out_tensor)
            tt_mm_out = ttnn.to_torch(tt_mm_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

            breakpoint()

            eq, output = comp_pcc(tt_mm_out, torch_mm_out_tensor)

            logger.info(f"{output}, iteration {i}")
            assert eq, f"{i} MM FAILED ag: {output}"

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


# tiles_per_chunk needs to be divisible by num_workers_per_link
# mm_cores_y is the number of in0 first col cores
# mm_block_h and mm_block_w is the mm_block of a single mm_core_y
# so the result of one chunk transfer will be mm_cores_y * mm_block_h * mm_block_w, which will be tiles_per_chunk.  tiles_per_chunk % num_workers_per_link must equal 0
@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "M, K, N, dim, num_workers_per_link, tiles_per_chunk, layout, ag_input_dtype, mm_cores_y, mm_block_m, mm_block_k, mm_block_n, subblock_h, subblock_w, mm_core_grid",
    [
        (64, 512, 512, 3, 1, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 1, 32, 32, 32, 1, 1, ttnn.CoreCoord(2, 2)),
    ],
    ids=[
        "1tile1chunk1worker1row",
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
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
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
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_strided_all_gather_async(
    mesh_device,
    M,
    K,
    N,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    enable_trace,
    all_gather_topology,
    num_iters,
    num_workers_per_link,
    tiles_per_chunk,
    mm_cores_y,
    mm_block_m,
    mm_block_k,
    mm_block_n,
    subblock_h,
    subblock_w,
    mm_core_grid,
    use_non_fused,
):
    run_strided_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        M,
        K,
        N,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        mem_config_mm,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
        num_workers_per_link=num_workers_per_link,
        tiles_per_chunk=tiles_per_chunk,
        mm_cores_y=mm_cores_y,
        mm_block_m=mm_block_m,
        mm_block_k=mm_block_k,
        mm_block_n=mm_block_n,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        mm_core_grid=mm_core_grid,
        use_non_fused=use_non_fused,
    )
