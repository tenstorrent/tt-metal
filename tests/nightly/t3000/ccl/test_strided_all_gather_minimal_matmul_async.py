# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import copy
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import is_unsupported_case
from models.common.utility_functions import skip_for_blackhole

from tracy import signpost


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def run_strided_all_gather_minimal_matmul_impl(
    mesh_device,
    num_devices,
    M,
    K,
    N,
    dim,
    other_dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    all_gather_topology,
    mm_block_m,
    mm_block_k,
    mm_block_n,
    subblock_h,
    subblock_w,
    num_iters=1,
    enable_trace=True,
    cluster_axis=1,
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
    shard_weights=False,
    ag_core_grid_offset=(0, 6),
    read_local_slice_from_input=False,
):
    torch.manual_seed(0)

    tile = (32, 32)

    ag_output_shape = [1, 1, M, K]

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
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )

    # create global semaphore handles
    ccl_semaphore_handles = [create_global_semaphores(mesh_device, num_devices, all_cores, 0) for _ in range(num_iters)]

    ##### All gather input setup #####
    logger.info(f"All gather output shape: {ag_output_shape}")
    logger.info(f"All gather dim: {dim}")

    input_tensor_mesh_list = []
    weight_tensor_mesh_list = []
    bias_tensor_mesh_list = []
    ag_output_tensor_goldens_list = []
    torch_matmul_output_list = []

    shard_dims = [other_dim, dim]
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
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
        )
        weight_tensor_mesh = ttnn.from_torch(
            weight_input,
            device=mesh_device,
            layout=layout,
            dtype=ag_input_dtype,
            memory_config=mem_config_input,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, dims=[None, dim if shard_weights else None], mesh_shape=tuple(mesh_device.shape)
            ),
        )
        if use_bias:
            bias_tensor_mesh = ttnn.from_torch(
                bias_input,
                device=mesh_device,
                layout=layout,
                dtype=ag_input_dtype,
                memory_config=mem_config_input,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, dims=[None, dim if shard_weights else None], mesh_shape=tuple(mesh_device.shape)
                ),
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

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    persistent_buffer_shape = copy.deepcopy(ag_output_shape)
    persistent_buffer_shape[other_dim] = persistent_buffer_shape[other_dim] // mesh_device.shape[0]
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(persistent_buffer_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ag_input_dtype,
            memory_config=mem_config_ag,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(num_iters)
    ]
    logger.info("Done creating persistent buffers")

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
                persistent_output_buffer=persistent_output_buffers[i],
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                memory_config=mem_config_ag,
                topology=all_gather_topology,
                cluster_axis=cluster_axis,
                tiles_per_chunk=mm_core_grid.y * (mm_block_m // 32) * (mm_block_k // 32),
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
                mm_cores_y=mm_core_grid.y,
                mm_block_ht=mm_block_m // 32,
                mm_block_wt=mm_block_k // 32,
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
                persistent_output_buffer=persistent_output_buffers[i],
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                strided_all_gather_core_grid_offset=ag_core_grid_offset,
                num_links=num_links,
                memory_config_ag=mem_config_ag,
                topology=all_gather_topology,
                cluster_axis=cluster_axis,
                bias=bias_tensor_mesh_list[i] if use_bias else None,
                fused_activation=activation_fn,
                config=matmul_config,
                memory_config_mm=mem_config_mm,
                compute_kernel_config=compute_config,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
                read_local_slice_from_input=read_local_slice_from_input,
            )
        return tt_all_gather_out_tensor, tt_matmul_out_tensor

    if enable_trace:
        # Compile the op
        run_op(0)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_all_gather_out_tensor, tt_matmul_out_tensor = run_op(0)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Done capturing trace")

        # Execute trace
        signpost("start")
        for i in range(num_iters):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)
            tt_matmul_out_tensor_list.append(tt_matmul_out_tensor)
        logger.info(f"Done executing trace")

        logger.info(f"Waiting for op")
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Done op")

        signpost("stop")
    else:
        for i in range(num_iters):
            ttnn.synchronize_device(mesh_device)
            tt_all_gather_out_tensor, tt_matmul_out_tensor = run_op(i)
            tt_all_gather_out_tensor_list.append(tt_all_gather_out_tensor)
            tt_matmul_out_tensor_list.append(tt_matmul_out_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(mesh_device)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    if not skip_check:
        for i in range(num_iters):
            tt_ag_out_tensor = tt_all_gather_out_tensor_list[i]
            torch_ag_out_tensor = ag_output_tensor_goldens_list[i if not enable_trace else 0]

            concat_dims = [other_dim, 0]
            if not read_local_slice_from_input:
                tt_ag_out = ttnn.from_device(tt_ag_out_tensor)
                tt_ag_out = ttnn.to_torch(
                    tt_ag_out,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims
                    ),
                )

                tt_ag_out_slice = tt_ag_out[0:1, :, :, :]
                eq, output = comp_pcc(tt_ag_out_slice, torch_ag_out_tensor, allowed_pcc)

                logger.info(f"{output}, iteration {i}")
                assert eq, f"iter {i} AG FAILED ag: {output}"

            tt_mm_out_tensor = tt_matmul_out_tensor_list[i]
            torch_mm_out_tensor = torch_matmul_output_list[i if not enable_trace else 0]

            tt_mm_out = ttnn.from_device(tt_mm_out_tensor)
            tt_mm_out = ttnn.to_torch(
                tt_mm_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims if shard_weights else concat_dims
                ),
            )
            if not shard_weights:
                for d in range(mesh_device.shape[1]):
                    tt_mm_out_slice = tt_mm_out[d : d + 1, :, :, :]
                    eq, output = comp_pcc(tt_mm_out_slice, torch_mm_out_tensor)
                logger.info(f"{output}, iteration {i}")
                assert eq, f"iter {i} MM FAILED ag: {output}"
            else:
                eq, output = comp_pcc(tt_mm_out, torch_mm_out_tensor)
                logger.info(f"{output}, iteration {i}")
                assert eq, f"iter {i} MM FAILED ag: {output}"


# tiles_per_chunk needs to be divisible by num_workers_per_link
# mm_cores_y is the number of in0 first col cores
# mm_block_h and mm_block_w is the mm_block of a single mm_core_y
# so the result of one chunk transfer will be mm_cores_y * mm_block_h * mm_block_w, which will be tiles_per_chunk.  tiles_per_chunk % num_workers_per_link must equal 0
@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "M, K, N, dim, other_dim, num_workers_per_link, layout, ag_input_dtype, mm_block_m, mm_block_k, mm_block_n, subblock_h, subblock_w, mm_core_grid, shard_weights",
    [
        # (64, 512, 512, 3, 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 32, 32, 32, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (64, 512, 1024, 3, 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 32, 32, 32, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (64, 512, 2048, 3, 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 32, 32, 32, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (64, 512, 512, 3, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 32, 32, 32, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (128, 512, 512, 3, 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 64, 32, 32, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (128, 512, 512, 3, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 64, 32, 32, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (64, 1024, 512, 3, 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 32, 64, 32, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (64, 1024, 512, 3, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 32, 64, 32, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (64, 512, 1024, 3, 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 32, 32, 64, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (64, 512, 1024, 3, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 32, 32, 64, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (64, 4096, 1024, 3, 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 32, 32, 32, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (4096, 4096, 4096, 3, 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 32, 32, 32, 1, 1, ttnn.CoreCoord(2, 2), False),
        # (4096, 4096, 4096, 3, 2, 1, ttnn.TILE_LAYOUT, ttnn.bfloat16, 32, 32, 32, 1, 1, ttnn.CoreCoord(4, 4), False),
        (4096, 4096, 4096, 3, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 256, 256, 256, 2, 2, ttnn.CoreCoord(4, 4), False),
        # (4096, 4096, 4096, 3, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 256, 160, 256, 1, 1, ttnn.CoreCoord(4, 4), False),
        # (4096, 4096, 4096, 3, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 160, 256, 256, 1, 1, ttnn.CoreCoord(4, 4), False),
    ],
    ids=[
        # "base",  # 1 forward pass through K
        # "forwardbackwardK",  # 1 forward, 1 backward (special because it's not reusing on the first backward)
        # "twiceforwardbackwardK",  # 2 forward, 2 backward (both the non reuse and reuse branches hit)
        # "2workercores",  # test two worker cores on the AG side
        # "mblock21worker",  # make m block size greater than 1
        # "mblock22workers",  # make m block size greater than 1, plus 2 workers
        # "kblock21worker",  # make k block size greater than 1
        # "kblock22workers",  # make m block size greater than 1, plus 2 workers
        # "nblock21worker",  # make n block size greater than 1
        # "nblock22workers",  # make n block size greater than 1, plus 2 workers
        # "morerows",
        # "4k4k4k",
        # "4x4mmcores",  # increase to a larger core grid
        "fulltest",
        # "unalignedK",
        # "unalignedM",
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
        (False, 1),
    ],
    ids=[
        "check",
    ],
)
@pytest.mark.parametrize(
    "use_non_fused",
    [
        False,
    ],
    ids=["fused"],
)
@pytest.mark.parametrize(
    "read_local_slice_from_input",
    [
        True,
    ],
    ids=[
        "read_local",
    ],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_strided_all_gather_minimal_matmul_async(
    mesh_device,
    M,
    K,
    N,
    dim,
    other_dim,
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
    mm_block_m,
    mm_block_k,
    mm_block_n,
    subblock_h,
    subblock_w,
    mm_core_grid,
    use_non_fused,
    shard_weights,
    read_local_slice_from_input,
):
    TILE_SIZE = 32
    assert not ((M // TILE_SIZE) % num_workers_per_link), f"worker must be divisible by num workers per link"
    Nt = N // TILE_SIZE
    if shard_weights:
        Nt_per_device = Nt // mesh_device.get_num_devices()
    else:
        Nt_per_device = Nt
    Nt_per_core = Nt_per_device // mm_core_grid.x
    assert Nt_per_core > (
        mm_block_n // TILE_SIZE
    ), f"block_n size is {mm_block_n // TILE_SIZE} tiles, but only {Nt_per_core} tiles of work per core"

    run_strided_all_gather_minimal_matmul_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        M,
        K,
        N,
        dim,
        other_dim,
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
        mm_block_m=mm_block_m,
        mm_block_k=mm_block_k,
        mm_block_n=mm_block_n,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        mm_core_grid=mm_core_grid,
        use_non_fused=use_non_fused,
        shard_weights=shard_weights,
        read_local_slice_from_input=read_local_slice_from_input,
    )
