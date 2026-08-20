# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import copy
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.nightly.t3000.ccl.test_all_gather import is_unsupported_case
from models.common.utility_functions import skip_for_blackhole
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

from tracy import signpost


def create_global_semaphores(mesh_device, num_devices, cores, initial_value, num_extra=0):
    # 2 out-ready semaphores (one per direction), plus num_extra aggregator per-worker semaphores
    ccl_semaphore_handles = [
        ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2 + num_extra)
    ]
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
    use_ternary=False,
    ternary_scalar=0.5,
    activation=None,
    chunks=1,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_acc=True,
    mm_core_grid=None,
    use_non_fused=True,
    shard_weights=False,
    ag_core_grid_offset=(0, 6),
    read_local_slice_from_input=False,
    mm_signal_aggregator_mode=ttnn.MMSignalAggregatorMode.Auto,
    fuse_swiglu=False,
):
    torch.manual_seed(0)

    tile = (32, 32)

    ag_output_shape = [1, 1, M, K]

    # Skip unsupported cases
    is_known_failure, message = is_unsupported_case(
        ag_output_shape,
        dim,
        mem_config_ag,
        num_devices,
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
    # For the writer-signals-matmul path, the aggregator needs 2 directions x
    num_agg_sems = 2 * num_links * (num_workers_per_link or 0)
    ccl_semaphore_handles = [
        create_global_semaphores(mesh_device, num_devices, all_cores, 0, num_extra=num_agg_sems)
        for _ in range(num_iters)
    ]

    ##### All gather input setup #####
    logger.info(f"All gather output shape: {ag_output_shape}")
    logger.info(f"All gather dim: {dim}")

    input_tensor_mesh_list = []
    weight_tensor_mesh_list = []
    bias_tensor_mesh_list = []
    ternary_a_tensor_mesh_list = []
    ternary_b_tensor_mesh_list = []
    ag_output_tensor_goldens_list = []
    torch_matmul_output_list = []

    shard_dims = [other_dim, dim]
    if use_ternary:
        assert (
            not use_non_fused
        ), "ternary (addcmul) is only wired into the fused strided_all_gather_minimal_matmul path"
    if chunks > 1:
        assert not use_non_fused, "chunks > 1 is only wired into the fused strided_all_gather_minimal_matmul path"
        assert N % chunks == 0, f"N ({N}) must be divisible by chunks ({chunks})"
    # N is the op's OUTPUT width. SwiGLU consumes a packed [up|gate] weight of width 2N and
    # collapses each gate/up tile pair to one output tile, so the device weight is twice as wide.
    weight_N = 2 * N if fuse_swiglu else N
    if fuse_swiglu:
        assert not use_non_fused, "fuse_swiglu is only wired into the fused strided_all_gather_minimal_matmul path"
        assert not use_ternary, "fuse_swiglu is mutually exclusive with ternary (addcmul)"
        assert activation is None, "fuse_swiglu is mutually exclusive with fused_activation"
    # Interleave over the number of devices the weight's N is actually split across (1 when replicated),
    # so each device's slice holds whole [gate, up] tile pairs.
    swiglu_ndev = mesh_device.shape[1] if shard_weights else 1
    for i in range(num_iters):
        torch_dtype = torch.float32
        ag_output_tensor = torch.randn(ag_output_shape, dtype=torch_dtype)
        ag_output_tensor_goldens_list.append(ag_output_tensor)
        weight_input = torch.randn((1, 1, K, weight_N), dtype=torch_dtype)
        if use_bias:
            bias_input = torch.randn((1, weight_N), dtype=torch_dtype)

        # SwiGLU: tile-pair interleave the weight/bias so each device's N-slice holds whole
        # [gate, up] pairs. The golden below still uses the original [up|gate] layout.
        weight_to_load = weight_input
        bias_to_load = bias_input if use_bias else None
        if fuse_swiglu:
            weight_to_load = prepare_for_fused_swiglu(weight_input.reshape(K, weight_N), ndev=swiglu_ndev).reshape(
                weight_input.shape
            )
            if use_bias:
                bias_to_load = prepare_for_fused_swiglu(bias_input.reshape(1, weight_N), ndev=swiglu_ndev).reshape(
                    bias_input.shape
                )
        activation_fn = None
        if activation == "gelu":
            activation_fn = (ttnn.UnaryOpType.GELU, False)
        elif activation == "gelu_tanh":
            activation_fn = ttnn.UnaryOpType.GELU_TANH
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
            weight_to_load,
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
                bias_to_load,
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

        if use_ternary:
            # addcmul: out = ternary_a + scalar * matmul_out * ternary_b
            ternary_a_input = torch.randn((1, 1, M, N), dtype=torch_dtype)
            ternary_b_input = torch.randn((1, 1, 1, N), dtype=torch_dtype)
            ternary_a_tensor_mesh = ttnn.from_torch(
                ternary_a_input,
                device=mesh_device,
                layout=layout,
                dtype=ag_input_dtype,
                memory_config=mem_config_input,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, dims=[other_dim, dim if shard_weights else None], mesh_shape=tuple(mesh_device.shape)
                ),
            )
            ternary_b_tensor_mesh = ttnn.from_torch(
                ternary_b_input,
                device=mesh_device,
                layout=layout,
                dtype=ag_input_dtype,
                memory_config=mem_config_input,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, dims=[None, dim if shard_weights else None], mesh_shape=tuple(mesh_device.shape)
                ),
            )
        else:
            ternary_a_tensor_mesh = None
            ternary_b_tensor_mesh = None
        ternary_a_tensor_mesh_list.append(ternary_a_tensor_mesh)
        ternary_b_tensor_mesh_list.append(ternary_b_tensor_mesh)

        matmul_output = torch.matmul(ag_output_tensor_goldens_list[i], weight_input)
        if use_bias:
            # Row-broadcast bias: bias_input is [1, N] and broadcasts over the M rows, matching add_bias_block.
            matmul_output = matmul_output + bias_input
        if fuse_swiglu:
            # weight_input is the original [up | gate] packing, so the kernel's silu(gate)*up is
            # first * silu(second) here. Applied after bias, matching the kernel's swiglu_block.
            first, second = torch.chunk(matmul_output, 2, dim=-1)
            matmul_output = first * torch.nn.functional.silu(second)
        if use_ternary:
            matmul_output = ternary_a_input + ternary_scalar * matmul_output * ternary_b_input
        # fused_activation is applied last (mutually exclusive with ternary; see the op's validate).
        if activation == "gelu":
            matmul_output = torch.nn.functional.gelu(matmul_output)
        elif activation == "gelu_tanh":
            matmul_output = torch.nn.functional.gelu(matmul_output, approximate="tanh")
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
            # Uniform list-of-chunks interface (single output for the non-fused path).
            tt_matmul_out_tensors = [tt_matmul_out_tensor]
        else:
            fused_outputs = ttnn.experimental.strided_all_gather_minimal_matmul_async(
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
                fused_ternary_input_a=ternary_a_tensor_mesh_list[i] if use_ternary else None,
                fused_ternary_input_b=ternary_b_tensor_mesh_list[i] if use_ternary else None,
                fused_ternary_scalar=ternary_scalar if use_ternary else None,
                chunks=chunks,
                mm_signal_aggregator_mode=mm_signal_aggregator_mode,
                fuse_swiglu=fuse_swiglu,
            )
            # Op returns [all_gather_output, matmul_chunk_0, ..., matmul_chunk_{chunks-1}].
            tt_all_gather_out_tensor = fused_outputs[0]
            tt_matmul_out_tensors = list(fused_outputs[1:])
        return tt_all_gather_out_tensor, tt_matmul_out_tensors

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

            # Matmul output is a list of chunk tensors (one for chunks=1)
            tt_mm_out_tensors = tt_matmul_out_tensor_list[i]
            torch_mm_out_tensor = torch_matmul_output_list[i if not enable_trace else 0]
            torch_mm_chunks = torch.chunk(torch_mm_out_tensor, len(tt_mm_out_tensors), dim=-1)

            for c, tt_mm_chunk_tensor in enumerate(tt_mm_out_tensors):
                tt_mm_out = ttnn.from_device(tt_mm_chunk_tensor)
                tt_mm_out = ttnn.to_torch(
                    tt_mm_out,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device,
                        mesh_shape=tuple(mesh_device.shape),
                        dims=shard_dims if shard_weights else concat_dims,
                    ),
                )
                if not shard_weights:
                    for d in range(mesh_device.shape[1]):
                        tt_mm_out_slice = tt_mm_out[d : d + 1, :, :, :]
                        eq, output = comp_pcc(tt_mm_out_slice, torch_mm_chunks[c])
                    logger.info(f"{output}, iteration {i} chunk {c}")
                    assert eq, f"iter {i} chunk {c} MM FAILED ag: {output}"
                else:
                    eq, output = comp_pcc(tt_mm_out, torch_mm_chunks[c])
                    logger.info(f"{output}, iteration {i} chunk {c}")
                    assert eq, f"iter {i} chunk {c} MM FAILED ag: {output}"


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


# N here is the OUTPUT width; the device weight is 2N wide (packed [up|gate]). The fabric-bound
# factory partitions on gate/up PAIRS, so N/32 must divide the core-grid x and mm_block_n/32 must
# be even for a pair to never straddle a core or an N block.
@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "M, K, N, dim, other_dim, num_workers_per_link, layout, ag_input_dtype, mm_block_m, mm_block_k, mm_block_n, subblock_h, subblock_w, mm_core_grid, shard_weights",
    [
        # replicated weight: 4096-wide packed weight -> 32 weight tiles/core, mm_block_n = 8 tiles
        (4096, 4096, 2048, 3, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 256, 256, 256, 2, 2, ttnn.CoreCoord(4, 4), False),
        # sharded weight: 8192-wide packed weight over 8 devices -> 8 weight tiles/core, so mm_block_n
        # drops to 4 tiles to stay under the per-core work
        (4096, 4096, 4096, 3, 2, 2, ttnn.TILE_LAYOUT, ttnn.bfloat16, 256, 256, 128, 2, 2, ttnn.CoreCoord(4, 4), True),
    ],
    ids=["swiglu_replicated_w", "swiglu_sharded_w"],
)
@pytest.mark.parametrize("use_bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("chunks", [1], ids=["1chunk"])
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
# num_iters > 1 exercises the cached-program override path, not just create().
@pytest.mark.parametrize("enable_trace,num_iters", [(False, 2)], ids=["check"])
@pytest.mark.parametrize("read_local_slice_from_input", [True], ids=["read_local"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_strided_all_gather_minimal_matmul_async_swiglu(
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
    shard_weights,
    use_bias,
    chunks,
    read_local_slice_from_input,
):
    TILE_SIZE = 32
    assert not ((M // TILE_SIZE) % num_workers_per_link), f"worker must be divisible by num workers per link"

    N_block_tiles = mm_block_n // TILE_SIZE
    assert N_block_tiles % 2 == 0, f"fuse_swiglu needs an even mm_block_n in tiles, got {N_block_tiles}"

    # The matmul sees the packed 2N weight; per-core work is measured against that width.
    weight_Nt = (2 * N) // TILE_SIZE
    weight_Nt_per_device = weight_Nt // mesh_device.shape[1] if shard_weights else weight_Nt
    weight_Nt_per_core = weight_Nt_per_device // mm_core_grid.x
    assert (
        weight_Nt_per_core % 2 == 0
    ), f"fuse_swiglu needs an even per-core weight tile count, got {weight_Nt_per_core}"
    assert (
        weight_Nt_per_core > N_block_tiles
    ), f"block_n size is {N_block_tiles} tiles, but only {weight_Nt_per_core} weight tiles of work per core"

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
        use_non_fused=False,
        shard_weights=shard_weights,
        use_bias=use_bias,
        chunks=chunks,
        read_local_slice_from_input=read_local_slice_from_input,
        fuse_swiglu=True,
    )
