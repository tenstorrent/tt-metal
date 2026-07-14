# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.nightly.t3000.ccl.test_all_gather import is_unsupported_case
from tests.tests_common.skip_reasons import LEGACY_CCL_SKIP

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
    use_barrier=False,
    use_persistent_buffers=True,
    chunks_per_sync=None,
    num_workers_per_link=None,
    num_buffers_per_channel=None,
    packer_l1_acc=True,
    precision_offset=None,
    matmul_1d_mcast_in0=None,
):
    if use_legacy_allgather:
        pytest.skip(LEGACY_CCL_SKIP)
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

        barrier_semaphore_handles = [
            ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)
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
        if precision_offset is not None:
            # A small random signal plus a large constant offset. Paired with the column-sum-zero
            # weights below, the offset contributes nothing to the K reduction of in0 @ weights, so
            # the true result is small while the running partial sums are large (~offset). Any loss of
            # precision in the cross-block reload of the fp32 partials destroys the small true result.
            ag_output_tensor = (torch.randn(ag_output_shape) + precision_offset).bfloat16()
        else:
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
        weights_tensor = torch.randn([hidden_dim, matmul_output_dim * num_devices])
        if precision_offset is not None:
            # Each column sums to zero over the K (hidden) dimension so the constant offset in the
            # activations cancels in in0 @ weights (see the input construction above).
            weights_tensor = weights_tensor - weights_tensor.mean(dim=0, keepdim=True)
        weights_tensor = weights_tensor.bfloat16()
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
        bias_tensor = torch.randn([1, matmul_output_dim * num_devices]).bfloat16()
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
    if matmul_1d_mcast_in0 is not None:
        # Single-core 1D multicast config: the whole per-device matmul runs on one core with K split
        # into many blocks, so the cross-block reload runs on the classic mcast_in0/in1 program path
        # (mcast_in0 selects which). Shapes are kept small so the single core's CBs fit in SRAM.
        n_tiles = max(1, matmul_output_dim // 32)
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
            in0_block_w=min(max_in0_block_w, hidden_dim // 32),  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=min(2, n_tiles),  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max(1, math.ceil(ag_output_shape[2] / 32)),  # full M on the single core
            per_core_N=n_tiles,  # full per-device N on the single core
            fuse_batch=True,
            mcast_in0=matmul_1d_mcast_in0,
        )
    else:
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
    # For the precision check, use full-mantissa HiFi4 with no approximation so the only source of
    # error under test is the fp32 cross-block reload, not the multiply fidelity.
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4 if precision_offset is not None else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False if precision_offset is not None else True,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )

    ##### Perform torch ops #####
    torch_matmul_output_list = []
    for i in range(num_iters):
        if use_bias and precision_offset is not None:
            # Reference the true (small) result in fp64 so the assertion detects any precision loss in
            # the device's fp32 accumulation of the large partial sums.
            matmul_output = torch.nn.functional.linear(
                ag_output_tensor_goldens_list[i].double(),
                weights_tensor.double().T.contiguous(),
                bias_tensor.double(),
            ).float()
        elif use_bias:
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
            tt_all_gather_out_tensor = ttnn.experimental.all_gather_async(
                input_tensor_mesh_list[i],
                persistent_output_buffer=persistent_output_buffers[i] if use_persistent_buffers else None,
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                memory_config=mem_config_ag,
                topology=all_gather_topology,
                subdevice_id=worker_sub_device_id,
                barrier_semaphore=barrier_semaphore_handles[i] if use_barrier else None,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
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
            tt_all_gather_out_tensor, tt_matmul_out_tensor = ttnn.experimental.all_gather_matmul_async(
                input_tensor_mesh_list[i],
                weight_tt,
                persistent_output_buffer=persistent_output_buffers[i] if use_persistent_buffers else None,
                dim=dim,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                all_gather_core_grid_offset=(0, 6),
                barrier_semaphore=barrier_semaphore_handles[i] if use_barrier else None,
                bias=bias_tt,
                num_links=num_links,
                memory_config_ag=mem_config_ag,
                topology=all_gather_topology,
                subdevice_id=worker_sub_device_id,
                memory_config_mm=mem_config_mm,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                chunks_per_sync=chunks_per_sync,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
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
        # The offset-cancellation construction makes correct fp32 accumulation land near the fp64
        # reference (PCC ~1) while a TF32-truncated reload collapses PCC, so a tight threshold cleanly
        # separates the two.
        if precision_offset is not None:
            eq, output = comp_pcc(tt_mm_out, torch_mm_out_tensor, 0.99)
        else:
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


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "num_links, ag_output_shape, dim, layout, matmul_output_dim, max_in0_block_w, matmul_weights_dtype, ag_input_dtype, use_bias, enable_trace, num_iters, use_barrier, use_persistent_buffers, chunks_per_sync, num_workers_per_link, num_buffers_per_channel",
    [
        # Shape 0 tests - fused only
        (
            1,
            [1, 1, 4096, 2560],
            3,
            ttnn.TILE_LAYOUT,
            960,
            2,
            ttnn.bfloat16,
            ttnn.bfloat16,
            True,
            True,
            10,
            False,
            True,
            10,
            2,
            2,
        ),  # perf, no_barrier_with_persistent, chunking
        (
            1,
            [1, 1, 4096, 2560],
            3,
            ttnn.TILE_LAYOUT,
            960,
            2,
            ttnn.bfloat16,
            ttnn.bfloat16,
            True,
            False,
            1,
            True,
            True,
            None,
            None,
            None,
        ),  # check, barrier_with_persistent, default
        (
            1,
            [1, 1, 4096, 2560],
            3,
            ttnn.TILE_LAYOUT,
            960,
            2,
            ttnn.bfloat16,
            ttnn.bfloat16,
            True,
            True,
            10,
            True,
            False,
            None,
            None,
            None,
        ),  # perf, barrier_without_persistent, default
        # Shape 1 tests - fused only
        (
            1,
            [1, 1, 32, 512],
            3,
            ttnn.TILE_LAYOUT,
            960,
            2,
            ttnn.bfloat16,
            ttnn.bfloat16,
            True,
            False,
            1,
            True,
            True,
            None,
            None,
            None,
        ),  # check, barrier_with_persistent, default
        (
            1,
            [1, 1, 32, 512],
            3,
            ttnn.TILE_LAYOUT,
            960,
            2,
            ttnn.bfloat16,
            ttnn.bfloat16,
            True,
            True,
            10,
            False,
            True,
            10,
            2,
            2,
        ),  # perf, no_barrier_with_persistent, chunking
        (
            1,
            [1, 1, 32, 512],
            3,
            ttnn.TILE_LAYOUT,
            960,
            2,
            ttnn.bfloat16,
            ttnn.bfloat16,
            True,
            False,
            1,
            True,
            False,
            10,
            2,
            2,
        ),  # check, barrier_without_persistent, chunking
    ],
    ids=[
        "ag_output_shape0-perf-no_barrier_with_persistent-chunking",
        "ag_output_shape0-check-barrier_with_persistent-default",
        "ag_output_shape0-perf-barrier_without_persistent-default",
        "ag_output_shape1-check-barrier_with_persistent-default",
        "ag_output_shape1-perf-no_barrier_with_persistent-chunking",
        "ag_output_shape1-check-barrier_without_persistent-chunking",
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
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_all_gather_matmul_async(
    mesh_device,
    num_links,
    ag_output_shape,
    dim,
    layout,
    matmul_output_dim,
    max_in0_block_w,
    matmul_weights_dtype,
    ag_input_dtype,
    use_bias,
    enable_trace,
    num_iters,
    use_barrier,
    use_persistent_buffers,
    chunks_per_sync,
    num_workers_per_link,
    num_buffers_per_channel,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    all_gather_topology,
):
    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
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
        use_non_fused=False,
        use_legacy_allgather=False,
        num_iters=num_iters,
        use_barrier=use_barrier,
        use_persistent_buffers=use_persistent_buffers,
        chunks_per_sync=chunks_per_sync,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )


@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_all_gather_matmul_async_fp32_reload_precision(mesh_device, all_gather_topology):
    """fp32-accumulation precision of the fused all-gather + matmul on the 2D multicast program path.

    all_gather_matmul routes a 2D program config to the classic mcast_in0_in1 program factory (a path
    normal ttnn.matmul does not use), so this is the only coverage of fp32 accumulation there. The
    activations carry a large constant offset and each weight column sums to zero over the contraction
    dimension, so the offset cancels in in0 @ weights and the true result is small while the running
    K-reduction partials are large. With packer_l1_acc disabled the fp32 partials are reloaded into
    DEST between K-blocks; unless that reload stays in fp32 the small true result is rounded away and
    PCC against the fp64 reference collapses. The fused bias makes the partials CB a second consumer
    read as an FPU operand (SrcA), which is the case that requires the reload to go through a separate
    UnpackToDestFp32 alias of the partials CB. A small in0_block_w splits K into many blocks so the
    reload runs on every block boundary.
    """
    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        ag_output_shape=[1, 1, 4096, 2560],
        dim=3,
        num_links=1,
        ag_input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        matmul_output_dim=960,
        matmul_weights_dtype=ttnn.bfloat16,
        max_in0_block_w=1,  # 1 tile per K-block -> ~80 blocks over K=2560, so the reload runs often
        use_bias=True,
        mem_config_input=dram,
        mem_config_ag=dram,
        mem_config_mm=dram,
        all_gather_topology=all_gather_topology,
        use_non_fused=False,
        use_legacy_allgather=False,
        num_iters=1,
        enable_trace=False,
        use_barrier=False,
        use_persistent_buffers=False,
        packer_l1_acc=False,
        precision_offset=1000.0,
    )


@pytest.mark.parametrize("matmul_1d_mcast_in0", [True, False])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
    ],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_all_gather_matmul_async_1d_fp32_reload_precision(mesh_device, all_gather_topology, matmul_1d_mcast_in0):
    """fp32-accumulation precision of the fused all-gather + matmul on the 1D multicast program path.

    A non-gather 1D program config routes all_gather_matmul to the classic mcast program factory:
    process_mcast_in0 when mcast_in0=True, process_mcast_in1 when mcast_in0=False (a path normal
    ttnn.matmul does not use, since non-gather 1D goes through the descriptor path). Same precision
    construction as the 2D case: a large offset in the activations cancels against column-sum-zero
    weights, so the true result is small while the K-reduction partials are large; with packer_l1_acc
    disabled the fp32 partials reload into DEST between blocks, and unless that reload stays fp32 the
    small result is rounded away and PCC against the fp64 reference collapses. Fused bias makes the
    partials CB a second SrcA consumer, exercising the UnpackToDestFp32 alias of the partials CB.
    """
    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_all_gather_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        ag_output_shape=[1, 1, 64, 4096],
        dim=3,
        num_links=1,
        ag_input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        matmul_output_dim=64,
        matmul_weights_dtype=ttnn.bfloat16,
        max_in0_block_w=1,  # 1 tile per K-block -> 128 blocks over K=4096, so the reload runs often
        use_bias=True,
        mem_config_input=dram,
        mem_config_ag=dram,
        mem_config_mm=dram,
        all_gather_topology=all_gather_topology,
        use_non_fused=False,
        use_legacy_allgather=False,
        num_iters=1,
        enable_trace=False,
        use_barrier=False,
        use_persistent_buffers=False,
        packer_l1_acc=False,
        precision_offset=1000.0,
        matmul_1d_mcast_in0=matmul_1d_mcast_in0,
    )
