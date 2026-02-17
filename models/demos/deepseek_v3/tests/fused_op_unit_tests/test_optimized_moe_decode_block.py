# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.t3000.ccl.test_all_to_all_dispatch import tt_to_torch_dtype
from tests.nightly.tg.ccl.moe.test_moe_compute_6U.py import (
    create_torch_w0,
    create_torch_w1,
    create_torch_w2,
    prepare_w0_w1_tensor,
    prepare_w2_tensor,
)


def gen_torch_expert_mapping(scheme, devices, experts, experts_per_device, dtype):
    if scheme == "random":
        perm = torch.randperm(experts)
        assignment = torch.empty(experts, dtype=tt_to_torch_dtype(torch.uint16))
        for d in range(devices):
            assignment[perm[d * experts_per_device : (d + 1) * experts_per_device]] = d
    else:
        assignment = torch.arange(experts) // experts_per_device

    return assignment.unsqueeze(0).repeat(devices, 1)


def gen_compute_matmul_cores(mesh_device):
    MATMUL_FULL_CORES = {0, 1, 8, 9}
    MATMUL_PAD_CORES = {2, 3, 4, 5, 6, 7, 10, 11}

    in0_core_coords = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, 0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)

    # Make a new list of core coords that are sorted in decreasing order by y coordinate and then x coordinate.
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        # key: ring_pos, value: (core_coord, dram_bank_id, pad_flag)
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in MATMUL_PAD_CORES else 0)

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    return ring2cores, dram_core_range_set


def gen_torch_compute_matmul_weights(ring2cores, num_layers, experts_per_device, hidden_size, N):
    torch_w0 = create_torch_w0(num_layers, experts_per_device, hidden_size, N)
    torch_w1 = create_torch_w1(num_layers, experts_per_device, hidden_size, N)
    torch_w2 = create_torch_w2(num_layers, experts_per_device, N, hidden_size)

    # Prepare w0_w1 tensor (interleaved, padded, and reordered)
    torch_w0_w1_reordered = prepare_w0_w1_tensor(
        torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, N, ring2cores
    )

    # Prepare w2 tensor (padded and reordered)
    torch_w2_reordered = prepare_w2_tensor(torch_w2, num_layers, experts_per_device, N, hidden_size, ring2cores)

    return torch_w0_w1_reordered, torch_w2_reordered


@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((16, 8), (16, 8), id="16x8_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("cluster_axis", [0])
@pytest.mark.parametrize("num_iterations", [1])
@pytest.mark.parametrize("layer_id", [1])
@pytest.mark.parametrize("batches_per_device", [32])
@pytest.mark.parametrize("shard_dim", [0])
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("matmul_N", [2048])
@pytest.mark.parametrize("scheme", ["random_sequential_experts"])
@pytest.mark.parametrize("combine_worker_core_range", [((5, 0), (6, 7))])
@pytest.mark.parametrize("combine_mux_core_range", [((3, 0), (4, 7))])
@pytest.mark.parametrize("combine_token_parallel_core_dim", [4])
@pytest.mark.parametrize("combine_data_parallel_core_dim", [4])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        },
    ],
    ids=["fabric_1D_ring"],
    indirect=True,
)
def test_optimized_moe_decode_block(
    mesh_shape,
    mesh_device,
    cluster_axis,
    num_iterations,
    layer_id,
    batches_per_device,
    shard_dim,
    experts,
    select_experts_k,
    seq,
    hidden_size,
    matmul_N,
    scheme,
    combine_worker_core_range,
    combine_mux_core_range,
    combine_token_parallel_core_dim,
    combine_data_parallel_core_dim,
    enable_trace,
):
    mesh_device.disable_and_clear_program_cache()

    ############################################
    # initial setup
    ############################################

    devices = mesh_shape[0] * mesh_shape[1]
    dispatch_devices = mesh_shape[cluster_axis]
    batch = batches_per_device * dispatch_devices
    total_tokens = batch * seq
    experts_per_device = experts // devices

    if cluster_axis == 1:
        shard_dims = (None, shard_dim)
    else:
        shard_dims = (shard_dim, None)

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    worker_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    combine_worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in combine_worker_core_range])])
    combine_mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(*[ttnn.CoreCoord(c) for c in combine_mux_core_range])])

    compute_tilize_drain_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(6, 9), ttnn.CoreCoord(6, 9))})

    ############################################
    # create (double buffered) global semaphores
    ############################################
    dispatch_global_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(2)]
    combine_global_semaphores = [ttnn.create_global_semaphore(mesh_device, worker_cores, 0) for _ in range(2)]

    ############################################
    # create constant input tensors
    ############################################
    logger.info(f"Begin creating constant input tensors")

    expert_mapping_dtype = ttnn.uint16
    torch_expert_mapping = gen_torch_expert_mapping(scheme, devices, experts, experts_per_device, expert_mapping_dtype)
    tt_expert_mapping = ttnn.from_torch(
        torch_expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=expert_mapping_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    # ------------------------------------------------------------------------
    # Matmul weights
    # ------------------------------------------------------------------------
    num_layers = layer_id + 1  # test only handles a single layer at a time, simulate all other layers being present
    ring2cores, compute_matmul_dram_core_range_set = gen_compute_matmul_cores(mesh_device)
    torch_w0_w1_reordered, torch_w2_reordered = gen_torch_compute_matmul_weights(
        ring2cores, num_layers, experts_per_device, hidden_size, matmul_N
    )

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w0_w1
    # Tensor shape: (num_layers, experts_per_device, hidden_size, 4608) -> padded and reordered to (12, num_layers, experts_per_device, 6, hidden_size, 64)
    # ------------------------------------------------------------------------
    w0_w1_shard_height = num_layers * experts_per_device * 3 * hidden_size
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE
    w0_w1_shard_spec = ttnn.ShardSpec(
        compute_matmul_dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w0_w1_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec
    )
    w0_w1_dtype = ttnn.bfloat4_b
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=w0_w1_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w2
    # Tensor shape: (num_layers, experts_per_device, N, hidden_size) -> padded and reordered to (12, num_layers, experts_per_device, 5, N + 192, 128)
    # ------------------------------------------------------------------------
    w2_shard_height = num_layers * experts_per_device * 5 * (matmul_N + 192)
    w2_shard_width = 4 * ttnn.TILE_SIZE
    w2_shard_spec = ttnn.ShardSpec(
        compute_matmul_dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    w2_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)
    w2_dtype = ttnn.bfloat4_b
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=w2_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
    )

    logger.info(f"Done creating constant input tensors")

    ############################################
    # create dynamic input tensors & goldens
    ############################################
    logger.info(f"Begin creating dynamic input tensors and goldens")
    tt_dispatch_input_tensors = []
    tt_dispatch_input_expert_indices = []
    tt_dispatch_input_expert_scores = []
    torch_goldens = []
    for iteration in range(num_iterations):
        # TODO: (GR)
        pass

    logger.info(f"Done creating dynamic input tensors and goldens")

    ############################################
    # create persistent dispatch output tensors
    ############################################
    logger.info(f"Begin creating persistent dispatch output tensors")

    dispatch_output_sparse_buffer_dtype = ttnn.bfloat16
    dispatch_output_sparse_buffer_shape = [devices, total_tokens, hidden_size]
    tt_preallocated_dispatch_output_sparse_buffer = ttnn.from_torch(
        torch.zeros(dispatch_output_sparse_buffer_shape, dtype=tt_to_torch_dtype(dispatch_output_sparse_buffer_dtype)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dispatch_output_sparse_buffer_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )
    logger.info(
        f"preallocated_dispatch_output_sparse_bufffer shape: {tt_preallocated_dispatch_output_sparse_buffer.shape}"
    )

    dispatch_output_expert_indices_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(compute_tilize_drain_core, compute_tilize_drain_core)}),
        [total_tokens * devices, select_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dispatch_output_expert_indices_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        dispatch_output_expert_indices_shard_spec,
    )
    dispatch_output_expert_indices_dtype = ttnn.uint16
    disptach_output_expert_indices_shape = [devices, total_tokens, select_experts_k]
    tt_preallocated_dispatch_output_expert_indices = ttnn.from_torch(
        torch.zeros(
            disptach_output_expert_indices_shape, dtype=tt_to_torch_dtype(dispatch_output_expert_indices_dtype)
        ),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dispatch_output_expert_indices_dtype,
        memory_config=dispatch_output_expert_indices_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )
    logger.info(
        f"preallocated_dispatch_output_expert_indices shape: {tt_preallocated_dispatch_output_expert_indices.shape}"
    )

    dispatch_output_expert_scores_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(compute_tilize_drain_core, compute_tilize_drain_core)}),
        [total_tokens * devices, select_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    dispatch_output_expert_scores_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        dispatch_output_expert_scores_shard_spec,
    )
    dispatch_output_expert_scores_dtype = ttnn.bfloat16
    disptach_output_expert_scores_shape = [devices, total_tokens, select_experts_k]
    tt_preallocated_dispatch_output_expert_scores = ttnn.from_torch(
        torch.zeros(disptach_output_expert_scores_shape, dtype=tt_to_torch_dtype(dispatch_output_expert_scores_dtype)),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dispatch_output_expert_scores_dtype,
        memory_config=dispatch_output_expert_scores_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )
    logger.info(
        f"preallocated_dispatch_output_expert_scores shape: {tt_preallocated_dispatch_output_expert_scores.shape}"
    )

    # NOTE: these don't have to be double buffered, as there is a global sync in combine after reading all dispatch output
    tt_dispatch_preallocated_output_tensors = (
        tt_preallocated_dispatch_output_sparse_buffer,
        tt_preallocated_dispatch_output_expert_indices,
        tt_preallocated_dispatch_output_expert_scores,
    )

    logger.info(f"Done creating persistent dispatch output tensors")

    ############################################
    # run op
    ############################################
    logger.info(f"Begin running op iterations")

    def run_op(iteration):
        # create persistent output tensor for combine
        # runtime since it needs to be a zeroed out tensor
        # allacote before dispatch, as dispatch serves as the barrier to ensure the tensor is allocated on all devices
        # TODO: (GR)
        tt_combine_output = ttnn.moreh_full()

        (
            tt_dispatch_output_sparse_buffer,
            tt_dispatch_output_expert_indices,
            tt_dispatch_output_expert_scores,
        ) = ttnn.experimental.all_to_all_dispatch_metadata(
            tt_dispatch_input_tensors[iteration],
            tt_dispatch_input_expert_indices[iteration],
            tt_dispatch_input_expert_scores[iteration],
            tt_expert_mapping,
            cluster_axis=cluster_axis,
            num_links=4,
            drain_sync_tilizer_core=None,
            worker_mode=ttnn.WorkerMode.DIRECT,
            dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
            output_tensors=tt_dispatch_preallocated_output_tensors,
            cross_device_semaphore=dispatch_global_semaphores[iteration % 2],
        )

        # NOTE:
        # - deallocate inputs to dispatch that are allocated in L1
        # - needed to run multiple iterations since combine uses just about all of L1
        ttnn.deallocate(tt_dispatch_input_tensors[iteration])
        ttnn.deallocate(tt_dispatch_input_expert_indices[iteration])
        ttnn.deallocate(tt_dispatch_input_expert_scores[iteration])

        (
            tt_compute_output_token_counts,
            tt_compute_output_dense_expert_activation,
            tt_compute_ouput_dense_e_t,
            tt_compute_output,
        ) = ttnn.experimental.moe_compute(
            tt_dispatch_output_sparse_buffer,
            tt_dispatch_output_expert_indices,
            tt_dispatch_output_expert_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=layer_id,
            cluster_axis=cluster_axis,
        )

        tt_combine_output = ttnn.experimental.selective_reduce_combine(
            tt_compute_output,
            tt_compute_output_dense_expert_activation,
            tt_compute_ouput_dense_e_t,
            tt_compute_output_token_counts,
            hidden_size,
            batch,
            seq,
            select_experts_k,
            experts,
            cluster_axis,
            topology=ttnn.Topology.Ring,
            num_links=4,
            token_parallel_core_dim=combine_token_parallel_core_dim,
            data_parallel_core_dim=combine_data_parallel_core_dim,
            worker_core_range_set=combine_worker_cores,
            mux_core_range_set=combine_mux_cores,
            output_tensor=tt_combine_output,
            optional_cross_device_semaphore=combine_global_semaphores[iteration % 2],
        )

        return tt_combine_output

    tt_outputs = []
    if enable_trace:
        logger.info(f"Begin compiling op")
        # when running multiple iterations, we have to deallocate dispatch input tensors
        # so we need an additional set of input tensors for the compile run
        run_op(num_iterations)
        logger.info(f"Done compiling op")

        logger.info(f"Begin capturing trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for iteration in range(num_iterations):
            tt_output = run_op(iteration)
            tt_outputs.append(tt_output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        logger.info(f"Begin executing trace")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")
    else:
        for iteration in range(num_iterations):
            tt_output = run_op(iteration)
            tt_outputs.append(tt_output)

    logger.info(f"Begin synchronizing devices")
    ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
    logger.info(f"Done synchronizing devices")

    logger.info(f"Done running op iterations")

    ############################################
    # Validate output
    ############################################
    logger.info(f"Begin validating output")
    for iteration in range(num_iterations):
        # TODO: (GR)
        pass

    logger.info(f"Done validating output")
