# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import ttnn


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
@pytest.mark.parametrize("experts", [256])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("seq", [1])
@pytest.mark.parametrize("hidden_size", [7168])
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
    experts,
    select_experts_k,
    seq,
    hidden_size,
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

    dispatch_devices = mesh_shape[cluster_axis]
    batch = batches_per_device * dispatch_devices

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
    # TODO: (GR)
    tt_expert_mapping = ()
    tt_w0_w1 = ()
    tt_w2 = ()

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

    # TODO: (GR) do we need to double buffer these
    tt_dispatch_preallocated_output_tensors = ()

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

        # deallocate inputs to dispatch that are allocated in L1
        # needed to run multiple iterations since combine uses just about all of L1
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
        run_op(0)
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
