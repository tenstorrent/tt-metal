# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from tests.nightly.tg.ccl.test_all_to_all_dispatch import (
    PACKET_WORKER_CRS,
    gen_tokens,
    gen_expert_mapping,
    get_metadata_tensor,
    get_expert_indices,
    get_output_tensor as get_input_tensor,
)

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tracy import signpost


def get_output_tensor(input_tokens, expert_indices, expert_mapping):
    # output tensor is [devices, batch, k, hidden_size]
    # we'll multiply the tokens by the index of their assigned expert to mock expert application.

    batch = input_tokens.shape[0]
    devices = expert_mapping.shape[3]
    hidden_size = input_tokens.shape[3]
    output_tensor = torch.randn(devices, batch, 1, hidden_size)
    selected_experts_k = expert_indices.shape[3]

    assert batch % devices == 0
    batch_per_device = batch // devices

    output_tensor = input_tokens.repeat([1, 1, selected_experts_k, 1])

    for b in range(batch):
        for k in range(selected_experts_k):
            expert_id = expert_indices[b, 0, 0, k].item()
            output_tensor[:, b, k, :] *= expert_id

    return output_tensor.reshape([devices, batch_per_device, 1, hidden_size])


def gen_tensors(batch, experts, selected_experts_k, hidden_size, devices, scheme="random"):
    torch.manual_seed(2005)
    # create input tokens
    assert batch % devices == 0
    assert experts % devices == 0
    assert selected_experts_k < experts

    input_tokens = gen_tokens(batch, hidden_size, devices, scheme)
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, scheme)

    input_expert_contributions = get_input_tensor(input_tokens, expert_indices, expert_mapping)
    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping)
    output_tensor = get_output_tensor(input_tokens, expert_indices, expert_mapping)

    # create expert indices
    return input_expert_contributions, expert_mapping, metadata_tensor, output_tensor


def compare_results(
    tt_output_contrib_tensor,
    input_token_tensor,
    metadata_tensor,
    expected_pcc=0.99999,
):
    batch = input_token_tensor.shape[1]
    devices = metadata_tensor.shape[0]
    selected_experts_k = metadata_tensor.shape[3]

    for d in range(devices):
        for b in range(batch_per_device):
            b_total = d * batch_per_device + b
            for k in range(selected_experts_k):
                expert_id = metadata_tensor[0, b, 0, k]
                comp_pcc(
                    tt_output_contrib_tensor[d, b, k, :],
                    input_token_tensor[0, b_total, 0, :] * expert_id,
                    expected_pcc,
                )


def run_all_to_all_combine_test(
    mesh_device,
    mesh_shape,
    batch,
    experts,
    select_experts_k,
    hidden_size,
    num_iters,
    warmup_iters,
    num_links=1,  # currently not passed through
    scheme="random",
    use_regular_grid=False,
    input_grid=None,
    output_grid=None,
    dtype=ttnn.bfloat16,
    topology=ttnn.Topology.Linear,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    mesh_device.enable_program_cache()
    devices = mesh_shape[0] * mesh_shape[1]
    # input, output, interm core range set
    compute_grid = (mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y)
    subdevice_shard_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(compute_grid[0] - 1, compute_grid[1] - 1),
            ),
        }
    )

    expert_mapping_tensors = []
    input_tensors = []
    metadata_tensors = []

    output_tensor_goldens_list = []

    for iter in range(num_iters):
        input_contrib, expert_mapping, metadata_tensor, output_contrib_tensor = gen_tensors(
            batch, experts, select_experts_k, hidden_size, devices, scheme=scheme
        )
        if iter == 0:
            logger.info(f"input_tokens shape: {input_tokens.shape}")
            logger.info(f"expert_indices shape: {expert_indices.shape}")
            logger.info(f"expert_mapping shape: {expert_mapping.shape}")
            logger.info(f"sparse_output_token_tensor shape: {sparse_output_token_tensor.shape}")
            logger.info(f"metadata_tensor shape: {metadata_tensor.shape}")

        output_tensor_goldens_list.append(output_contrib_tensor)

        tt_input_contribs = ttnn.from_torch(
            input_contrib,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        tt_expert_mapping = ttnn.from_torch(
            expert_mapping,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        tt_metadata = ttnn.from_torch(
            metadata_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        if iter == 0:
            logger.info(f"tt_input shape: {tt_input.shape}")
            logger.info(f"tt_expert_mapping shape: {tt_expert_mapping.shape}")

        input_tensors.append(tt_input_contribs)
        expert_mapping_tensors.append(tt_expert_mapping)
        metadata_tensors.append(tt_metadata)

    ccl_sub_device_crs = subdevice_shard_cores_grid
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
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_iters)]

    tt_out_tensor_list = []

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []
        tt_metadata_list = []

        for i in range(n_iters):
            buffer_index = 0 if trace_mode else i
            tt_out_tensor = ttnn.all_to_all_combine(
                input_tensors[buffer_index],
                expert_mapping_tensors[buffer_index],
                metadata_tensors[buffer_index],
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                global_semaphore=ccl_semaphore_handles[buffer_index],
                subdevice_id=worker_sub_device_id,
            )

            if not trace_mode:
                ttnn.synchronize_device(mesh_device)
            if store_all_results:
                tt_output_list.append(tt_out_tensor)
                tt_metadata_list.append(tt_metadata)
        if store_all_results:
            return tt_output_list, tt_metadata_list
        else:
            return [tt_out_tensor], [tt_metadata]

    if trace_mode:
        # compile run:
        logger.info("Compiling model")
        tt_out_tensor_list = run_op(1, store_all_results=False)

        logger.info("Capturing Warmup")

        if warmup_iters > 0:
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            run_op(warmup_iters, store_all_results=False)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)

        logger.info("Capturing Trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_outs = run_op(num_iters, store_all_results=False)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        logger.info("Starting Trace perf test...")
        profiler.start("reduce-scatter-trace-warmup")
        if warmup_iters > 0:
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)
        profiler.end("reduce-scatter-trace-warmup")

        signpost("start")
        profiler.start("reduce-scatter-trace")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)
        profiler.end("reduce-scatter-trace")
        signpost("stop")

        time_taken = profiler.get_duration("reduce-scatter-trace") - profiler.get_duration(
            "reduce-scatter-trace-warmup"
        )
        logger.info(f"Time taken e2e: {time_taken} s")
    else:
        signpost("start")
        tt_out_tensor_list = run_op(num_iters, store_all_results=True)
        signpost("stop")

    mesh_device.reset_sub_device_stall_group()

    passed = True
    metadata_passed = True
    first_failed_tensor_index = None
    first_failed_metadata_index = None
    failed_indices = []
    failed_metadata_indices = []
    expected_pcc = 0.9999 if dtype == ttnn.bfloat8_b else 0.999990

    for tensor_index in range(len(tt_out_tensor_list)):
        tt_torch_tensor = ttnn.to_torch(
            tt_out_tensor_list[tensor_index],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )

        logger.info(f"tt_output_tensor shape: {tt_torch_tensor.shape}")
        logger.info(f"golden_output_tensor shape: {output_tensor_goldens_list[tensor_index].shape}")

        compare_results(
            tt_torch_tensor,
        )

    logger.info(f"Device has {mesh_device.num_program_cache_entries()} program cache entries")
    assert (
        mesh_device.num_program_cache_entries() == 1 or mesh_device.num_program_cache_entries() == num_iters
    ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"

    if not metadata_passed:
        logger.info(f"Failed metadata indices: {failed_metadata_indices}")
        assert metadata_passed, f"{first_failed_metadata_index} FAILED metadata indices: {failed_metadata_indices}"

    if not passed:
        logger.info(f"Failed data indices: {failed_indices}")
        assert passed, f"{first_failed_tensor_index} FAILED data indices: {output_results}"


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
def test_all_to_all_dispatch_no_trace(mesh_device, trace_mode, mesh_shape):
    devices = mesh_shape[0] * mesh_shape[1]
    batch = 2 * devices
    experts = 4 * devices
    select_experts_k = 8
    hidden_size = 2000
    num_iters = 1
    warmup_iters = 0
    trace_mode = trace_mode
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    num_links = 1
    topology = ttnn.Topology.Linear

    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="sequential",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
    )


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
def test_simple_tensor_gen(mesh_device, mesh_shape):
    torch.set_printoptions(threshold=10000)
    devices = mesh_shape[0] * mesh_shape[1]
    batch = 2 * devices
    experts = 4 * devices
    select_experts_k = 8
    hidden_size = 2000
    input_tokens, expert_indices, expert_mapping, sparse_output_token_tensor, metadata_tensor = gen_tensors(
        batch, experts, select_experts_k, hidden_size, devices, scheme="sequential"
    )
    assert input_tokens.shape == (batch, 1, 1, hidden_size)
    assert expert_indices.shape == (batch, 1, 1, select_experts_k)
    assert expert_mapping.shape == (devices, 1, experts, devices)
    assert sparse_output_token_tensor.shape == (devices, batch, 1, hidden_size)
    assert metadata_tensor.shape == (devices, batch, 1, select_experts_k)

    logger.info(f"Expert indices {expert_indices}")
    logger.info(f"Expert mapping {expert_mapping[0, :, :, :]}")
    logger.info(f"Metadata tensor {metadata_tensor[0, :, :, :]}")

    compare_results(
        sparse_output_token_tensor, metadata_tensor, sparse_output_token_tensor, metadata_tensor, expert_mapping
    )
