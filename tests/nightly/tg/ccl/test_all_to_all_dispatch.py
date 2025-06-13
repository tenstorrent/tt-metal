# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from tracy import signpost

PACKET_WORKER_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 1)),
        ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 2)),
    ]
)


def gen_tokens(batch, hidden_size, mesh_shape, devices, scheme="random"):
    per_batch_tokens = []
    factor = 0
    for _ in range(batch):
        if scheme == "random":
            per_batch_tokens.append(torch.rand(1, 1, 1, hidden_size))
        elif scheme == "sequential":
            per_batch_tokens.append(torch.ones(1, 1, 1, hidden_size) * factor)
            factor += 1
        else:
            raise ValueError(f"Invalid scheme: {scheme}")
    res = torch.cat(per_batch_tokens, dim=0)
    res = res.repeat(mesh_shape[0], 1, 1, 1)  # each token is duplicated across devices
    return res


def gen_expert_mapping(experts, devices, scheme="random"):
    expert_mapping = torch.zeros(1, 1, experts, devices, dtype=torch.int16)
    for i in range(experts):
        if scheme == "sequential":
            device_id = i // devices
            expert_mapping[0, 0, i, device_id] = 1
        elif scheme == "random":
            device_id = torch.randint(0, devices, (1,))
            expert_mapping[0, 0, i, device_id] = 1
        else:
            raise ValueError(f"Invalid scheme: {scheme}")

    res = expert_mapping.repeat(devices, 1, 1, 1)
    return res


def get_metadata_tensor(expert_indices, expert_mapping):
    # metadata tensor is expert_indices duplicated for each device
    # [D, B, 1, K]
    batch = expert_indices.shape[0]
    devices = expert_mapping.shape[0]
    selected_experts_k = expert_indices.shape[3]
    metadata_tensor = expert_indices.repeat(devices, 1, 1, 1)
    return torch.reshape(metadata_tensor, (devices, batch, 1, selected_experts_k))


def get_expert_indices(batch, experts, selected_experts_k, mesh_shape, scheme="random"):
    expert_indices = torch.zeros(batch, 1, 1, selected_experts_k, dtype=torch.int16)
    current_expert = 0
    for b in range(batch):
        for k in range(selected_experts_k):
            if scheme == "sequential":
                expert_indices[b, 0, 0, k] = current_expert % experts
                current_expert += 1
            elif scheme == "random":
                expert_indices[b, 0, 0, k] = torch.randint(0, experts, (1,))
            else:
                raise ValueError(f"Invalid scheme: {scheme}")
    expert_indices = expert_indices.repeat(mesh_shape[0], 1, 1, 1)
    return expert_indices


def get_output_tensor(input_tokens, expert_indices, expert_mapping, mesh_shape):
    # output tensor is [devices, batch, 1, hidden_size]
    # depending on the expert indices, the input tokens are scattered to different experts
    # these experts are sent to one ore more device based on the expert mapping
    batch_times_mesh0 = input_tokens.shape[0]
    batch = batch_times_mesh0 // mesh_shape[0]
    devices = expert_mapping.shape[3]
    experts = expert_mapping.shape[2]
    experts_per_mesh0 = experts // mesh_shape[0]
    hidden_size = input_tokens.shape[3]
    output_tensor = torch.randn(devices, batch_times_mesh0, 1, hidden_size)
    selected_experts_k = expert_indices.shape[3]

    for m0 in range(mesh_shape[0]):
        first_expert_in_m0 = m0 * experts_per_mesh0
        last_expert_in_m0 = first_expert_in_m0 + experts_per_mesh0
        for b in range(batch):
            for k in range(selected_experts_k):
                # if k selects an expert in our m0, then we add it to the output tensor
                # if k is not in our m0, then we skip it
                if k < first_expert_in_m0 or k >= last_expert_in_m0:
                    continue
                expert_id = expert_indices[m0 * batch + b, 0, 0, k]
                # Get which devices should handle this expert (shape: devices,)
                device_assignment = expert_mapping[0, 0, expert_id, :]
                # Find device indices that have value 1
                device_indices = torch.where(device_assignment == 1)[0]
                # Assign input tokens to those devices
                for device_idx in device_indices:
                    output_tensor[device_idx, m0 * batch + b, 0, :] = input_tokens[m0 * batch + b, 0, 0, :]

    return output_tensor


def gen_tensors(batch, experts, selected_experts_k, hidden_size, mesh_shape, devices, scheme="random"):
    torch.manual_seed(2005)
    # create input tokens
    assert batch % devices == 0
    assert experts % devices == 0
    assert selected_experts_k < experts

    input_tokens = gen_tokens(batch, hidden_size, mesh_shape, devices, scheme)
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, mesh_shape, scheme)

    output_tensor = get_output_tensor(input_tokens, expert_indices, expert_mapping, mesh_shape)
    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping)

    # create expert indices
    return input_tokens, expert_indices, expert_mapping, output_tensor, metadata_tensor


def compare_results(
    tt_sparse_output_token_tensor,
    tt_metadata_tensor,
    torch_sparse_output_token_tensor,
    torch_metadata_tensor,
    expert_mapping,
    expected_pcc=0.99999,
):
    # compare the output tensor from the tt_sparse_output_token_tensor and the output_tensor_golden
    # the output_tensor_golden is the output tensor from the expert_indices and expert_mapping
    # the tt_sparse_output_token_tensor is the output tensor from the all_to_all_dispatch
    # the tt_metadata_tensor is the metadata tensor from the all_to_all_dispatch
    # compare the output tensor from the tt_sparse_output_token_tensor and the output_tensor_golden
    # compare the metadata tensor from the tt_metadata_tensor and the metadata_tensor
    # since it's sparsely populated into a buffer full of garbage, we should only make sure each input token is present in the correct place in the output tensor
    # and that the metadata tensor is correct

    batch = tt_sparse_output_token_tensor.shape[1]
    devices = tt_metadata_tensor.shape[0]
    selected_experts_k = tt_metadata_tensor.shape[3]

    for b in range(batch):
        for k in range(selected_experts_k):
            expert_id = tt_metadata_tensor[0, b, 0, k]
            for d in range(devices):
                if expert_mapping[d, 0, expert_id, d] == 1:
                    comp_pcc(
                        tt_sparse_output_token_tensor[d, b, 0, :],
                        torch_sparse_output_token_tensor[d, b, 0, :],
                        expected_pcc,
                    )
                    assert torch.allclose(
                        tt_sparse_output_token_tensor[d, b, 0, :], torch_sparse_output_token_tensor[d, b, 0, :]
                    ), f"Output tensor mismatch at batch {b}, expert {expert_id}, device {d}"


def run_all_to_all_dispatch_test(
    mesh_device,
    mesh_shape,
    batch,
    experts,
    select_experts_k,
    hidden_size,
    num_iters,
    warmup_iters,
    trace_mode,
    num_links=3,
    scheme="random",
    use_regular_grid=False,
    input_grid=None,
    output_grid=None,
    dtype=ttnn.bfloat16,
    profiler=BenchmarkProfiler(),
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

    expert_indices_tensors = []
    expert_mapping_tensors = []
    input_tensors = []

    output_tensor_goldens_list = []
    output_metadata_goldens_list = []

    for iter in range(num_iters):
        input_tokens, expert_indices, expert_mapping, sparse_output_token_tensor, metadata_tensor = gen_tensors(
            batch, experts, select_experts_k, hidden_size, devices, scheme=scheme
        )
        if iter == 0:
            logger.info(f"input_tokens shape: {input_tokens.shape}")
            logger.info(f"expert_indices shape: {expert_indices.shape}")
            logger.info(f"expert_mapping shape: {expert_mapping.shape}")
            logger.info(f"sparse_output_token_tensor shape: {sparse_output_token_tensor.shape}")
            logger.info(f"metadata_tensor shape: {metadata_tensor.shape}")

        output_tensor_goldens_list.append(sparse_output_token_tensor)
        output_metadata_goldens_list.append(metadata_tensor)

        tt_input = ttnn.from_torch(
            input_tokens,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        tt_expert_indices = ttnn.from_torch(
            expert_indices,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
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

        if iter == 0:
            logger.info(f"tt_input shape: {tt_input.shape}")
            logger.info(f"tt_expert_indices shape: {tt_expert_indices.shape}")
            logger.info(f"tt_expert_mapping shape: {tt_expert_mapping.shape}")

        input_tensors.append(tt_input)
        expert_indices_tensors.append(tt_expert_indices)
        expert_mapping_tensors.append(tt_expert_mapping)

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
            tt_out_tensor, tt_metadata = ttnn.all_to_all_dispatch(
                input_tensors[buffer_index],
                expert_indices_tensors[buffer_index],
                expert_mapping_tensors[buffer_index],
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
        tt_out_tensor_list, tt_metadata_list = run_op(num_iters, store_all_results=True)
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

        tt_metadata_tensor = ttnn.to_torch(
            tt_metadata_list[tensor_index],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )

        logger.info(f"tt_output_tensor shape: {tt_torch_tensor.shape}")
        logger.info(f"tt_metadata_tensor shape: {tt_metadata_tensor.shape}")

        logger.info(f"golden_output_tensor shape: {output_tensor_goldens_list[tensor_index].shape}")
        logger.info(f"golden_metadata_tensor shape: {output_metadata_goldens_list[tensor_index].shape}")

        batch = tt_torch_tensor.shape[1]
        devices = tt_metadata_tensor.shape[0]
        selected_experts_k = tt_metadata_tensor.shape[3]

        metadata_all_close = torch.allclose(tt_metadata_tensor, output_metadata_goldens_list[tensor_index])
        if not metadata_all_close:
            metadata_passed = False
            first_failed_metadata_index = tensor_index
            failed_metadata_indices = torch.where(tt_metadata_tensor != output_metadata_goldens_list[tensor_index])
            break

        for b in range(batch):
            for k in range(selected_experts_k):
                expert_id = tt_metadata_tensor[0, b, 0, k]
                for d in range(devices):
                    if expert_mapping[d, 0, expert_id, d] == 1:
                        eq, output_results = comp_pcc(
                            tt_torch_tensor[d, b, 0, :],
                            output_tensor_goldens_list[tensor_index][d, b, 0, :],
                            expected_pcc,
                        )
                        logger.info(
                            f"Output tensor {tensor_index} at batch {b}, expert {expert_id}, device {d} has result {output_results}"
                        )
                        is_all_close = torch.allclose(
                            tt_torch_tensor[d, b, 0, :], output_tensor_goldens_list[tensor_index][d, b, 0, :]
                        )
                        if not is_all_close:
                            logger.info(f"Output tensor mismatch at batch {b}, expert {expert_id}, device {d}")

                        if not eq or not is_all_close:
                            passed = False
                            first_failed_tensor_index = tensor_index
                            failed_indices = torch.where(
                                tt_torch_tensor[d, b, 0, :] != output_tensor_goldens_list[tensor_index][d, b, 0, :]
                            )
                            break
                if not passed:
                    break
            if not passed:
                break

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
    experts = 8 * devices
    select_experts_k = 8
    hidden_size = 7000
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
    experts = 8 * devices
    select_experts_k = 8
    hidden_size = 7000
    input_tokens, expert_indices, expert_mapping, sparse_output_token_tensor, metadata_tensor = gen_tensors(
        batch, experts, select_experts_k, hidden_size, mesh_shape, devices, scheme="sequential"
    )
    assert input_tokens.shape == (batch * mesh_shape[0], 1, 1, hidden_size)
    assert expert_indices.shape == (batch * mesh_shape[0], 1, 1, select_experts_k)
    assert expert_mapping.shape == (devices, 1, experts, devices)
    assert sparse_output_token_tensor.shape == (devices, batch * mesh_shape[0], 1, hidden_size)
    assert metadata_tensor.shape == (devices, batch * mesh_shape[0], 1, select_experts_k)

    logger.info(f"Expert indices {expert_indices}")
    logger.info(f"Expert mapping {expert_mapping[0, :, :, :]}")
    logger.info(f"Metadata tensor {metadata_tensor[0, :, :, :]}")

    compare_results(
        sparse_output_token_tensor, metadata_tensor, sparse_output_token_tensor, metadata_tensor, expert_mapping
    )
