# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import random
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from tracy import signpost


def tt_to_torch_dtype(tt_dtype):
    if tt_dtype == ttnn.bfloat16:
        return torch.bfloat16
    elif tt_dtype == ttnn.bfloat8_b:
        return torch.bfloat16
    elif tt_dtype == ttnn.float32:
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {tt_dtype}")


def get_pcc_threshold(dtype):
    if dtype == ttnn.bfloat16:
        return 1.0
    elif dtype == ttnn.bfloat8_b:
        return 0.9999
    elif dtype == ttnn.float32:
        return 1.0
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def gen_tokens(batch, hidden_size, seq_len, mesh_shape, devices, scheme="random", dtype=torch.bfloat16):
    tokens = []
    factor = 1
    for _ in range(batch):
        for _ in range(seq_len):
            if scheme == "random" or scheme == "worst_perf":
                tokens.append(torch.rand(1, 1, 1, hidden_size, dtype=dtype))
            elif scheme == "sequential":
                tokens.append(torch.ones(1, 1, 1, hidden_size, dtype=dtype) * factor)
                factor += 1
            else:
                raise ValueError(f"Invalid scheme: {scheme}")
    res = torch.cat(tokens, dim=0)
    return res.reshape(batch, 1, seq_len, hidden_size)


def gen_expert_mapping(experts, devices, scheme="random"):
    assert experts % devices == 0

    expert_mapping = torch.zeros(1, 1, experts, devices, dtype=torch.int16)
    device_id = 0
    experts_per_devices = experts // devices
    device_expert_count = {d: 0 for d in range(devices)}
    for i in range(experts):
        if scheme == "sequential" or scheme == "worst_perf":
            if i > 0 and i % experts_per_devices == 0:
                device_id += 1
            expert_mapping[0, 0, i, device_id] = 1
        elif scheme == "random":
            device_id = random.choice(
                [d for d, _ in filter(lambda kv: kv[1] < experts_per_devices, device_expert_count.items())]
            )
            expert_mapping[0, 0, i, device_id] = 1
            device_expert_count[device_id] += 1

        else:
            raise ValueError(f"Invalid scheme: {scheme}")

    # identical across all devices
    return expert_mapping


def get_metadata_tensor(expert_indices, expert_mapping, mesh_shape):
    # metadata tensor is expert_indices duplicated for each device
    # [D, B, 1, K]
    batch = expert_indices.shape[0]
    seq_len = expert_indices.shape[2]
    devices = mesh_shape[0] * mesh_shape[1]
    selected_experts_k = expert_indices.shape[3]
    metadata_tensor = torch.reshape(expert_indices, (1, batch, seq_len, selected_experts_k))
    return metadata_tensor.repeat(devices, 1, 1, 1)


def get_expert_indices(batch, experts, selected_experts_k, seq_len, mesh_shape, scheme="random"):
    expert_indices = torch.ones(batch, 1, seq_len, selected_experts_k, dtype=torch.int16) * -1
    current_expert = 0
    for b in range(batch):
        for s in range(seq_len):
            for k in range(selected_experts_k):
                if scheme == "sequential":
                    expert_indices[b, 0, s, k] = current_expert % experts
                    current_expert += 1 + (k % 2)
                elif scheme == "random":
                    # need to ensure a set of unique indices
                    current_indices = expert_indices[b, 0, s, :].tolist()
                    expert_indices[b, 0, s, k] = random.choice(
                        list(filter(lambda e: e not in current_indices, range(experts)))
                    )
                elif scheme == "worst_perf":  # worst perf is when the expert index is always on the last device
                    expert_indices[b, 0, s, k] = (
                        experts - 1
                    )  # technically each expert index should be different, but we're sending to the same device regardless
                else:
                    raise ValueError(f"Invalid scheme: {scheme}")
    return expert_indices


def get_output_tensor(input_tokens, expert_indices, expert_mapping, seq_len, mesh_shape, dtype=torch.bfloat16):
    # output tensor is [devices, batch, seq_len, hidden_size]
    # depending on the expert indices, the input tokens are scattered to different experts
    # these experts are sent to one ore more device based on the expert mapping
    batch = input_tokens.shape[0]
    mesh_columns = mesh_shape[1]
    mesh_rows = mesh_shape[0]
    devices = mesh_columns * mesh_rows
    hidden_size = input_tokens.shape[3]
    selected_experts_k = expert_indices.shape[3]

    # initialize the output tensor with random values so we can test the non-populated rows too
    output_tensor = torch.rand(devices, batch, seq_len, hidden_size, dtype=dtype)

    for b in range(batch):
        for s in range(seq_len):
            for k in range(selected_experts_k):
                expert_id = expert_indices[b, 0, s, k]
                device_assignment = expert_mapping[0, 0, expert_id, :]
                device_indices = torch.where(device_assignment == 1)[0]
                for device_idx in device_indices:
                    output_tensor[device_idx, b, s, :] = input_tokens[b, 0, s, :]

    return output_tensor


def gen_tensors(
    batch, experts, selected_experts_k, hidden_size, seq_len, mesh_shape, devices, scheme="random", dtype=torch.bfloat16
):
    # create input tokens
    assert batch % devices == 0
    assert experts % devices == 0
    assert selected_experts_k < experts

    input_tokens = gen_tokens(batch, hidden_size, seq_len, mesh_shape, devices, scheme, dtype)
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, seq_len, mesh_shape, scheme)

    output_tensor = get_output_tensor(input_tokens, expert_indices, expert_mapping, seq_len, mesh_shape, dtype)
    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping, mesh_shape)

    # create expert indices
    return input_tokens, expert_indices, expert_mapping, output_tensor, metadata_tensor


def log_statistics(
    input_tokens,
    expert_indices,
    expert_mapping,
    output_tensor,
    metadata_tensor,
    cluster_axis,
    mesh_shape,
    topology,
    num_links,
):
    if cluster_axis is None:
        return
    if num_links != 1:
        return
    if topology != ttnn.Topology.Linear:
        return
    if input_tokens.shape[3] != 7168:
        return
    if input_tokens.shape[0] // mesh_shape[cluster_axis] * input_tokens.shape[2] > 1088:
        return
    # based on the cluster_axis, mesh_shape, topology, num_links, log the statistics of the input_tokens, expert_indices, expert_mapping, output_tensor, metadata_tensor
    # first print the number of devices along the cluster_axis
    dispatch_devices = mesh_shape[cluster_axis]
    duplicated_devices = mesh_shape[1 - cluster_axis]
    total_devices = mesh_shape[0] * mesh_shape[1]

    # [duplicated_devices * tokens, devices tensor]
    # count all device to device sends
    #
    total_token_packets_sent = torch.zeros(total_devices, total_devices, dtype=torch.int32)
    total_unique_packets_sent = torch.zeros(total_devices, total_devices, dtype=torch.int32)
    was_token_sent = torch.zeros(input_tokens.shape[0], input_tokens.shape[2], total_devices, dtype=torch.int32)
    for b in range(expert_indices.shape[0]):
        for s in range(expert_indices.shape[2]):
            for k in range(expert_indices.shape[3]):
                expert_id = expert_indices[b, 0, s, k]

                # this is all the devices the expert is assigned to
                token_initial_device = b // (
                    input_tokens.shape[0] // dispatch_devices
                )  # this is the column device the token starts on all rows of duplicated
                for device_assignment in range(total_devices):
                    if expert_mapping[0, 0, expert_id, device_assignment] == 0:
                        continue

                    non_dispatch_device_idx = device_assignment // dispatch_devices

                    starting_device_idx = non_dispatch_device_idx * dispatch_devices + token_initial_device
                    if starting_device_idx != device_assignment:
                        if was_token_sent[b, s, device_assignment] == 0:
                            total_unique_packets_sent[starting_device_idx, device_assignment] += 7
                        was_token_sent[b, s, device_assignment] = 1
                        total_token_packets_sent[starting_device_idx, device_assignment] += 7  # 7 packets per token

    # currently we also send batch extra
    total_metadata_packets_sent = torch.zeros(total_devices, total_devices, dtype=torch.int32)
    total_unique_metadata_packets_sent = torch.zeros(total_devices, total_devices, dtype=torch.int32)
    for i in range(total_devices):
        for j in range(total_devices):
            if i != j and i // dispatch_devices == j // dispatch_devices:
                total_metadata_packets_sent[i, j] = (input_tokens.shape[0] // dispatch_devices) * input_tokens.shape[2]
                total_unique_metadata_packets_sent[i, j] = 1
            else:
                total_metadata_packets_sent[
                    i, j
                ] = 0  # no metadata packets sent to self or to devices that are not on the same row

    logger.info(f"Total token packets received: {torch.sum(total_token_packets_sent, dim=0)}")
    logger.info(f"Total metadata packets received: {torch.sum(total_metadata_packets_sent, dim=0)}")
    logger.info(f"Total unique token packets sent: {torch.sum(total_unique_packets_sent, dim=0)}")
    logger.info(f"Total unique metadata packets sent: {torch.sum(total_unique_metadata_packets_sent, dim=0)}")

    total_packets_sent = torch.sum(total_token_packets_sent, dim=0) + torch.sum(total_metadata_packets_sent, dim=0)
    total_unique_packets_sent = torch.sum(total_unique_packets_sent, dim=0) + torch.sum(
        total_unique_metadata_packets_sent, dim=0
    )
    logger.info(f"All packets sent: {total_packets_sent}")
    logger.info(f"All unique packets sent: {total_unique_packets_sent}")
    logger.info(f"Percentage of unique packets sent: {total_unique_packets_sent / total_packets_sent}")


def compare_results(
    tt_sparse_output_token_tensor,
    tt_metadata_tensor,
    torch_sparse_output_token_tensor,
    torch_metadata_tensor,
    expert_mapping,
    mesh_shape,
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

    batch = tt_metadata_tensor.shape[2]
    devices = tt_metadata_tensor.shape[0]
    selected_experts_k = tt_metadata_tensor.shape[3]
    hidden_size = tt_sparse_output_token_tensor.shape[3]

    for b in range(batch):
        for k in range(selected_experts_k):
            expert_id = tt_metadata_tensor[0, b, 0, k]
            for d in range(devices):
                if expert_mapping[0, 0, expert_id, d] == 1:
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
    seq_len,
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
    cluster_axis=1,
):
    torch.manual_seed(2005)
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
    output_tensors = []
    metadata_tensors = []

    torch_expert_mappings = []

    output_tensor_goldens_list = []
    output_metadata_goldens_list = []
    if cluster_axis is None:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    elif cluster_axis == 1:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_shape)
    elif cluster_axis == 0:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_shape)
    else:
        raise ValueError(f"Invalid cluster_axis: {cluster_axis}")

    for iter in range(num_iters):
        input_tokens, expert_indices, expert_mapping, sparse_output_token_tensor, metadata_tensor = gen_tensors(
            batch,
            experts,
            select_experts_k,
            hidden_size,
            seq_len,
            mesh_shape,
            devices,
            scheme=scheme,
            dtype=tt_to_torch_dtype(dtype),
        )
        if iter == 0:
            log_statistics(
                input_tokens,
                expert_indices,
                expert_mapping,
                sparse_output_token_tensor,
                metadata_tensor,
                cluster_axis,
                mesh_shape,
                topology,
                num_links,
            )
        preallocated_output_tensor = torch.zeros((devices, batch, seq_len, hidden_size), dtype=tt_to_torch_dtype(dtype))
        preallocated_metadata_tensor = torch.zeros((devices, batch, seq_len, select_experts_k), dtype=torch.int32)

        if iter == 0:
            logger.info(f"input_tokens shape: {input_tokens.shape}")
            logger.info(f"expert_indices shape: {expert_indices.shape}")
            logger.info(f"expert_mapping shape: {expert_mapping.shape}")
            logger.info(f"sparse_output_token_tensor shape: {sparse_output_token_tensor.shape}")
            logger.info(f"metadata_tensor shape: {metadata_tensor.shape}")

        output_tensor_goldens_list.append(sparse_output_token_tensor)
        output_metadata_goldens_list.append(metadata_tensor)
        torch_expert_mappings.append(expert_mapping)

        tt_input = ttnn.from_torch(
            input_tokens,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=input_memory_config,
            mesh_mapper=mesh_mapper,
        )

        tt_expert_indices = ttnn.from_torch(
            expert_indices,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=mesh_mapper,
        )

        tt_expert_mapping = ttnn.from_torch(
            expert_mapping,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )

        tt_output_tensor = ttnn.from_torch(
            preallocated_output_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=output_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        tt_metadata_tensor = ttnn.from_torch(
            preallocated_metadata_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=output_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        if iter == 0:
            logger.info(f"tt_input shape: {tt_input.shape}")
            logger.info(f"tt_expert_indices shape: {tt_expert_indices.shape}")
            logger.info(f"tt_expert_mapping shape: {tt_expert_mapping.shape}")

        input_tensors.append(tt_input)
        expert_indices_tensors.append(tt_expert_indices)
        expert_mapping_tensors.append(tt_expert_mapping)
        output_tensors.append(tt_output_tensor)
        metadata_tensors.append(tt_metadata_tensor)

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
            ttnn.all_to_all_dispatch(
                input_tensors[buffer_index],
                expert_indices_tensors[buffer_index],
                expert_mapping_tensors[buffer_index],
                cluster_axis=cluster_axis,
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                global_semaphore=ccl_semaphore_handles[buffer_index],
                subdevice_id=worker_sub_device_id,
                output_tensors=[output_tensors[buffer_index], metadata_tensors[buffer_index]],
            )

            tt_out_tensor = output_tensors[buffer_index]
            tt_metadata = metadata_tensors[buffer_index]

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
            logger.info(f"Capturing Warmup {warmup_iters} iterations")
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            run_op(warmup_iters, store_all_results=False)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)

        logger.info("Capturing Trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_out_tensor_list, tt_metadata_list = run_op(num_iters, store_all_results=False)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        logger.info("Starting Trace perf test...")
        profiler.start("all-to-all-dispatch-trace-warmup")
        if warmup_iters > 0:
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
            ttnn.synchronize_device(mesh_device)
        profiler.end("all-to-all-dispatch-trace-warmup")

        signpost("start")
        profiler.start("all-to-all-dispatch-trace")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)
        profiler.end("all-to-all-dispatch-trace")
        signpost("stop")

        time_taken = profiler.get_duration("all-to-all-dispatch-trace") - profiler.get_duration(
            "all-to-all-dispatch-trace-warmup"
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
    first_failed_batch_index = None
    first_failed_expert_index = None
    first_failed_device_index = None
    first_failed_sequence_index = None

    first_failed_metadata_index = None
    failed_indices = []
    failed_metadata_indices = []
    expected_pcc = get_pcc_threshold(dtype)

    for tensor_index in range(len(tt_out_tensor_list)):
        tt_torch_tensor = ttnn.to_torch(
            tt_out_tensor_list[tensor_index],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )

        tt_metadata_tensor = ttnn.to_torch(
            tt_metadata_list[tensor_index],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )

        batch = tt_torch_tensor.shape[1]
        devices = tt_metadata_tensor.shape[0]
        selected_experts_k = tt_metadata_tensor.shape[3]

        metadata_all_close = torch.allclose(tt_metadata_tensor, output_metadata_goldens_list[tensor_index])
        if not metadata_all_close:
            metadata_passed = False
            first_failed_metadata_index = tensor_index
            failed_metadata_indices = torch.where(tt_metadata_tensor != output_metadata_goldens_list[tensor_index])
            logger.info(f"All failed metadata devices: {failed_metadata_indices}")
            logger.info(f"Failing tt_metadata_tensor tensor {tt_metadata_tensor[failed_metadata_indices]}")
            logger.info(
                f"Relevant output_metadata_goldens_list tensor {output_metadata_goldens_list[tensor_index][failed_metadata_indices]}"
            )
            break

        for b in range(batch):
            for s in range(seq_len):
                for k in range(selected_experts_k):
                    expert_id = tt_metadata_tensor[0, b, s, k]
                    for d in range(devices):
                        if torch_expert_mappings[tensor_index][0, 0, expert_id, d] == 1:
                            eq, output_results = comp_pcc(
                                tt_torch_tensor[d, b, s, :],
                                output_tensor_goldens_list[tensor_index][d, b, s, :],
                                expected_pcc,
                            )
                            is_all_close = torch.allclose(
                                tt_torch_tensor[d, b, s, :], output_tensor_goldens_list[tensor_index][d, b, s, :]
                            )
                            if not is_all_close:
                                logger.info(
                                    f"Output tensor {tensor_index} mismatch at batch {b}, sequence {s}, expert {expert_id}, device {d}"
                                )

                            if not eq or not is_all_close:
                                passed = False
                                first_failed_tensor_index = tensor_index
                                first_failed_batch_index = b
                                failed_indices = torch.where(
                                    tt_torch_tensor[d, b, s, :] != output_tensor_goldens_list[tensor_index][d, b, s, :]
                                )
                                first_10_fail_idx = failed_indices[0][:10]
                                logger.info(f"First 10 failing indices: {first_10_fail_idx}")
                                logger.info(
                                    f"Failing tt_torch_tensor tensor (first 10) {tt_torch_tensor[d, b, s, first_10_fail_idx]}"
                                )
                                logger.info(
                                    f"Relevant output_tensor_goldens_list tensor (first 10) {output_tensor_goldens_list[tensor_index][d, b, s, first_10_fail_idx]}"
                                )
                                first_failed_expert_index = expert_id
                                first_failed_device_index = d
                                first_failed_sequence_index = s
                                break
                if not passed:
                    break
            if not passed:
                break

    logger.info(f"Device has {mesh_device.num_program_cache_entries()} program cache entries")
    assert (
        mesh_device.num_program_cache_entries() == 1
    ), f"Device has {mesh_device.num_program_cache_entries()} program cache entries"

    if not metadata_passed:
        logger.info(f"Failed metadata indices: {failed_metadata_indices}")
        assert metadata_passed, f"{first_failed_metadata_index} FAILED metadata indices: {failed_metadata_indices}"

    if not passed:
        logger.info(f"Failed data indices: {failed_indices}")
        assert (
            passed
        ), f"First failing index: {first_failed_tensor_index} batch {first_failed_batch_index} sequence {first_failed_sequence_index} expert {first_failed_expert_index} device {first_failed_device_index} FAILED data indices: {failed_indices}"


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_2D},
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [
        (2, 5, 1),
    ],
    ids=["s2"],
)
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_all_to_all_dispatch_no_trace(
    mesh_device,
    trace_mode,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="random",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "trace_region_size": 500000,
        },
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 500000,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [
        (128, 2, 1),
        (1, 5, 2),
    ],
    ids=["s128", "s1"],
)
@pytest.mark.parametrize(
    "input_memory_config",
    [
        ttnn.DRAM_MEMORY_CONFIG,
    ],
    ids=["dram"],
)
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_dispatch_trace(
    mesh_device,
    trace_mode,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="random",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [
        (1, 40, 10),
    ],
    ids=["s1"],
)
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_decode_perf(
    mesh_device,
    trace_mode,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="worst_perf",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 500000,
        },
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("batches_per_device", [8])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "seq_len, num_iters, warmup_iters",
    [
        (128, 1, 1),
    ],
    ids=["s128"],
)
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_prefill_perf(
    mesh_device,
    trace_mode,
    mesh_shape,
    cluster_axis,
    batches_per_device,
    experts_per_device,
    select_experts_k,
    hidden_size,
    seq_len,
    num_iters,
    warmup_iters,
    num_links,
    topology,
    dtype,
    input_memory_config,
    output_memory_config,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    run_all_to_all_dispatch_test(
        mesh_device,
        mesh_shape,
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        num_iters,
        warmup_iters,
        trace_mode,
        num_links=num_links,
        scheme="worst_perf",
        topology=topology,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
    )
