# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import random
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.utils_for_testing import is_unsigned_tensor

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from tracy import signpost


def get_max_links(cluster_axis, fabric_config):
    if fabric_config == ttnn.FabricConfig.FABRIC_2D:
        return 1
    elif cluster_axis is None:
        return 1
    else:
        return 2 if cluster_axis == 0 else 1


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
            if scheme == "sequential":
                tokens.append(torch.ones(1, 1, 1, hidden_size, dtype=dtype) * factor)
                factor += 1
            else:
                tokens.append(torch.rand(1, 1, 1, hidden_size, dtype=dtype))
    res = torch.cat(tokens, dim=0)
    return res.reshape(batch, 1, seq_len, hidden_size)


def gen_expert_mapping(experts, devices, scheme="random"):
    assert experts % devices == 0

    expert_mapping = torch.zeros(1, 1, experts, devices, dtype=torch.int16)
    device_id = 0
    experts_per_devices = experts // devices
    device_expert_count = {d: 0 for d in range(devices)}
    for i in range(experts):
        if scheme == "random":
            device_id = random.choice(
                [d for d, _ in filter(lambda kv: kv[1] < experts_per_devices, device_expert_count.items())]
            )
            expert_mapping[0, 0, i, device_id] = 1
            device_expert_count[device_id] += 1
        else:
            if i > 0 and i % experts_per_devices == 0:
                device_id += 1
            expert_mapping[0, 0, i, device_id] = 1

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

    # For avg_perf scheme, track how many tokens are assigned to each expert
    if scheme == "avg_perf":
        tokens = batch * seq_len
        # Use ceiling division to ensure we have enough capacity, with minimum of 1
        max_tokens_per_expert = max(1, (tokens * selected_experts_k + experts - 1) // experts)
        expert_token_count = {e: 0 for e in range(experts)}

    token = 0
    for b in range(batch):
        for s in range(seq_len):
            token += 1
            for k in range(selected_experts_k):
                if scheme == "sequential":
                    expert_indices[b, 0, s, k] = current_expert % experts
                    current_expert += 1 + (k % 2)
                elif scheme == "random" or scheme == "random_sequential_experts":
                    # need to ensure a set of unique indices
                    current_indices = expert_indices[b, 0, s, :].tolist()
                    expert_indices[b, 0, s, k] = random.choice(
                        list(filter(lambda e: e not in current_indices, range(experts)))
                    )
                elif scheme == "avg_perf":
                    # Random selection but each expert is capped at max_tokens_per_expert
                    # to simulate average performance case
                    current_indices = expert_indices[b, 0, s, :].tolist()
                    # First, try to find experts under the cap
                    available_experts = [
                        e
                        for e in range(experts)
                        if e not in current_indices and expert_token_count[e] < max_tokens_per_expert
                    ]
                    # Fallback: if all capped experts are used, pick any expert not in current token
                    if not available_experts:
                        available_experts = [e for e in range(experts) if e not in current_indices]
                    chosen_expert = random.choice(available_experts)
                    expert_indices[b, 0, s, k] = chosen_expert
                    expert_token_count[chosen_expert] += 1
                elif scheme == "worst_perf":  # worst perf is when the expert index is always on the last device
                    expert_indices[b, 0, s, k] = (
                        experts - 1
                    )  # technically each expert index should be different, but we're sending to the same device regardless
                elif scheme == "worst_congestion":
                    # Worst case link congestion: each token selects experts on consecutive
                    # devices in one direction, maximizing traffic through intermediate links.
                    # From device D, select experts on devices D+1, D+2, ..., D+k
                    # This creates a cascade where link i→i+1 carries traffic from all
                    # devices 0..i to all devices i+1..n, maximizing link utilization.
                    devices = mesh_shape[0] * mesh_shape[1]
                    experts_per_device = experts // devices

                    # Determine source device for this token (batch is sharded in chunks across devices)
                    # Device 0 gets batch 0 to batch/devices-1, device 1 gets next chunk, etc.
                    batches_per_device = batch // devices
                    src_device = b // batches_per_device

                    # Target device is src_device + k + 1 (wrapping around)
                    target_device = (src_device + 1 + k) % devices

                    # Expert offset: if we wrap around and visit a device multiple times,
                    # pick a different expert on that device each time
                    expert_offset = k // devices
                    expert_id = target_device * experts_per_device + (expert_offset % experts_per_device)

                    expert_indices[b, 0, s, k] = expert_id
                elif scheme == "best_congestion":
                    # Best case for congestion: tokens prefer local experts first, then nearest
                    # neighbors, minimizing average hop distance and spreading traffic.
                    # Algorithm:
                    # 1. Fill local device first (up to experts_per_device)
                    # 2. Alternate between CCW and CW neighbors, picking 1 expert at a time
                    # 3. When a device is full, move to the next device in that direction
                    #
                    # Example with k=8, experts_per_device=2:
                    # - 2 local
                    # - 2 from CCW neighbor (1-hop)
                    # - 2 from CW neighbor (1-hop)
                    # - 1 from CW 2-hop neighbor
                    # - 1 from CCW 2-hop neighbor
                    devices = mesh_shape[0] * mesh_shape[1]
                    experts_per_device = experts // devices
                    # Batch is sharded in chunks across devices (not round-robin)
                    # Device 0 gets batch 0 to batch/devices-1, device 1 gets next chunk, etc.
                    batches_per_device = batch // devices
                    src_device = b // batches_per_device

                    # Build list of (device, local_expert_idx) for all k experts
                    if k == 0:
                        # First expert slot - start building the selection list
                        # We need to compute all k experts at once for this token
                        picked = []
                        remaining = selected_experts_k

                        # Step 1: Fill local device
                        local_count = min(remaining, experts_per_device)
                        for i in range(local_count):
                            picked.append((src_device, i))
                        remaining -= local_count

                        # Step 2: Alternate CCW and CW
                        ccw_hop = 1
                        cw_hop = 1
                        ccw_count = 0  # experts picked from current CCW device
                        cw_count = 0  # experts picked from current CW device
                        use_ccw = True  # alternate starting with CCW

                        while remaining > 0:
                            if use_ccw:
                                ccw_device = (src_device - ccw_hop) % devices
                                if ccw_count < experts_per_device:
                                    picked.append((ccw_device, ccw_count))
                                    ccw_count += 1
                                    remaining -= 1
                                else:
                                    # Move to next CCW device
                                    ccw_hop += 1
                                    ccw_count = 0
                                    continue  # Don't switch direction yet, retry with new device
                            else:
                                cw_device = (src_device + cw_hop) % devices
                                if cw_count < experts_per_device:
                                    picked.append((cw_device, cw_count))
                                    cw_count += 1
                                    remaining -= 1
                                else:
                                    # Move to next CW device
                                    cw_hop += 1
                                    cw_count = 0
                                    continue  # Don't switch direction yet, retry with new device
                            use_ccw = not use_ccw  # Alternate direction

                        # Store the selection in a way we can access for subsequent k values
                        # We use a simple approach: just compute all k at once and store in tensor
                        for idx, (device, local_idx) in enumerate(picked):
                            expert_id = device * experts_per_device + local_idx
                            expert_indices[b, 0, s, idx] = expert_id
                    # For k > 0, the values were already set when k == 0
                elif scheme == "worst_congestion_descending":
                    # Worst case for sparse multicast: send to furthest device first (antipode),
                    # then progressively closer devices. This maximizes the hop distance for
                    # each packet, as opposed to worst_congestion (ascending) which starts
                    # from the nearest neighbor.
                    #
                    # For a 16-device ring from src_device:
                    # - k=0: hop 8 (antipode, maximum distance)
                    # - k=1: hop 7
                    # - k=2: hop 6
                    # - ...
                    # - k=7: hop 1 (nearest neighbor)
                    # - k=8: hop 0 (local, if k > devices/2)
                    #
                    # This is the worst case for sparse multicast because:
                    # 1. Each packet travels maximum distance before hitting any destination
                    # 2. No benefit from early delivery along the path
                    # 3. Maximum total hop-distance across all packets
                    devices = mesh_shape[0] * mesh_shape[1]
                    experts_per_device = experts // devices

                    # Batch is sharded in chunks across devices
                    batches_per_device = batch // devices
                    src_device = b // batches_per_device

                    # Start from antipode (devices // 2 hops away) and decrement
                    antipode_hop = devices // 2
                    hop_distance = max(0, antipode_hop - k)  # Clamp to 0 for local when k > antipode
                    target_device = (src_device + hop_distance) % devices

                    # Expert offset: if we wrap around and visit a device multiple times,
                    # pick a different expert on that device each time
                    expert_offset = k // devices
                    expert_id = target_device * experts_per_device + (expert_offset % experts_per_device)

                    expert_indices[b, 0, s, k] = expert_id
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
    assert experts % devices == 0
    assert selected_experts_k < experts

    input_tokens = gen_tokens(batch, hidden_size, seq_len, mesh_shape, devices, scheme, dtype)
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, seq_len, mesh_shape, scheme)

    output_tensor = get_output_tensor(input_tokens, expert_indices, expert_mapping, seq_len, mesh_shape, dtype)
    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping, mesh_shape)

    # create expert indices
    return input_tokens, expert_indices, expert_mapping, output_tensor, metadata_tensor


def get_mesh_mapper(mesh_device, mesh_shape, cluster_axis, shard_dim):
    if cluster_axis is None:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim)
    elif cluster_axis == 1:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, shard_dim), mesh_shape=mesh_shape)
    elif cluster_axis == 0:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(shard_dim, None), mesh_shape=mesh_shape)
    else:
        raise ValueError(f"Invalid cluster_axis: {cluster_axis}")
    return mesh_mapper


# we import and use this function in tests.sweeps_framework.sweeps.ccl.generality.all_to_all_dispatch.py
# so be sure to carry over interface changes!
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
    topology=None,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    cluster_axis=1,
    use_optional_output_tensors=False,
    test_skew=False,
    shard_dim=0,
):
    use_sub_devices = False
    torch.manual_seed(2005)
    random.seed(2005)
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
    mesh_mapper = get_mesh_mapper(mesh_device, mesh_shape, cluster_axis, shard_dim)

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
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim),
        )

        tt_metadata_tensor = ttnn.from_torch(
            preallocated_metadata_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=output_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim),
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
    if use_sub_devices:
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    tt_out_tensor_list = []
    if test_skew:
        delays = []
        for i in range(mesh_shape[0]):
            delay_at_i = []
            for j in range(mesh_shape[1]):
                delay_at_i.append(0)
            delays.append(delay_at_i)
        delays[0][0] = 400000

    def run_op(n_iters, store_all_results=True):
        tt_output_list = []
        tt_metadata_list = []

        for i in range(n_iters):
            buffer_index = i
            if test_skew:
                ttnn.apply_device_delay(mesh_device, delays)
            output_tensor, metadata_tensor = ttnn.all_to_all_dispatch(
                input_tensors[buffer_index],
                expert_indices_tensors[buffer_index],
                expert_mapping_tensors[buffer_index],
                cluster_axis=cluster_axis,
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                subdevice_id=worker_sub_device_id,
                output_tensors=[output_tensors[buffer_index], metadata_tensors[buffer_index]]
                if use_optional_output_tensors
                else None,
            )

            tt_out_tensor = output_tensors[buffer_index] if use_optional_output_tensors else output_tensor
            tt_metadata = metadata_tensors[buffer_index] if use_optional_output_tensors else metadata_tensor

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
        tt_out_tensor_list, tt_metadata_list = run_op(1, store_all_results=True)
        ttnn.synchronize_device(mesh_device)

        logger.info("Capturing Warmup")

        if warmup_iters > 0:
            logger.info(f"Capturing Warmup {warmup_iters} iterations")
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            tt_out_tensor_list, tt_metadata_list = run_op(warmup_iters, store_all_results=True)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)
        logger.info("Warmup done")

        logger.info("Capturing Trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_out_tensor_list, tt_metadata_list = run_op(num_iters, store_all_results=True)
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

    for tensor_index in range(len(tt_out_tensor_list)):
        tt_torch_tensor = ttnn.to_torch(
            tt_out_tensor_list[tensor_index],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim),
        )

        tt_metadata_tensor = ttnn.to_torch(
            tt_metadata_list[tensor_index],
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim),
        )

        if is_unsigned_tensor(tt_metadata_tensor):
            tt_metadata_tensor = tt_metadata_tensor.to(output_metadata_goldens_list[tensor_index].dtype)

        batch = tt_torch_tensor.shape[1]
        devices = tt_metadata_tensor.shape[0]
        selected_experts_k = tt_metadata_tensor.shape[3]

        metadata_all_close = torch.allclose(tt_metadata_tensor, output_metadata_goldens_list[tensor_index])
        metadata_all_equal = torch.equal(tt_metadata_tensor, output_metadata_goldens_list[tensor_index])
        if not metadata_all_close or not metadata_all_equal:
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
                            is_all_equal = torch.equal(
                                tt_torch_tensor[d, b, s, :], output_tensor_goldens_list[tensor_index][d, b, s, :]
                            )
                            if not is_all_equal:
                                logger.info(
                                    f"Output tensor {tensor_index} mismatch at batch {b}, sequence {s}, expert {expert_id}, device {d}"
                                )

                            if not is_all_equal:
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
    num_program_cache_entries = 1
    if test_skew:
        num_program_cache_entries = 2
    logger.info(f"Device has {mesh_device.num_program_cache_entries()} program cache entries")
    assert (
        mesh_device.num_program_cache_entries() == num_program_cache_entries
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
    ids=["fabric_2d", "fabric_1d_linear"],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [False])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["cluster_row", "cluster_col"])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "batches_per_device, seq_len, num_iters, warmup_iters, input_memory_config, output_memory_config",
    [
        (16, 2, 2, 1, ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),  # b16s2, dram-dram
        (1, 3, 2, 1, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),  # b1s3, l1-l1
    ],
    ids=["b16s2-dram", "b1s3-l1"],
)
@pytest.mark.parametrize("num_links", ["MAX_LINKS"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_all_to_all_dispatch_no_trace(
    mesh_device,
    trace_mode,
    mesh_shape,
    cluster_axis,
    experts_per_device,
    select_experts_k,
    hidden_size,
    batches_per_device,
    seq_len,
    num_iters,
    warmup_iters,
    input_memory_config,
    output_memory_config,
    num_links,
    dtype,
    device_params,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    if num_links == "MAX_LINKS":
        num_links = get_max_links(cluster_axis, device_params["fabric_config"])

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
    ids=["fabric_2d", "fabric_1d_linear"],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=["cluster_axis_0", "cluster_axis_1"])
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
@pytest.mark.parametrize("num_links", ["MAX_LINKS"])
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
    dtype,
    input_memory_config,
    output_memory_config,
    device_params,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    if num_links == "MAX_LINKS":
        num_links = get_max_links(cluster_axis, device_params["fabric_config"])

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
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
        use_optional_output_tensors=True,
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
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
        use_optional_output_tensors=True,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 50000},
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((1, 8), (1, 8), id="1x8_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1], ids=["cluster_row"])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "batches_per_device, seq_len, num_iters, warmup_iters",
    [
        (16, 7, 10, 5),
    ],
    ids=["b16s2"],
)
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring, ttnn.Topology.Linear])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
def test_all_to_all_dispatch_ring_trace(
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
    device_params,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    if num_links == "MAX_LINKS":
        num_links = get_max_links(cluster_axis, device_params["fabric_config"])

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
        scheme="sequential",
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
        topology=topology,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 50000},
    ],
    indirect=True,
)
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((1, 8), (1, 8), id="1x8_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1], ids=["cluster_row"])
@pytest.mark.parametrize("experts_per_device", [8])
@pytest.mark.parametrize("select_experts_k", [8])
@pytest.mark.parametrize("hidden_size", [600])
@pytest.mark.parametrize(
    "batches_per_device, seq_len, num_iters, warmup_iters",
    [
        (2, 2, 20, 5),
    ],
    ids=["b2s2"],
)
@pytest.mark.parametrize("topology", [ttnn.Topology.Ring])
@pytest.mark.parametrize("num_links", [None])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG], ids=["l1"])
def test_all_to_all_dispatch_skew(
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
    device_params,
):
    if cluster_axis is None:
        dispatch_devices = mesh_shape[0] * mesh_shape[1]
    else:
        dispatch_devices = mesh_shape[cluster_axis]

    batch = batches_per_device * dispatch_devices
    experts = experts_per_device * dispatch_devices

    if num_links == "MAX_LINKS":
        num_links = get_max_links(cluster_axis, device_params["fabric_config"])

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
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
        dtype=dtype,
        cluster_axis=cluster_axis,
        topology=topology,
        test_skew=True,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((2, 4), (2, 4), id="2x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize("cluster_axis", [1], ids=["cluster_col"])
@pytest.mark.parametrize("experts", [8])
@pytest.mark.parametrize("select_experts_k", [2])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize(
    "batch, seq_len",
    [
        (1, 8),
    ],
    ids=["b1s8"],
)
@pytest.mark.parametrize("num_links", ["MAX_LINKS"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
def test_all_to_all_dispatch_no_trace_batch1(
    mesh_device,
    mesh_shape,
    cluster_axis,
    batch,
    experts,
    select_experts_k,
    hidden_size,
    seq_len,
    num_links,
    dtype,
    input_memory_config,
    output_memory_config,
    device_params,
):
    if num_links == "MAX_LINKS":
        num_links = get_max_links(cluster_axis, device_params["fabric_config"])
    devices = mesh_shape[0] * mesh_shape[1]
    input_tokens, expert_indices, expert_mapping, sparse_output_token_tensor, metadata_tensor = gen_tensors(
        batch,
        experts,
        select_experts_k,
        hidden_size,
        seq_len,
        mesh_shape,
        devices,
        scheme="random",
        dtype=tt_to_torch_dtype(dtype),
    )

    logger.info(f"Batch = 1 input_tokens global shape: {input_tokens.shape}")
    logger.info(f"Batch = 1 expert_indices global shape: {expert_indices.shape}")

    mesh_mapper = get_mesh_mapper(mesh_device, mesh_shape, cluster_axis, 2)

    tt_input_tokens = ttnn.from_torch(
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

    logger.info(f"Batch = 1 tt_input_tokens per-device shape: {tt_input_tokens.shape}")
    logger.info(f"Batch = 1 tt_expert_indices per-device shape: {tt_expert_indices.shape}")
    logger.info(f"Batch = 1 tt_expert_mapping per-device shape: {tt_expert_mapping.shape}")

    tt_output_tensor, tt_metadata_tensor = ttnn.all_to_all_dispatch(
        tt_input_tokens,
        tt_expert_indices,
        tt_expert_mapping,
        cluster_axis=cluster_axis,
        num_links=num_links,
        memory_config=output_memory_config,
        output_concat_dim=2,
    )

    logger.info(f"Batch = 1 tt_output_tensor per-device shape: {tt_output_tensor.shape}")
    logger.info(f"Batch = 1 tt_metadata_tensor per-device shape: {tt_metadata_tensor.shape}")

    torch_tt_output_tensor = ttnn.to_torch(
        tt_output_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
    )

    torch_tt_metadata_tensor = ttnn.to_torch(
        tt_metadata_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2),
    )

    # reshape to (-1, H)/
    torch_tt_output_tensor = torch_tt_output_tensor.reshape(-1, torch_tt_output_tensor.shape[-1])
    torch_tt_metadata_tensor = torch_tt_metadata_tensor.reshape(-1, torch_tt_metadata_tensor.shape[-1])
    sparse_output_token_tensor = sparse_output_token_tensor.reshape(-1, sparse_output_token_tensor.shape[-1])
    metadata_tensor = metadata_tensor.reshape(-1, metadata_tensor.shape[-1])

    torch.allclose(torch_tt_output_tensor, sparse_output_token_tensor)
    torch.allclose(torch_tt_metadata_tensor, metadata_tensor)
