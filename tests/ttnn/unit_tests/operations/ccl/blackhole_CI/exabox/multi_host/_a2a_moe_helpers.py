# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
#  SPDX-License-Identifier: Apache-2.0

"""All-to-all dispatch + combine MoE test helpers.

Vendored from tests/nightly/t3000/ccl/test_all_to_all_dispatch.py and
test_all_to_all_combine.py so the exabox tests in this folder are self-contained
(no imports from tests/nightly/t3000/...).

Public surface:

  - run_all_to_all_dispatch_test(...) — used by test_all_to_all_dispatch_exabox.py
  - run_all_to_all_combine_test(...)  — used by test_all_to_all_combine_exabox.py

Everything else here is a private dependency of those two functions.
"""

import random

import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import is_unsigned_tensor
from tracy import signpost


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tensor generation (shared by dispatch + combine)
# ---------------------------------------------------------------------------


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
                    available_experts = [
                        e
                        for e in range(experts)
                        if e not in current_indices and expert_token_count[e] < max_tokens_per_expert
                    ]
                    if not available_experts:
                        available_experts = [e for e in range(experts) if e not in current_indices]
                    chosen_expert = random.choice(available_experts)
                    expert_indices[b, 0, s, k] = chosen_expert
                    expert_token_count[chosen_expert] += 1
                elif scheme == "worst_perf":
                    expert_indices[b, 0, s, k] = experts - 1
                elif scheme == "worst_congestion":
                    devices = mesh_shape[0] * mesh_shape[1]
                    experts_per_device = experts // devices
                    batches_per_device = batch // devices
                    src_device = b // batches_per_device
                    target_device = (src_device + 1 + k) % devices
                    expert_offset = k // devices
                    expert_id = target_device * experts_per_device + (expert_offset % experts_per_device)
                    expert_indices[b, 0, s, k] = expert_id
                elif scheme == "best_congestion":
                    devices = mesh_shape[0] * mesh_shape[1]
                    experts_per_device = experts // devices
                    batches_per_device = batch // devices
                    src_device = b // batches_per_device

                    if k == 0:
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
                        ccw_count = 0
                        cw_count = 0
                        use_ccw = True

                        while remaining > 0:
                            if use_ccw:
                                ccw_device = (src_device - ccw_hop) % devices
                                if ccw_count < experts_per_device:
                                    picked.append((ccw_device, ccw_count))
                                    ccw_count += 1
                                    remaining -= 1
                                else:
                                    ccw_hop += 1
                                    ccw_count = 0
                                    continue
                            else:
                                cw_device = (src_device + cw_hop) % devices
                                if cw_count < experts_per_device:
                                    picked.append((cw_device, cw_count))
                                    cw_count += 1
                                    remaining -= 1
                                else:
                                    cw_hop += 1
                                    cw_count = 0
                                    continue
                            use_ccw = not use_ccw

                        for idx, (device, local_idx) in enumerate(picked):
                            expert_id = device * experts_per_device + local_idx
                            expert_indices[b, 0, s, idx] = expert_id
                    # For k > 0, the values were already set when k == 0
                elif scheme == "worst_congestion_descending":
                    devices = mesh_shape[0] * mesh_shape[1]
                    experts_per_device = experts // devices
                    batches_per_device = batch // devices
                    src_device = b // batches_per_device

                    antipode_hop = devices // 2
                    hop_distance = max(0, antipode_hop - k)
                    target_device = (src_device + hop_distance) % devices

                    expert_offset = k // devices
                    expert_id = target_device * experts_per_device + (expert_offset % experts_per_device)
                    expert_indices[b, 0, s, k] = expert_id
                else:
                    raise ValueError(f"Invalid scheme: {scheme}")
    return expert_indices


def get_output_tensor(input_tokens, expert_indices, expert_mapping, seq_len, mesh_shape, dtype=torch.bfloat16):
    # output tensor is [devices, batch, seq_len, hidden_size]
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


# ---------------------------------------------------------------------------
# Dispatch path
# ---------------------------------------------------------------------------


def _gen_dispatch_tensors(
    batch, experts, selected_experts_k, hidden_size, seq_len, mesh_shape, devices, scheme="random", dtype=torch.bfloat16
):
    assert experts % devices == 0
    assert selected_experts_k < experts

    input_tokens = gen_tokens(batch, hidden_size, seq_len, mesh_shape, devices, scheme, dtype)
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, seq_len, mesh_shape, scheme)

    output_tensor = get_output_tensor(input_tokens, expert_indices, expert_mapping, seq_len, mesh_shape, dtype)
    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping, mesh_shape)

    return input_tokens, expert_indices, expert_mapping, output_tensor, metadata_tensor


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
    torch.manual_seed(2005)
    random.seed(2005)
    mesh_device.enable_program_cache()
    devices = mesh_shape[0] * mesh_shape[1]
    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    mesh_device.cache_entries_counter = CacheEntriesCounter(mesh_device)

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
        (
            input_tokens,
            expert_indices,
            expert_mapping,
            sparse_output_token_tensor,
            metadata_tensor,
        ) = _gen_dispatch_tensors(
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

    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
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
        with mesh_device.cache_entries_counter.measure():
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
        logger.info("Compiling model")
        run_op(1, store_all_results=True)
        ttnn.synchronize_device(mesh_device)

        if warmup_iters > 0:
            logger.info(f"Capturing Warmup {warmup_iters} iterations")
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            run_op(warmup_iters, store_all_results=True)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)

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
    logger.info(f"Device has {mesh_device.cache_entries_counter.total} program cache entries")
    assert (
        mesh_device.cache_entries_counter.total == num_program_cache_entries
    ), f"Device has {mesh_device.cache_entries_counter.total} program cache entries"

    if not metadata_passed:
        logger.info(f"Failed metadata indices: {failed_metadata_indices}")
        assert metadata_passed, f"{first_failed_metadata_index} FAILED metadata indices: {failed_metadata_indices}"

    if not passed:
        logger.info(f"Failed data indices: {failed_indices}")
        assert (
            passed
        ), f"First failing index: {first_failed_tensor_index} batch {first_failed_batch_index} sequence {first_failed_sequence_index} expert {first_failed_expert_index} device {first_failed_device_index} FAILED data indices: {failed_indices}"


# ---------------------------------------------------------------------------
# Combine path
# ---------------------------------------------------------------------------


def _get_experts_on_device(num_experts, expert_mapping, device):
    return [e for e in range(num_experts) if expert_mapping[0, 0, e, device] == 1]


def _get_cluster_dims(replication_axis, mesh_shape):
    if replication_axis == 1:
        replication_dim = mesh_shape[0]
        replication_group = mesh_shape[1]
    elif replication_axis == 0:
        replication_dim = mesh_shape[1]
        replication_group = mesh_shape[0]
    else:
        assert replication_axis == -1
        replication_dim = 1
        replication_group = mesh_shape[0] * mesh_shape[1]
    return replication_dim, replication_group, mesh_shape[0] * mesh_shape[1]


def _get_batch_cluster_idxr(replication_axis, batch):
    def _idxr(m0, m1, b):
        if replication_axis == 0:
            return m1 * batch + b
        elif replication_axis == 1:
            return m0 * batch + b
        else:
            return b

    return _idxr


def _get_input_sparse_contribs(
    sparse_tokens, expert_indices, expert_mapping, mesh_shape, axis, apply_fake_expert=True, local_reduce=False
):
    batch = expert_indices.shape[0]
    devices = expert_mapping.shape[-1]
    experts = expert_mapping.shape[-2]
    hidden_size = sparse_tokens.shape[-1]
    selected_experts_k = expert_indices.shape[-1]
    seq = sparse_tokens.shape[-2]

    assert experts % devices == 0
    experts_per_device = experts // devices

    if local_reduce:
        expert_dim = devices
        expert_idxr = lambda d, _: d
    else:
        expert_dim = experts
        expert_idxr = lambda d, local_idx: d * experts_per_device + local_idx

    input_contribs_tensor = torch.zeros([expert_dim, batch, seq, hidden_size])

    token_expert_count = 0
    for d in range(devices):
        experts_on_device = _get_experts_on_device(experts, expert_mapping, d)
        assert len(experts_on_device) == experts_per_device
        for b in range(batch):
            for k in range(selected_experts_k):
                for s in range(seq):
                    expert_idx = expert_indices[b, 0, s, k].item()
                    if expert_idx not in experts_on_device:
                        continue

                    local_expert_idx = expert_idxr(d, experts_on_device.index(expert_idx))

                    if apply_fake_expert:
                        contrib = sparse_tokens[d, b, s, :] * (-1 if expert_idx == 0 else expert_idx)
                    else:
                        contrib = sparse_tokens[d, b, s, :]
                    input_contribs_tensor[local_expert_idx, b, s, :] += contrib

                    token_expert_count += 1

    assert token_expert_count == batch * seq * selected_experts_k
    return input_contribs_tensor


def _get_output_combined_contribs(
    sparse_contribs, expert_indices, expert_mapping, mesh_shape, replication_axis, local_reduce=False
):
    batch = expert_indices.shape[0]
    experts = expert_mapping.shape[-2]
    selected_experts_k = expert_indices.shape[-1]
    hidden = sparse_contribs.shape[-1]
    seq = sparse_contribs.shape[-2]

    replication_dim, replication_group, devices = _get_cluster_dims(replication_axis, mesh_shape)

    assert experts % devices == 0
    experts_per_device = experts // devices

    batch_rep_idxr = _get_batch_cluster_idxr(replication_axis, batch)

    if local_reduce:
        local_contrib_idx_func = lambda d, _: d
    else:
        local_contrib_idx_func = lambda d, local_idx: d * experts_per_device + local_idx

    output_combined_contribs_tensor = torch.zeros(selected_experts_k, batch * replication_dim, seq, hidden).bfloat16()
    real_data_map = torch.zeros(output_combined_contribs_tensor.shape[:-1])

    total_token_expert_count = 0
    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            d = m0 * mesh_shape[1] + m1
            device_expert_list = _get_experts_on_device(experts, expert_mapping, d)

            for b in range(batch):
                for s in range(seq):
                    token_experts = expert_indices[b, 0, s, :].tolist()
                    for eg in device_expert_list:
                        if eg in token_experts:
                            k = token_experts.index(eg)
                        else:
                            continue

                        axis_batch_idx = batch_rep_idxr(m0, m1, b)
                        local_contrib_idx = local_contrib_idx_func(d, device_expert_list.index(eg))

                        sc = sparse_contribs[local_contrib_idx, b, s, :]
                        output_combined_contribs_tensor[k, axis_batch_idx, s, :] = sc

                        real_data_map[k, axis_batch_idx, s] = 1
                        total_token_expert_count += 1

                        if local_reduce:
                            break
    return output_combined_contribs_tensor, real_data_map


def _gen_combine_tensors(
    batch,
    experts,
    selected_experts_k,
    hidden_size,
    seq,
    mesh_shape,
    replication_axis,
    devices,
    scheme="random",
    local_reduce=False,
):
    torch.manual_seed(20)
    random.seed(20)
    assert experts % devices == 0
    assert selected_experts_k < experts

    input_tokens = gen_tokens(batch, hidden_size, seq, mesh_shape, devices, scheme)
    expert_mapping = gen_expert_mapping(experts, devices, scheme)
    expert_indices = get_expert_indices(batch, experts, selected_experts_k, seq, mesh_shape, scheme)

    # In the combine path, the dispatch's get_output_tensor produces "sparse tokens"
    # (devices x batch x seq x hidden) — same function, different role.
    sparse_dispatched_tokens = get_output_tensor(input_tokens, expert_indices, expert_mapping, seq, mesh_shape)
    input_sparse_contribs_tensor = _get_input_sparse_contribs(
        sparse_dispatched_tokens,
        expert_indices,
        expert_mapping,
        mesh_shape,
        replication_axis,
        local_reduce=local_reduce,
    )

    output_tensor, data_map = _get_output_combined_contribs(
        input_sparse_contribs_tensor,
        expert_indices,
        expert_mapping,
        mesh_shape,
        replication_axis,
        local_reduce=local_reduce,
    )

    metadata_tensor = get_metadata_tensor(expert_indices, expert_mapping, mesh_shape)

    return (
        sparse_dispatched_tokens,
        input_sparse_contribs_tensor,
        expert_mapping,
        metadata_tensor,
        output_tensor,
        data_map,
    )


def _check_combine_results(test_tensor, ref_tensor, data_map):
    for k in range(ref_tensor.shape[0]):
        for b in range(ref_tensor.shape[1]):
            for s in range(ref_tensor.shape[2]):
                if data_map[k, b, s].item() == 1:
                    assert torch.equal(
                        test_tensor[k, b, s, :], ref_tensor[k, b, s, :]
                    ), f"Equal check failed for k={k}, b={b}, s={s} with test_tensor {test_tensor[k, b, s, :]} and ref_tensor {ref_tensor[k, b, s, :]}"


def run_all_to_all_combine_test(
    mesh_device,
    mesh_shape,
    axis,
    batch,
    seq,
    local_reduce,
    experts,
    select_experts_k,
    hidden_size,
    num_iters,
    num_links,
    scheme="random",
    use_regular_grid=False,
    input_grid=None,
    output_grid=None,
    dtype=ttnn.bfloat16,
    topology=None,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    test_skew=False,
):
    if test_skew and local_reduce:
        pytest.skip("Skip skew test for local reduce")
    devices = mesh_shape[0] * mesh_shape[1]

    expert_mapping_tensors = []
    input_tensors = []
    metadata_tensors = []
    output_tensor_goldens_list = []

    for iter in range(num_iters):
        _, input_contrib, expert_mapping, metadata_tensor, output_contrib_tensor, data_map = _gen_combine_tensors(
            batch,
            experts,
            select_experts_k,
            hidden_size,
            seq,
            mesh_shape,
            axis,
            devices,
            scheme=scheme,
            local_reduce=local_reduce,
        )

        output_tensor_goldens_list.append((output_contrib_tensor, data_map))

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
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape),
        )
        tt_metadata = ttnn.from_torch(
            metadata_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        input_tensors.append(tt_input_contribs)
        expert_mapping_tensors.append(tt_expert_mapping)
        metadata_tensors.append(tt_metadata)

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
        for i in range(n_iters):
            if test_skew:
                ttnn.apply_device_delay(mesh_device, delays)
            tt_out_tensor = ttnn.all_to_all_combine(
                input_tensors[i],
                metadata_tensors[i],
                expert_mapping_tensors[i],
                num_links=num_links,
                topology=topology,
                memory_config=output_memory_config,
                local_reduce=local_reduce,
                cluster_axis=axis,
            )

            ttnn.synchronize_device(mesh_device)
            if store_all_results:
                tt_output_list.append(tt_out_tensor)
        if store_all_results:
            return tt_output_list
        else:
            return [tt_out_tensor]

    tt_out_tensor_list = run_op(num_iters, store_all_results=True)

    for tt_out, (ref, data_map) in zip(tt_out_tensor_list, output_tensor_goldens_list):
        if axis == 0:
            # transposed ordering: gather per-device shards manually because the
            # output's natural device order doesn't compose into output dim 1 directly
            # via ConcatMeshToTensor.
            device_shards = [ttnn.to_torch(ittout, mesh_composer=None) for ittout in ttnn.get_device_tensors(tt_out)]
            ordered_shards = []
            for ir in range(mesh_shape[1]):
                for ic in range(mesh_shape[0]):
                    ordered_shards.append(device_shards[ic * mesh_shape[1] + ir])
            tt_out_agg = torch.cat(ordered_shards, dim=1)
        else:
            tt_out_agg = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
        _check_combine_results(tt_out_agg, ref, data_map)
