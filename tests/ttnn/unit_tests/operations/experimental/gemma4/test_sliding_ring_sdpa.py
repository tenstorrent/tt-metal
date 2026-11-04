# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.gemma4.sliding_ring_sdpa matches the shared ring joint SDPA bit for bit on Gemma4's sliding configs.

Each test reruns a shared-op sliding test from tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py twice in one
process, once as written and once with its sliding calls routed to the fork, and compares every tensor the test
reads back. The shared tests' own accuracy and determinism checks run in both passes. Trace replay is covered by
rerunning the whole capture and replay sequence. The lanes tests check several requests per call against one call
per request.
"""

import functools
import math
import random
from contextlib import contextmanager
from unittest import mock

import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.blackhole.sdpa import test_ring_joint_sdpa as shared
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    CHUNKED_PREFILL_PCC_THRESHOLD,
    CHUNKED_PREFILL_SEED,
    DEFAULT_RMSE_THRESHOLD,
    RING_JOINT_TRACE_REGION_SIZE,
    assert_pcc_rmse,
    build_kv_pad_rotation_inputs,
    close_ring_joint_sdpa_runtime,
    fa_rand,
    gpt_oss_chunked_mesh_config,
    open_ring_joint_sdpa_runtime,
    to_balanced_growing_cache_layout,
    torch_chunked_causal_sdpa_reference,
)

_shared_op = ttnn.transformer.ring_joint_scaled_dot_product_attention


def _fork_supports(kwargs, joint_tensors):
    return (
        kwargs.get("sliding_window_size") is not None
        and all(t is None for t in joint_tensors)
        and kwargs.get("is_causal", False)
        and not kwargs.get("is_balanced", False)
        and not kwargs.get("is_cross", False)
        and kwargs.get("attention_sink") is None
        and not kwargs.get("circular_kv_cache", False)
        and kwargs.get("kv_valid_end_tensor") is None
        and kwargs.get("logical_l", 0) == 0
        and isinstance(kwargs["logical_n"], int)
        and kwargs.get("persistent_output_buffer_joint_k") is None
        and kwargs.get("persistent_output_buffer_joint_v") is None
    )


def _call_fork(q, k, v, kwargs):
    return ttnn.experimental.gemma4.sliding_ring_sdpa(
        q,
        k,
        v,
        persistent_output_buffer_k=kwargs["persistent_output_buffer_k"],
        persistent_output_buffer_v=kwargs["persistent_output_buffer_v"],
        logical_n=kwargs["logical_n"],
        program_config=kwargs["program_config"],
        sliding_window_size=kwargs["sliding_window_size"],
        scale=kwargs.get("scale"),
        compute_kernel_config=kwargs.get("compute_kernel_config"),
        dim=kwargs["dim"],
        multi_device_global_semaphore=kwargs["multi_device_global_semaphore"],
        num_links=kwargs["num_links"],
        cluster_axis=kwargs["cluster_axis"],
        mesh_device=kwargs["mesh_device"],
        topology=kwargs["topology"],
        subdevice_id=kwargs.get("subdevice_id"),
        ccl_core_grid_offset=kwargs["ccl_core_grid_offset"],
        use_column_major_ccl=kwargs.get("use_column_major_ccl", False),
        kv_cache_batch_idx=kwargs.get("kv_cache_batch_idx"),
        kv_actual_isl=kwargs.get("kv_actual_isl"),
        slot_id=kwargs.get("slot_id"),
        kv_actual_isl_tensor=kwargs.get("kv_actual_isl_tensor"),
        kv_cache_num_layers=kwargs.get("kv_cache_num_layers"),
        kv_cache_layer_idx=kwargs.get("kv_cache_layer_idx"),
    )


@contextmanager
def _route_sliding_to_fork(calls, **overrides):
    """Route the shared op's sliding calls that the fork supports to the fork, with overrides applied to their
    arguments. calls["fork"] counts them; calls["shared_sliding"] counts sliding calls the fork cannot take, which
    stay on the shared op."""

    def routed(q, k, v, joint_q=None, joint_k=None, joint_v=None, **kwargs):
        if not _fork_supports(kwargs, (joint_q, joint_k, joint_v)):
            if kwargs.get("sliding_window_size") is not None:
                calls["shared_sliding"] += 1
            return _shared_op(q, k, v, joint_q, joint_k, joint_v, **kwargs)
        calls["fork"] += 1
        return _call_fork(q, k, v, {**kwargs, **overrides}), None, None

    with mock.patch.object(ttnn.transformer, "ring_joint_scaled_dot_product_attention", routed):
        yield


@contextmanager
def _record_readbacks(readbacks):
    to_torch = ttnn.to_torch

    def recording(*args, **kwargs):
        result = to_torch(*args, **kwargs)
        readbacks.append(result.clone())
        return result

    with mock.patch.object(ttnn, "to_torch", recording):
        yield


def _run_pass(test_fn, kwargs, *, fork):
    torch.manual_seed(0)
    random.seed(0)
    readbacks, calls = [], {"fork": 0, "shared_sliding": 0}
    with _record_readbacks(readbacks):
        if fork:
            with _route_sliding_to_fork(calls):
                test_fn(**kwargs)
        else:
            test_fn(**kwargs)
    return readbacks, calls


def _matches_shared(test_fn):
    """Rerun test_fn (and its parametrization) through the fork and require identical readbacks."""

    @functools.wraps(test_fn)
    def wrapper(**kwargs):
        shared_readbacks, _ = _run_pass(test_fn, kwargs, fork=False)
        fork_readbacks, calls = _run_pass(test_fn, kwargs, fork=True)
        assert calls["fork"] > 0, "No sliding call reached the fork"
        assert calls["shared_sliding"] == 0, f"{calls['shared_sliding']} sliding calls stayed on the shared op"
        assert len(fork_readbacks) == len(shared_readbacks)
        for i, (expected, actual) in enumerate(zip(shared_readbacks, fork_readbacks)):
            assert expected.shape == actual.shape, f"readback {i}: {expected.shape} vs {actual.shape}"
            assert torch.equal(expected, actual), f"readback {i} differs from the shared op"

    return wrapper


def _through_fork(test_fn):
    """Run test_fn (and its parametrization) once with its sliding calls on the fork."""

    @functools.wraps(test_fn)
    def wrapper(**kwargs):
        calls = {"fork": 0, "shared_sliding": 0}
        with _route_sliding_to_fork(calls):
            test_fn(**kwargs)
        assert calls["fork"] > 0, "No sliding call reached the fork"
        assert calls["shared_sliding"] == 0, f"{calls['shared_sliding']} sliding calls stayed on the shared op"

    return wrapper


# Gemma4 sliding geometry: complete groups, multi-hop halos, block-cyclic reuse, K split.
test_gemma_complete_group_sliding_geometry = _matches_shared(
    shared.test_ring_joint_attention_gemma_complete_group_sliding_geometry
)
test_gemma_multi_hop_sliding_halo_geometry = _matches_shared(
    shared.test_ring_joint_attention_gemma_multi_hop_sliding_halo_geometry
)
test_gemma_sliding_ksplit_accuracy = _matches_shared(shared.test_ring_joint_attention_gemma_sliding_ksplit_accuracy)
test_block_cyclic_sliding_reuse = _matches_shared(shared.test_ring_joint_attention_block_cyclic_sliding_reuse)
test_multi_hop_block_cyclic_sliding_reuse = _matches_shared(
    shared.test_ring_joint_attention_multi_hop_block_cyclic_sliding_reuse
)
test_multi_hop_wrapping_block_cyclic_sliding_reuse = _matches_shared(
    shared.test_ring_joint_attention_multi_hop_wrapping_block_cyclic_sliding_reuse
)
test_multi_hop_sliding_halo_linear_topology_accuracy = _matches_shared(
    shared.test_ring_joint_attention_multi_hop_sliding_halo_linear_topology_accuracy
)

# Trace replay with the slot/prefix metadata refreshed per replay (eager, program-cache hit and replay).
test_metadata_trace_replay_sliding = _matches_shared(
    shared.test_ring_joint_metadata_trace_replay_mixed_sliding_dense_three_semaphores
)

# Gemma4's production sliding chunks: the shared perf gate's utilization bands, measured on the fork. The shared
# test's other marks (the power guard) are kept; its parametrization is narrowed to the sliding configs.
test_gemma4_chunked_perf_check = _through_fork(shared.test_ring_joint_attention_gemma4_chunked_perf_check)
test_gemma4_chunked_perf_check.pytestmark = [
    m for m in test_gemma4_chunked_perf_check.pytestmark if m.name != "parametrize"
]
test_gemma4_chunked_perf_check = pytest.mark.parametrize(
    "perf_config",
    [c for c in shared.GEMMA4_CHUNKED_PRODUCTION_PERF_CONFIGS if c.model.sliding_window_size is not None],
    ids=shared.gemma4_built_tuning_id,
)(test_gemma4_chunked_perf_check)


def test_rejects_fp32_dest_acc(expect_error):
    """The fork has only the streaming compute path; the shared op would silently build its v1 path instead."""

    def fp32_dest_acc(config):
        return ttnn.init_device_compute_kernel_config(
            ttnn.device.Arch.BLACKHOLE,
            math_fidelity=config.math_fidelity,
            math_approx_mode=config.math_approx_mode,
            fp32_dest_acc_en=True,
            packer_l1_acc=config.packer_l1_acc,
        )

    calls = {"fork": 0, "shared_sliding": 0}

    def routed_with_fp32(q, k, v, joint_q=None, joint_k=None, joint_v=None, **kwargs):
        if _fork_supports(kwargs, (joint_q, joint_k, joint_v)):
            kwargs["compute_kernel_config"] = fp32_dest_acc(kwargs["compute_kernel_config"])
            calls["fork"] += 1
            return _call_fork(q, k, v, kwargs), None, None
        return _shared_op(q, k, v, joint_q, joint_k, joint_v, **kwargs)

    with mock.patch.object(ttnn.transformer, "ring_joint_scaled_dot_product_attention", routed_with_fp32):
        with expect_error(RuntimeError, "requires streaming compute; set fp32_dest_acc_en=false"):
            shared.test_ring_joint_attention_gemma_complete_group_sliding_geometry(expect_error=expect_error)
    assert calls["fork"] == 1


# Lanes: several requests per call.


def _lane_metadata_tensor(mesh_device, values):
    """One uint32 element per lane, replicated in DRAM; one value is the single-request metadata tensor."""
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int64).reshape(1, 1, 1, len(values)),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _lane_metadata_host_tensor(mesh_device, values):
    """HOST-side twin of _lane_metadata_tensor, for refreshing it in place with copy_host_to_device_tensor."""
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int64).reshape(1, 1, 1, len(values)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _sliding_ring_sdpa_call(
    tt_q,
    tt_k,
    tt_v,
    logical_n,
    is_causal,
    is_balanced,
    p_buf_k,
    p_buf_v,
    program_config,
    compute_kernel_config,
    ccl_semaphore_handles,
    num_links,
    sp_axis,
    mesh_device,
    topology,
    worker_sub_device_id,
    ccl_column,
    *,
    sliding_window_size,
    slot_id,
    kv_actual_isl_tensor,
):
    """The shared tests' call_sdpa, on the fork (always causal and unbalanced)."""
    assert is_causal and not is_balanced
    return ttnn.experimental.gemma4.sliding_ring_sdpa(
        tt_q,
        tt_k,
        tt_v,
        persistent_output_buffer_k=p_buf_k,
        persistent_output_buffer_v=p_buf_v,
        logical_n=logical_n,
        program_config=program_config,
        sliding_window_size=sliding_window_size,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_semaphore_handles,
        num_links=num_links,
        cluster_axis=sp_axis,
        mesh_device=mesh_device,
        topology=topology,
        subdevice_id=worker_sub_device_id,
        ccl_core_grid_offset=(ccl_column, 0),
        use_column_major_ccl=True,
        slot_id=slot_id,
        kv_actual_isl_tensor=kv_actual_isl_tensor,
    )


def run_sliding_lanes_case(
    prefix_chunks,
    step_chunks,
    expect_error,
    chunk_size_local=256,
    sliding_window_size=1024,
    local_q_heads=4,
    local_kv_heads=2,
    head_dim=256,
    q_chunk_size=128,
    k_chunk_size=128,
    replay=False,
    compute_grid=None,
    prefix_offsets=None,
    halo_slots=1,
):
    """Check one lanes call (B-element slot_id / kv_actual_isl) against one metadata-path call per lane.

    With replay, the same B-lane program then serves the lanes in other orders, eagerly on a program-cache hit and
    by trace replay, with the Q, metadata and halo buffers refreshed in place, so every lane's slot and prefix change.

    prefix_chunks[s] is how many whole chunks cache slot s holds before the step; step_chunks lists the step's lanes
    as (slot, chunk index), so lane b's prefix is chunk index * chunk_global, plus prefix_offsets[b] tokens when
    given. Every lane's K/V is in the cache before attention, so a later chunk of the same request attends its earlier
    lanes through the cache. Rows past each slot's filled length are high-magnitude garbage.

    compute_grid replaces the runtime's SDPA grid (default) in the program config. halo_slots sizes each lane's halo
    batch; a lane whose prefix is not a multiple of chunk_size_local wraps a slab and needs two.
    """
    mesh_config = gpt_oss_chunked_mesh_config()
    sp_size = mesh_config.sp_size
    num_lanes = len(step_chunks)
    num_slots = len(prefix_chunks)
    nhq = local_q_heads * mesh_config.tp_size
    nhk = local_kv_heads * mesh_config.tp_size
    gqa_ratio = local_q_heads // local_kv_heads
    chunk_global = chunk_size_local * sp_size
    lane_slots = [slot for slot, _ in step_chunks]
    lane_prefixes = [idx * chunk_global for _, idx in step_chunks]
    if prefix_offsets is not None:
        lane_prefixes = [prefix + offset for prefix, offset in zip(lane_prefixes, prefix_offsets, strict=True)]
    # Tokens each slot holds: its whole prefix chunks, or through the end of its deepest lane's chunk.
    filled = [
        max([n * chunk_global] + [p + chunk_global for s, p in zip(lane_slots, lane_prefixes) if s == slot])
        for slot, n in enumerate(prefix_chunks)
    ]
    # One spare group of garbage past the deepest slot; the host logical_n stays at this capacity and every lane's
    # length comes from its kv_actual_isl element.
    cache_groups = math.ceil(max(filled) / chunk_global) + 1
    cache_seq_per_dev = cache_groups * chunk_size_local
    capacity = sp_size * cache_seq_per_dev
    halo_tokens = math.ceil((sliding_window_size - 1) / k_chunk_size) * k_chunk_size

    torch.manual_seed(CHUNKED_PREFILL_SEED + 901)
    # Each slot's request K/V stream, through its deepest chunk; Q only for the step's chunks.
    k_tokens = [fa_rand(1, nhk, n, head_dim) for n in filled]
    v_tokens = [fa_rand(1, nhk, n, head_dim) for n in filled]
    lane_q = [fa_rand(1, nhq, chunk_global, head_dim) for _ in step_chunks]

    # A chunk-aligned lane's Q is its tokens in order (device d holds tokens [d * local, (d + 1) * local)). A rotated
    # lane's chunk starts mid-group, so each device's Q slab holds the chunk's tokens that land in that device's cache
    # slabs; build_kv_pad_rotation_inputs places them, and valid_rows maps each token to its row for the torch check.
    lane_q_dev, lane_valid_rows = [], []
    for slot, prefix, q in zip(lane_slots, lane_prefixes, lane_q):
        if prefix % chunk_global == 0:
            lane_q_dev.append(q)
            lane_valid_rows.append(list(range(chunk_global)))
            continue
        k, v = k_tokens[slot], v_tokens[slot]
        end = prefix + chunk_global
        q_dev, _, _, valid_rows, _ = build_kv_pad_rotation_inputs(
            k[:, :, :prefix],
            v[:, :, :prefix],
            q,
            k[:, :, prefix:end],
            v[:, :, prefix:end],
            prefix,
            sp_size,
            chunk_size_local,
        )
        lane_q_dev.append(q_dev)
        lane_valid_rows.append(valid_rows)

    def cache_host(tokens):
        # Each slot's filled tokens in the block-cyclic layout (device d's rows [g * local, (g + 1) * local) hold
        # group g's device-d slab); rows past a slot's filled length stay garbage. A slot filled to a length that is
        # not a whole group has its stream padded with garbage to the group's end first.
        per_dev = torch.randn(num_slots, nhk, sp_size, cache_seq_per_dev, head_dim) * 100
        for slot, n in enumerate(filled):
            groups = math.ceil(n / chunk_global)
            stream = tokens[slot]
            if groups * chunk_global > n:
                garbage = torch.randn(1, nhk, groups * chunk_global - n, head_dim) * 100
                stream = torch.cat([stream, garbage], dim=2)
            packed = to_balanced_growing_cache_layout(stream, sp_size, chunk_global, groups - 1)
            rows = groups * chunk_size_local
            per_dev[slot, :, :, :rows] = packed.reshape(nhk, sp_size, rows, head_dim)
        return per_dev.reshape(num_slots, nhk, capacity, head_dim)

    # Each device holds every lane's CP-local slab, request-major: [1, NH, B * local, D].
    lanes_q_host = (
        torch.stack(lane_q_dev, dim=2)
        .reshape(1, nhq, num_lanes, sp_size, chunk_size_local, head_dim)
        .transpose(2, 3)
        .reshape(1, nhq, sp_size * num_lanes * chunk_size_local, head_dim)
    )

    runtime = open_ring_joint_sdpa_runtime(mesh_config, trace_region_size=RING_JOINT_TRACE_REGION_SIZE if replay else 0)
    mesh_device = runtime.mesh_device
    trace_id = None
    try:
        mesh_device.enable_program_cache()
        input_dims = [None, None]
        input_dims[runtime.sp_axis] = 2
        if mesh_config.tp_size > 1:
            input_dims[runtime.tp_axis] = 1
        halo_dims = [None, None]
        if mesh_config.tp_size > 1:
            halo_dims[runtime.tp_axis] = 1

        def upload(host, dtype, dims):
            return ttnn.from_torch(
                host,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            )

        tt_k = upload(cache_host(k_tokens), ttnn.bfloat8_b, input_dims)
        tt_v = upload(cache_host(v_tokens), ttnn.bfloat8_b, input_dims)
        composer = ttnn.create_mesh_composer(
            mesh_device,
            ttnn.MeshComposerConfig(
                input_dims[0] if input_dims[0] is not None else -1,
                input_dims[1] if input_dims[1] is not None else -1,
            ),
        )

        def make_program_config(k_splits):
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=runtime.sdpa_compute_grid if compute_grid is None else compute_grid,
                q_chunk_size=q_chunk_size,
                k_chunk_size=k_chunk_size,
                exp_approx_mode=False,
                k_splits=k_splits,
            )

        program_config = make_program_config(1)

        def halo_buffer(batch):
            # The halo buffer gets one batch per lane.
            return upload(torch.zeros(batch, nhk, halo_slots * halo_tokens, head_dim), ttnn.bfloat8_b, halo_dims)

        def invoke(tt_q, slots, prefixes, *, config=program_config, halo_batch=None, halos=None, metadata=None):
            if halos is None:
                halos = [halo_buffer(len(slots) if halo_batch is None else halo_batch) for _ in range(2)]
                # The zero fill must land on every device before any neighbour's halo exchange writes into it.
                ttnn.synchronize_device(mesh_device)
            if metadata is None:
                metadata = [_lane_metadata_tensor(mesh_device, values) for values in (slots, prefixes)]
            return _sliding_ring_sdpa_call(
                tt_q,
                tt_k,
                tt_v,
                capacity,
                is_causal=True,
                is_balanced=False,
                p_buf_k=halos[0],
                p_buf_v=halos[1],
                program_config=config,
                compute_kernel_config=runtime.compute_kernel_config,
                ccl_semaphore_handles=runtime.ccl_semaphore_handles,
                num_links=runtime.num_links,
                sp_axis=runtime.sp_axis,
                mesh_device=mesh_device,
                topology=runtime.topology,
                worker_sub_device_id=runtime.worker_sub_device_id,
                ccl_column=runtime.ccl_column,
                sliding_window_size=sliding_window_size,
                slot_id=metadata[0],
                kv_actual_isl_tensor=metadata[1],
            )

        logger.info(
            f"sliding lanes: B={num_lanes}, TP={mesh_config.tp_size}, SP={sp_size}, local={chunk_size_local}, "
            f"window={sliding_window_size}, slots={lane_slots}, prefixes={lane_prefixes}, capacity={capacity}, "
            f"grid={runtime.sdpa_compute_grid if compute_grid is None else compute_grid}, halo_slots={halo_slots}"
        )

        per_request = []
        for lane, (slot, prefix) in enumerate(zip(lane_slots, lane_prefixes, strict=True)):
            tt_out = invoke(upload(lane_q_dev[lane], ttnn.bfloat16, input_dims), [slot], [prefix])
            ttnn.synchronize_device(mesh_device)
            per_request.append(ttnn.to_torch(tt_out, mesh_composer=composer))
            ttnn.deallocate(tt_out)

        def stacked_q(order):
            # Each device holds every lane's CP-local slab, request-major: [1, NH, B * local, D].
            return (
                torch.stack([lane_q_dev[lane] for lane in order], dim=2)
                .reshape(1, nhq, num_lanes, sp_size, chunk_size_local, head_dim)
                .transpose(2, 3)
                .reshape(1, nhq, sp_size * num_lanes * chunk_size_local, head_dim)
            )

        def read_lanes(tt_out):
            return (
                ttnn.to_torch(tt_out, mesh_composer=composer)
                .reshape(1, nhq, sp_size, num_lanes, chunk_size_local, head_dim)
                .transpose(2, 3)
                .reshape(1, nhq, num_lanes, chunk_global, head_dim)
            )

        def check_order(lanes_out, order, label):
            # Lane i of this call serves the step's lane order[i], so it must equal that lane's single-request call.
            for position, lane in enumerate(order):
                got, expected = lanes_out[:, :, position], per_request[lane]
                assert torch.equal(got, expected), (
                    f"{label}: position {position} (slot {lane_slots[lane]}, prefix {lane_prefixes[lane]}) differs "
                    f"from its single-request call; max abs diff={(got - expected).abs().max().item()}"
                )

        tt_lanes_q = upload(lanes_q_host, ttnn.bfloat16, input_dims)
        tt_out = invoke(tt_lanes_q, lane_slots, lane_prefixes)
        ttnn.synchronize_device(mesh_device)
        lanes_out = read_lanes(tt_out)
        ttnn.deallocate(tt_out)

        for lane, (slot, prefix) in enumerate(zip(lane_slots, lane_prefixes, strict=True)):
            got, expected = lanes_out[:, :, lane], per_request[lane]
            mismatch = got != expected
            assert not mismatch.any(), (
                f"lane {lane} (slot {slot}, prefix {prefix}) differs from its single-request call: "
                f"mismatches={mismatch.sum().item()}, "
                f"heads={mismatch.any(dim=(0, 2, 3)).nonzero().flatten().tolist()}, "
                f"max abs diff={(got - expected).abs().max().item()}"
            )
            ref_out = torch_chunked_causal_sdpa_reference(
                lane_q[lane],
                k_tokens[slot][:, :, : prefix + chunk_global].repeat_interleave(gqa_ratio, dim=1),
                v_tokens[slot][:, :, : prefix + chunk_global].repeat_interleave(gqa_ratio, dim=1),
                prefix,
                sliding_window_size=sliding_window_size,
            )
            assert_pcc_rmse(
                ref_out,
                got[:, :, lane_valid_rows[lane]],
                CHUNKED_PREFILL_PCC_THRESHOLD,
                DEFAULT_RMSE_THRESHOLD,
                f"sliding lane {lane} (slot {slot}, prefix {prefix})",
            )

        if replay:
            identity = list(range(num_lanes))
            orders = [identity[1:] + identity[:1], identity[::-1]]
            halos = [halo_buffer(num_lanes) for _ in range(2)]
            zero_halo = ttnn.from_torch(
                torch.zeros(num_lanes, nhk, halo_slots * halo_tokens, head_dim),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=halo_dims),
            )
            metadata = [_lane_metadata_tensor(mesh_device, values) for values in (lane_slots, lane_prefixes)]

            def stage(order):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        stacked_q(order),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ShardTensor2dMesh(
                            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=input_dims
                        ),
                    ),
                    tt_lanes_q,
                )
                for halo in halos:
                    ttnn.copy_host_to_device_tensor(zero_halo, halo)
                for tensor, values in zip(
                    metadata, ([lane_slots[lane] for lane in order], [lane_prefixes[lane] for lane in order])
                ):
                    ttnn.copy_host_to_device_tensor(_lane_metadata_host_tensor(mesh_device, values), tensor)
                # Every device's writes must land before any device runs: a neighbour's halo exchange writes into
                # this device's halo buffer over fabric, which this device's queue does not order after its own
                # zero fill, so a late fill could overwrite halo rows that already arrived.
                ttnn.synchronize_device(mesh_device)

            # Program-cache hit: the same B-lane program with every lane's slot and prefix changed.
            entries = mesh_device.num_program_cache_entries()
            stage(orders[0])
            tt_out = invoke(tt_lanes_q, None, None, halos=halos, metadata=metadata)
            ttnn.synchronize_device(mesh_device)
            assert mesh_device.num_program_cache_entries() == entries, "a reordered lanes call compiled a new program"
            check_order(read_lanes(tt_out), orders[0], "cache hit")
            ttnn.deallocate(tt_out)

            # Trace replay: captured in one order, replayed in others, with only the buffers refreshed.
            stage(identity)
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            traced_out = invoke(tt_lanes_q, None, None, halos=halos, metadata=metadata)
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            for order in (orders[1], identity, orders[0]):
                stage(order)
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                check_order(read_lanes(traced_out), order, f"trace replay {order}")
            return

        # A lanes call runs one unit per (lane, head, Q chunk): a K split is rejected at validation.
        with expect_error(RuntimeError, "k_splits must be 1 and segmented accumulation off"):
            invoke(tt_lanes_q, lane_slots, lane_prefixes, config=make_program_config(3))
        # The other lanes contract checks.
        with expect_error(RuntimeError, "must have one element per lane like slot_id"):
            invoke(tt_lanes_q, lane_slots, lane_prefixes[:-1])
        with expect_error(RuntimeError, "must contain 1..8 elements"):
            invoke(tt_lanes_q, [0] * 9, [0] * 9)
        # The program hash splits Q's rows by the lane count before validation runs, so an empty slot_id is rejected there.
        with expect_error(RuntimeError, "slot_id must contain at least one element"):
            invoke(tt_lanes_q, [], [])
        with expect_error(RuntimeError, "Lanes need a halo buffer with one batch per lane"):
            invoke(tt_lanes_q, lane_slots, lane_prefixes, halo_batch=num_lanes + 1)
        # One element per ROW_MAJOR row would put lanes 1.. on pages the reader never reads.
        column = [
            ttnn.from_torch(
                torch.tensor(values, dtype=torch.int64).reshape(1, 1, num_lanes, 1),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            for values in (lane_slots, lane_prefixes)
        ]
        with expect_error(RuntimeError, "must hold all its lane elements in one ROW_MAJOR row"):
            invoke(tt_lanes_q, lane_slots, lane_prefixes, metadata=column)
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)
        close_ring_joint_sdpa_runtime(runtime, clear_program_cache=True)


# name -> (whole chunks each cache slot holds before the step, the step's lanes as (slot, chunk index)).
_SLIDING_LANES_SCENARIOS = {
    "lane0_deepest": ((3, 0, 1, 2), [(0, 3), (1, 0), (2, 1), (3, 2)]),
    # Lane 0 has the shortest (empty) prefix; it alone drives the call's ring-iteration masks.
    "lane0_empty": ((0, 3, 1, 2), [(0, 0), (1, 3), (2, 1), (3, 2)]),
    # One request's consecutive chunks in one step: lane k attends lanes 0..k-1 through the cache.
    "same_one_request": ((3, 0, 0, 0), [(0, 3), (0, 4), (0, 5), (0, 6)]),
    "same_two_by_two": ((2, 0, 0, 0), [(1, 0), (1, 1), (0, 2), (0, 3)]),
    "two_lanes": ((0, 2), [(1, 2), (0, 0)]),
    # An odd lane count (a 12 B metadata read) and the full kMaxLanes, which fills the derived page and the halo
    # metadata page.
    "three_lanes": ((1, 0, 2), [(0, 1), (1, 0), (2, 2)]),
    "eight_lanes": ((0, 1, 2, 3, 0, 1, 2, 3), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (5, 1), (6, 2), (7, 3)]),
}


@pytest.mark.parametrize("scenario", list(_SLIDING_LANES_SCENARIOS))
def test_sliding_lanes_match_per_request(scenario, expect_error):
    """Gemma4's sliding attention shape (W1024, D256, 4Q:2KV heads per device, 256-token slabs per device) with several
    requests in one call: each lane is bit-identical to a one-request metadata-path call and matches torch."""
    prefix_chunks, step_chunks = _SLIDING_LANES_SCENARIOS[scenario]
    run_sliding_lanes_case(prefix_chunks, step_chunks, expect_error)


@pytest.mark.parametrize("scenario", ["lane0_deepest", "same_two_by_two"])
def test_sliding_lanes_cache_hit_and_trace_replay(scenario, expect_error):
    """A B-lane program reused with every lane's slot and prefix changed, eagerly on a program-cache hit and by trace
    replay: each lane stays bit-identical to its single-request call."""
    prefix_chunks, step_chunks = _SLIDING_LANES_SCENARIOS[scenario]
    run_sliding_lanes_case(prefix_chunks, step_chunks, expect_error, replay=True)


# A 3x2 SDPA grid (6 cores) instead of ~100: each lane has 4 heads x 2 Q chunks = 8 units, so B=4 spreads 32 units
# as 6,6,5,5,5,5 and B=8 spreads 64 as 11,11,11,11,10,10. Units per core > 1 and not a multiple of 8, so a core's
# units cross a lane boundary (3 of 6 cores at B=4, all 6 at B=8) and it switches slot, prefix and Q mapping
# mid-run. The grid sits at the origin, clear of the CCL column (the device grid's last column) the halo workers use.
@pytest.mark.parametrize("replay", [False, True], ids=["eager", "replay"])
@pytest.mark.parametrize("scenario", ["lane0_deepest", "eight_lanes"])
def test_sliding_lanes_cross_lane_cores(scenario, replay, expect_error):
    """Lanes on a grid too small for one unit per core: each lane is still bit-identical to its single-request call,
    eagerly and, with replay, with the lanes reordered on a program-cache hit and in trace replay."""
    prefix_chunks, step_chunks = _SLIDING_LANES_SCENARIOS[scenario]
    run_sliding_lanes_case(prefix_chunks, step_chunks, expect_error, replay=replay, compute_grid=(3, 2))


@pytest.mark.parametrize("replay", [False, True], ids=["eager", "replay"])
def test_sliding_lanes_mixed_q_rotation(replay, expect_error):
    """Lanes whose prefixes are not whole chunks, mixed in one call: an aligned prefix, a slab-rotated one (a
    multiple of the 256-token slab, so every device's Q stays in one slab) and two tile-rotated ones (a multiple of
    32 only, so each device's Q wraps into a second slab and its halo needs the second slot). Each lane is
    bit-identical to its single-request call and matches torch, eagerly and with the lanes reordered on a
    program-cache hit and in trace replay."""
    run_sliding_lanes_case(
        prefix_chunks=(0, 0, 0, 0),
        # Prefixes 2 * chunk_global, chunk_global + 512, 32 and chunk_global + 352 (slab 1 plus three tiles); slot 0's
        # stream is the deepest, so the others end mid-group with garbage past their filled length.
        step_chunks=[(0, 2), (1, 1), (2, 0), (3, 1)],
        expect_error=expect_error,
        prefix_offsets=[0, 512, 32, 256 + 96],
        halo_slots=2,
        replay=replay,
    )
