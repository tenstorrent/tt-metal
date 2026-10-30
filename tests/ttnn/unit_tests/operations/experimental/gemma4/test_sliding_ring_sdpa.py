# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.gemma4.sliding_ring_sdpa matches the shared ring joint SDPA bit for bit on Gemma4's sliding configs.

Each test reruns a shared-op sliding test from tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py twice in one
process, once as written and once with its sliding calls routed to the fork, and compares every tensor the test
reads back. The shared tests' own accuracy and determinism checks run in both passes. Trace replay is covered by
rerunning the whole capture and replay sequence.
"""

import functools
import random
from contextlib import contextmanager
from unittest import mock

import pytest
import torch

import ttnn
from tests.nightly.blackhole.sdpa import test_ring_joint_sdpa as shared

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
