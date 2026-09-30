# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent sampling outputs with an unrelated trace already resident.

Run with TT_METAL_TRACE_ALLOC_TRACKING=1 and TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0.
No model weights are required.
"""

from types import SimpleNamespace

import pytest
import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params
from models.common.tests.test_tt_sampling import (
    BATCH_SIZE,
    build_hot_logits,
    extract_tokens_all_devices,
    make_sampling_args,
    make_sharded_logits,
)
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.generator import Generator

pytestmark = pytest.mark.skipif(
    not trace_allocation_tracker.TRACE_ALLOC_TRACKING,
    reason="requires TT_METAL_TRACE_ALLOC_TRACKING=1 before importing ttnn",
)


MESH_CONFIGS = [
    pytest.param(1, {"trace_region_size": 24 * 1024 * 1024}, id="single"),
    pytest.param(
        (1, 8),
        {"trace_region_size": 24 * 1024 * 1024, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
        id="t3k",
    ),
]


def _make_sampler(mesh_device, *, allow_argmax=True, topk_logprobs=False):
    args = make_sampling_args(mesh_device)
    args.use_topk_logprobs = topk_logprobs
    if allow_argmax:
        args.model_config["SAMPLING_AG_CONFIG"] = {
            "allow_force_argmax": True,
            "num_links": 1,
            "topology": ttnn.Topology.Ring,
        }
    ccl = TT_CCL(mesh_device) if mesh_device.get_num_devices() > 1 else None
    sampler = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=ccl)
    params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    sampler.reset_sampling_params(format_sampling_params(params, BATCH_SIZE))
    logits = make_sharded_logits(build_hot_logits(args, hot_tokens=[101]), mesh_device, args)
    return sampler, logits


def _older_trace(mesh_device):
    # Called after sampling warmup, before any sampling capture. Its output
    # remains live, so subsequent sampling allocations must avoid its addresses.
    source = ttnn.from_torch(
        torch.ones(1, 1, 32, 32, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    warm = ttnn.neg(source)
    del warm
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = ttnn.neg(source)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    return trace_id, source, output


def _assert_tokens(output, expected=101):
    for tokens in extract_tokens_all_devices(output):
        assert tokens == [expected] * BATCH_SIZE


@pytest.mark.parametrize("mesh_device,device_params", MESH_CONFIGS, indirect=True)
@pytest.mark.parametrize("argmax_first", [False, True])
@pytest.mark.parametrize("provided_output", [False, True])
def test_sampling_outputs_survive_older_trace_and_reset(mesh_device, device_params, argmax_first, provided_output):
    sampler, logits = _make_sampler(mesh_device)
    alternate_logits = make_sharded_logits(
        build_hot_logits(make_sampling_args(mesh_device), hot_tokens=[202]),
        mesh_device,
        make_sampling_args(mesh_device),
    )
    supplied = (
        ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, 1, BATCH_SIZE)), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device
        )
        if provided_output
        else None
    )
    sampler.precompile(logits, tt_out_tok=supplied, all_configs=True)
    retained = {}
    for force_argmax in (False, True):
        shape = (1, 1, 1, BATCH_SIZE) if provided_output or not force_argmax else (1, 1, BATCH_SIZE)
        source = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device)
        retained[force_argmax] = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device
        )
        ttnn.copy(source, retained[force_argmax])
        del source
    older_id, older_input, older_output = _older_trace(mesh_device)
    try:
        # Select keys directly: reset_sampling_params still releases traces on a
        # force-argmax transition until the independent #57704 change lands.
        keys = []
        for bucket in (1, 2):
            sampler.set_trace_bucket(bucket)
            for force_argmax in (argmax_first, not argmax_first):
                sampler.tt_sampling._force_argmax_sampling = force_argmax
                sampler.capture_trace(
                    alternate_logits if force_argmax else logits, tt_out_tok=supplied, skip_precompile=True
                )
                key, _ = sampler._trace_slot(False, False, force_argmax)
                keys.append(key)
        captured_ids = {key: sampler._trace_states[key]["id"] for key in keys}
        cache_entries = mesh_device.num_program_cache_entries()
        addresses = {}
        snapshots = []
        for key in keys * 2:
            output, log_probs = sampler._execute_trace(key)
            assert log_probs is None
            expected = 202 if key.force_argmax else 101
            _assert_tokens(output, expected)
            # Queue the host read before a later trace can reuse the output, but
            # defer inspecting the snapshot until all writers have finished.
            snapshots.append((expected, ttnn.from_device(output, blocking=False)))
            ttnn.copy(output, retained[key.force_argmax])
            expected_shape = (1, 1, 1, BATCH_SIZE) if provided_output or not key.force_argmax else (1, 1, BATCH_SIZE)
            assert tuple(output.shape) == expected_shape
            if provided_output:
                assert output is supplied
            addresses[key.force_argmax] = output.buffer_address()
            ttnn.execute_trace(mesh_device, older_id, cq_id=0, blocking=True)
            _assert_tokens(output, expected)
        assert {key: sampler._trace_states[key]["id"] for key in keys} == captured_ids
        for expected, snapshot in snapshots:
            _assert_tokens(snapshot, expected)
        _assert_tokens(retained[False], 101)
        _assert_tokens(retained[True], 202)

        sampler.reset_trace()
        for key in keys:
            sampler.set_trace_bucket(key.bucket)
            sampler.tt_sampling._force_argmax_sampling = key.force_argmax
            sampler.capture_trace(
                alternate_logits if key.force_argmax else logits, tt_out_tok=supplied, skip_precompile=True
            )
            output, _ = sampler._execute_trace(key)
            assert output.buffer_address() == addresses[key.force_argmax]
            ttnn.execute_trace(mesh_device, older_id, cq_id=0, blocking=True)
            _assert_tokens(output, 202 if key.force_argmax else 101)
        assert mesh_device.num_program_cache_entries() == cache_entries
    finally:
        sampler.reset_trace()
        ttnn.release_trace(mesh_device, older_id)
    assert older_input.is_allocated() and older_output.is_allocated()


@pytest.mark.parametrize("mesh_device,device_params", MESH_CONFIGS[:1], indirect=True)
@pytest.mark.parametrize("late_output", [False, True], ids=["capture-allocation", "caller-output"])
def test_sampling_capture_does_not_hide_unsafe_allocations(
    mesh_device, device_params, monkeypatch, expect_error, late_output
):
    sampler, logits = _make_sampler(mesh_device, allow_argmax=False)
    sampler.precompile(logits)
    if late_output:
        # Warm the caller-output form explicitly, including on the unpatched
        # baseline whose default warmup still uses the allocating form.
        compile_output = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, 1, BATCH_SIZE)), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device
        )
        sampler.precompile(logits, tt_out_tok=compile_output)
        del compile_output
    older_id, older_input, older_output = _older_trace(mesh_device)
    late = []

    def allocate_late():
        value = ttnn.allocate_tensor_on_device(
            ttnn.Shape((1, 1, 1, BATCH_SIZE)), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device
        )
        late.append(value)
        return value

    try:
        sampler.set_trace_bucket(1)
        if late_output:
            supplied = allocate_late()
        else:
            supplied = None
            run_sampling = sampler._run_sampling

            def run_with_survivor(*args, **kwargs):
                result = run_sampling(*args, **kwargs)
                allocate_late()
                return result

            monkeypatch.setattr(sampler, "_run_sampling", run_with_survivor)
        sampler.capture_trace(logits, tt_out_tok=supplied, skip_precompile=True)
        unsafe = trace_allocation_tracker.get_unsafe_tracked_ids(mesh_device, older_id)
        assert late[0].buffer_unique_id() in unsafe
        with expect_error(RuntimeError, "still alive before trace replay"):
            ttnn.execute_trace(mesh_device, older_id, cq_id=0, blocking=True)
    finally:
        sampler.reset_trace()
        ttnn.release_trace(mesh_device, older_id)
    assert older_input.is_allocated() and older_output.is_allocated()


@pytest.mark.parametrize("mesh_device,device_params", MESH_CONFIGS[1:], indirect=True)
@pytest.mark.parametrize("topk_logprobs", [False, True])
def test_sampling_logprobs_and_penalties_share_early_outputs(mesh_device, device_params, topk_logprobs):
    sampler, logits = _make_sampler(mesh_device, allow_argmax=False, topk_logprobs=topk_logprobs)
    sampler.precompile(logits, all_configs=True)
    # precompile intentionally does not count dummy tokens. Compile the penalty
    # bookkeeping separately while this synthetic sampler has no request state.
    sampler.tt_penalties.update_output_tokens(sampler._trace_token_outputs[False])
    sampler.tt_penalties.reset_output_tokens()
    older_id, older_input, older_output = _older_trace(mesh_device)
    try:
        outputs = []
        for penalties_on, log_probs_on in ((False, False), (False, True), (True, True), (True, False)):
            sampler._penalties_active = penalties_on
            sampler._log_probs_active = log_probs_on
            sampler.tt_sampling.log_probs_calculator.set_log_probs_mode(log_probs_on, num_logprobs=0)
            sampler.capture_trace(logits, skip_precompile=True)
            key, _ = sampler._trace_slot(penalties_on, log_probs_on, False)
            outputs.append((key, sampler._trace_states[key]["output"]))
        assert len({output[0].buffer_address() for _, output in outputs}) == 1
        cache_entries = mesh_device.num_program_cache_entries()
        for key, _ in outputs * 2:
            output, log_probs = sampler._execute_trace(key)
            _assert_tokens(output)
            if key.log_probs_on:
                lp = log_probs.topk_logprobs if topk_logprobs else log_probs
                before = [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(lp)]
                ttnn.execute_trace(mesh_device, older_id, cq_id=0, blocking=True)
                after = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(lp)]
                for lhs, rhs in zip(before, after):
                    torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
            else:
                assert log_probs is None
                ttnn.execute_trace(mesh_device, older_id, cq_id=0, blocking=True)
        assert mesh_device.num_program_cache_entries() == cache_entries
    finally:
        sampler.reset_trace()
        ttnn.release_trace(mesh_device, older_id)
    assert older_input.is_allocated() and older_output.is_allocated()


@pytest.mark.parametrize("mesh_device,device_params", MESH_CONFIGS, indirect=True)
def test_prefill_sampling_capture_uses_prepared_output(mesh_device, device_params):
    sampler, hot_logits = _make_sampler(mesh_device, allow_argmax=False)
    args = make_sampling_args(mesh_device)
    # Exercise the generator's real preparation/capture boundary. A device clone
    # stands in for norm + lm_head so this needs no model weights.
    model = SimpleNamespace(sampling=sampler, _apply_norm_and_lm_head=ttnn.clone)
    model_args = SimpleNamespace(mesh_device=mesh_device, dim=args.padded_vocab_size)
    generator = Generator([model], [model_args], mesh_device)
    prepared = generator._prepare_trace_prefill_sampling(0, BATCH_SIZE)
    ttnn.copy(hot_logits, prepared["input"])
    older_id, older_input, older_output = _older_trace(mesh_device)
    trace_id = None
    try:
        cache_entries = mesh_device.num_program_cache_entries()
        trace_id, (output, log_probs), _ = generator._record_trace_prefill_sampling(prepared)
        assert output.buffer_unique_id() == prepared["token_output"].buffer_unique_id()
        assert log_probs is None
        for _ in range(2):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            _assert_tokens(output)
            ttnn.execute_trace(mesh_device, older_id, cq_id=0, blocking=True)
            _assert_tokens(output)
        assert mesh_device.num_program_cache_entries() == cache_entries
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)
        ttnn.release_trace(mesh_device, older_id)
    assert older_input.is_allocated() and older_output.is_allocated()
