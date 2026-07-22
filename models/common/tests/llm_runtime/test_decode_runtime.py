# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

import models.common.llm_runtime.decode as decode_module
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.llm_runtime.decode import (
    DecodeDeviceInputs,
    DecodePersistentInputs,
    DecodeProgramSignature,
    DecodeRuntime,
    DecodeTraceSignature,
    InvocationResult,
)
from models.common.llm_runtime.output_reader import OutputReader, PendingRead
from models.common.sampling import SamplingParams


class FakeMesh:
    shape = (1, 1)


class FakeSampling:
    config = SimpleNamespace(allow_force_argmax=True, max_batch_size=2)

    def decode_forward(self, logits, **kwargs):
        return logits, None


class FakeRope:
    def get_rot_idxs(self, positions, *, on_host):
        assert on_host
        return ("rotary", positions.clone())

    def get_rot_mats(self, rotary_indices):
        return ("cos", "sin")


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_batch_size=2)
        self.sampling = FakeSampling()
        self.rope_setup = FakeRope()
        self.vocab_size = 8
        self.num_devices = 1

    def iter_executor_named_modules(self):
        return iter(())

    def increment_positions(self, positions, rotary_indices):
        return None


def make_runtime(*, sampling=True, force_greedy_top_k=False):
    mesh = FakeMesh()
    model = FakeModel()
    layout = SimpleNamespace(raw_capacity_width=8, decode_width=8, block_size=32)
    return DecodeRuntime(
        model,
        mesh,
        OutputReader(mesh),
        lane_capacity=2,
        page_table_layout=layout,
        device_sampling_enabled=sampling,
        force_greedy_top_k=force_greedy_top_k,
    )


def greedy_sampling():
    return SamplingParams(temperature=[0.0, 0.0], top_k=[1, 1], top_p=[1.0, 1.0])


def test_sampling_values_keep_tile_padded_device_contract_for_partial_lane():
    values = decode_module._formatted_sampling_values(
        SamplingParams(temperature=1.0, top_k=32, top_p=0.08),
        2,
    )

    assert tuple(len(field) for field in values[:3]) == (32, 32, 32)
    assert values[0][0] == 32
    assert values[1][0] == pytest.approx(0.08)
    assert values[2][0] == 1.0
    assert (values[0][-1], values[1][-1], values[2][-1]) == (1, 0.0, 1.0)


def test_sampling_values_accept_vector_tensor_fields_for_full_lane():
    values = decode_module._formatted_sampling_values(
        SamplingParams(
            temperature=torch.ones(32),
            top_k=torch.full((32,), 32, dtype=torch.int32),
            top_p=torch.full((32,), 0.08),
        ),
        32,
    )

    assert tuple(len(field) for field in values[:3]) == (32, 32, 32)
    assert values[0] == (32,) * 32
    assert values[1] == pytest.approx((0.08,) * 32)
    assert values[2] == (1.0,) * 32


def prepare(runtime, *, positions=(0, -1), page_table=None, sampling_params=None, reset=False):
    if page_table is None:
        page_table = torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.int32)
    return runtime.prepare(
        torch.tensor([11, 0]),
        torch.tensor(positions),
        page_table,
        sampling_params,
        reset_batch=reset,
    )


def test_signatures_expose_ordered_material_and_separate_types():
    runtime = make_runtime()
    prepared = prepare(runtime, sampling_params=greedy_sampling())

    program = runtime.program_signature(prepared)
    trace = runtime.trace_signature(prepared)

    assert isinstance(program, DecodeProgramSignature)
    assert isinstance(trace, DecodeTraceSignature)
    assert program.key_material() == (
        ("operation", "decode"),
        ("batch_size", 2),
        ("page_table_width", 8),
        ("sampling_path", "argmax"),
        ("device_feedback", True),
    )
    assert trace.key_material() == program.key_material()
    assert runtime.program_signature(prepare(runtime)).sampling_path == "logits"


def test_configured_topk_policy_is_not_collapsed_to_argmax_by_greedy_temperature():
    runtime = make_runtime(force_greedy_top_k=True)
    sampling = SamplingParams(temperature=[0.0, 0.0], top_k=[32, 32], top_p=[0.08, 0.08])

    prepared = prepare(runtime, sampling_params=sampling)

    assert prepared.sampling_path == "topk"


def test_unconfigured_topk_values_use_argmax_for_greedy_temperature():
    runtime = make_runtime()
    sampling = SamplingParams(temperature=[0.0, 0.0], top_k=[32, 32], top_p=[0.08, 0.08])

    assert prepare(runtime, sampling_params=sampling).sampling_path == "argmax"


def test_sampling_values_are_formatted_once_during_prepare(monkeypatch):
    runtime = make_runtime(force_greedy_top_k=True)
    calls = []
    formatter = decode_module._formatted_sampling_values
    monkeypatch.setattr(
        decode_module,
        "_formatted_sampling_values",
        lambda *args: calls.append(args) or formatter(*args),
    )
    prepared = prepare(runtime, sampling_params=greedy_sampling())
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: "mapper")
    monkeypatch.setattr(ttnn, "from_torch", lambda value, **kwargs: value)

    runtime._make_host_kpt(prepared)
    runtime._sampling_path(prepared.sampling_values)

    assert len(calls) == 1


def test_normalization_preserves_feedback_lookahead_and_inactive_convention():
    runtime = make_runtime()
    page_table = torch.tensor([[10, 11, 99], [20, 21, 98]], dtype=torch.int64)

    prepared = prepare(
        runtime,
        positions=(31, -1),
        page_table=page_table,
        sampling_params=greedy_sampling(),
    )

    assert prepared.page_table.dtype == torch.int32
    assert prepared.page_table.shape == (2, 8)
    assert prepared.page_table[0].tolist() == [10, 11, 0, 0, 0, 0, 0, 0]
    assert prepared.page_table[1].tolist() == [0, 0, 0, 0, 0, 0, 0, 0]


def test_normalization_reuses_equal_source_with_same_copy_counts():
    runtime = make_runtime()
    first = prepare(runtime, positions=(0, -1))
    second = prepare(runtime, positions=(1, -1))

    assert second.page_table is first.page_table


def test_normalization_cache_detects_in_place_source_mutation():
    runtime = make_runtime()
    source = torch.tensor([[3, 4], [6, 7]], dtype=torch.int32)
    first = prepare(runtime, positions=(0, -1), page_table=source)
    source[0, 0] = 9
    second = prepare(runtime, positions=(0, -1), page_table=source)

    assert second.page_table is not first.page_table
    assert second.page_table[0, 0].item() == 9


def test_normalization_cache_misses_when_copy_counts_or_feedback_change():
    runtime = make_runtime()
    source = torch.tensor([[3, 4], [6, 7]], dtype=torch.int32)
    one_block = prepare(runtime, positions=(0, -1), page_table=source)
    two_blocks = prepare(runtime, positions=(32, -1), page_table=source)
    no_feedback = prepare(runtime, positions=(31, -1), page_table=source)
    with_feedback = prepare(
        runtime,
        positions=(31, -1),
        page_table=source,
        sampling_params=greedy_sampling(),
    )

    assert two_blocks.page_table is not one_block.page_table
    assert no_feedback.page_table[0, 1].item() == 0
    assert with_feedback.page_table is not no_feedback.page_table
    assert with_feedback.page_table[0, 1].item() == 4


def test_fixed_capacity_and_page_table_capacity_are_validated(expect_error):
    runtime = make_runtime()
    with expect_error(ValueError, "must equal lane capacity"):
        runtime.prepare(torch.tensor([1]), torch.tensor([0]), torch.tensor([[1]]))
    with expect_error(ValueError, "batches must match"):
        runtime.prepare(
            torch.tensor([1, 2]),
            torch.tensor([0]),
            torch.tensor([[1], [2]]),
        )
    with expect_error(ValueError, "paged-KV capacity"):
        prepare(runtime, positions=(8 * 32, -1))
    with expect_error(ValueError, "too narrow"):
        prepare(
            runtime,
            positions=(64, -1),
            page_table=torch.tensor([[1, 2], [0, 0]], dtype=torch.int32),
        )


def test_preparation_tracks_first_used_page_change_reset_and_ignores_unused_tail():
    runtime = make_runtime()
    first = prepare(runtime, positions=(0, -1), reset=True)
    assert first.page_table_changed
    assert first.reset_batch

    runtime.note_submitted(first)
    same_semantics = prepare(
        runtime,
        positions=(0, -1),
        page_table=torch.tensor([[3, 90, 91], [88, 87, 86]], dtype=torch.int32),
    )
    assert not same_semantics.page_table_changed
    assert not same_semantics.reset_batch

    changed = prepare(
        runtime,
        positions=(0, -1),
        page_table=torch.tensor([[4, 90, 91], [88, 87, 86]], dtype=torch.int32),
    )
    assert changed.page_table_changed


def test_submission_state_tracks_last_table_despite_stale_prepare_change_hint():
    runtime = make_runtime()
    baseline = prepare(
        runtime,
        positions=(0, -1),
        page_table=torch.tensor([[3], [0]], dtype=torch.int32),
    )
    runtime.note_submitted(baseline)

    changed = prepare(
        runtime,
        positions=(0, -1),
        page_table=torch.tensor([[4], [0]], dtype=torch.int32),
    )
    back_to_baseline = prepare(
        runtime,
        positions=(0, -1),
        page_table=torch.tensor([[3], [0]], dtype=torch.int32),
    )
    assert changed.page_table_changed
    assert not back_to_baseline.page_table_changed

    runtime.note_submitted(changed)
    runtime.note_submitted(back_to_baseline)

    assert torch.equal(runtime.previous_page_table, back_to_baseline.page_table)
    assert prepare(
        runtime,
        positions=(0, -1),
        page_table=torch.tensor([[4], [0]], dtype=torch.int32),
    ).page_table_changed


def test_capture_plan_describes_full_step_refresh_and_typed_persistent_inputs(monkeypatch):
    runtime = make_runtime()
    prepared = prepare(runtime, sampling_params=greedy_sampling())
    device = DecodeDeviceInputs("tokens", "positions", "rotary", "page_table")
    monkeypatch.setattr(runtime, "_prepare_inputs_host", lambda request: "host")
    monkeypatch.setattr(runtime, "_stage_inputs_and_kpt", lambda host, request: (device, "kpt"))
    monkeypatch.setattr(runtime, "_run_body", lambda *args, **kwargs: "captured")

    plan = runtime.capture_plan(prepared)
    persistent = plan.prepare_inputs()

    assert persistent.device_inputs is device
    assert persistent.kpt == "kpt"
    assert persistent.kpt_signature == [prepared.sampling_values[:3]]
    assert plan.capture(persistent) == "captured"
    assert plan.refresh_policy.every_replay == ("sampling",)
    assert plan.refresh_policy.full_on_batch_reset
    assert plan.refresh_policy.full_on_graph_switch
    assert plan.refresh_policy.full_without_device_feedback
    assert plan.refresh_policy.refresh_page_table_on_change


def test_trace_refresh_skips_unchanged_sampling_values(monkeypatch):
    runtime = make_runtime(force_greedy_top_k=True)
    prepared = prepare(runtime, sampling_params=greedy_sampling())
    persistent = DecodePersistentInputs(
        device_inputs=DecodeDeviceInputs("tokens", "positions", "rotary", "page_table"),
        kpt="kpt",
        kpt_signature=[prepared.sampling_values[:3]],
    )
    monkeypatch.setattr(runtime, "_refresh_kpt", lambda *args: pytest.fail("unchanged KPT was refreshed"))

    runtime.refresh_trace(
        persistent,
        prepared,
        SimpleNamespace(full=False, page_table=False),
    )


def test_eager_invoke_returns_owned_result_and_advances_submission_state(monkeypatch):
    runtime = make_runtime()
    prepared = prepare(runtime)
    device = DecodeDeviceInputs("tokens", "positions", "rotary", "page_table")
    calls = []
    monkeypatch.setattr(runtime, "_prepare_inputs_host", lambda request: "host")
    monkeypatch.setattr(runtime, "_stage_inputs_and_kpt", lambda host, request: (device, None))
    monkeypatch.setattr(runtime, "_validation_context", contextlib.nullcontext)
    monkeypatch.setattr(
        runtime,
        "_run_body",
        lambda inputs, sampling, kpt, *, device_feedback: calls.append(device_feedback) or ("raw", None),
    )

    result = runtime.invoke(prepared)

    assert isinstance(result, InvocationResult)
    assert result.value == ("raw", None)
    assert result.owned == (("raw", None), (device, None))
    assert not result.is_tokens
    assert calls == [False]
    assert torch.equal(runtime.previous_page_table, prepared.page_table)


def test_blocking_consume_normalizes_logits_and_releases_owned_values(monkeypatch):
    runtime = make_runtime()
    host_logits = torch.arange(16, dtype=torch.float32).reshape(1, 1, 2, 8)
    released = []
    monkeypatch.setattr(runtime.output_reader, "read", lambda value, *, blocking: (host_logits, "probs"))
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda value: released.append(value) or [])
    result = InvocationResult(value="raw", owned="owned", is_tokens=False)

    logits, log_probs = runtime.consume(result)

    assert logits.shape == (2, 1, 8)
    assert log_probs == "probs"
    assert released == ["owned"]
    assert runtime.external_lease_count == 0


def test_raw_blocking_and_async_leases_release_exact_records(monkeypatch):
    runtime = make_runtime()
    deallocated = []
    monkeypatch.setattr(
        decode_module,
        "best_effort_deallocate_owned_tensors",
        lambda values, completed: deallocated.append(values) or [],
    )

    first = InvocationResult(value=object(), owned="first-owned", is_tokens=False)
    assert runtime.consume(first, read_from_device=False) is first.value
    monkeypatch.setattr(runtime.output_reader, "read", lambda value, *, blocking: "first-host")
    assert runtime.read_decode_output(first.value) == "first-host"
    assert runtime.external_lease_count == 0

    second = InvocationResult(value=object(), owned="second-owned", is_tokens=True)
    runtime.consume(second, read_from_device=False)
    host_tokens = torch.tensor([[[[7], [8]]]], dtype=torch.int32)
    pending = PendingRead(value=(host_tokens, None), events=("event",), sequence=4, _owner=object())
    monkeypatch.setattr(runtime.output_reader, "submit", lambda value: pending)
    monkeypatch.setattr(runtime.output_reader, "complete", lambda value: pending.value)
    monkeypatch.setattr(runtime.output_reader, "complete", lambda value: pending.value)

    host, events = runtime.read_decode_output(second.value, async_read=True)
    assert host is pending.value
    assert events == ["event"]
    tokens, log_probs = runtime.process_decode_output_host(host, is_tokens=True)
    assert tokens.tolist() == [7, 8]
    assert tokens.dtype == torch.int64
    assert log_probs is None
    assert runtime.external_lease_count == 0
    assert deallocated == [
        (first.value, "first-owned"),
        (second.value, "second-owned"),
    ]


def test_async_trace_lease_never_releases_borrowed_trace_output(monkeypatch):
    runtime = make_runtime()
    deallocated = []
    raw = object()
    host_tokens = torch.tensor([[[[7], [8]]]], dtype=torch.int32)
    pending = PendingRead(value=(host_tokens, None), events=("event",), sequence=4, _owner=object())
    monkeypatch.setattr(
        decode_module,
        "best_effort_deallocate_owned_tensors",
        lambda values, completed: deallocated.append(values) or [],
    )
    monkeypatch.setattr(runtime.output_reader, "submit", lambda value: pending)
    monkeypatch.setattr(runtime.output_reader, "complete", lambda value: pending.value)
    monkeypatch.setattr(runtime.output_reader, "complete", lambda value: pending.value)

    result = InvocationResult(value=raw, owned=None, is_tokens=True)
    assert runtime.consume(result, read_from_device=False) is raw
    assert runtime.external_lease_count == 0
    host, events = runtime.read_decode_output(raw, async_read=True)
    assert events == ["event"]
    tokens, log_probs = runtime.process_decode_output_host(host, is_tokens=True)

    assert tokens.tolist() == [7, 8]
    assert log_probs is None
    assert runtime.external_lease_count == 0
    assert deallocated == []


def test_failed_transient_release_blocks_use_and_cleanup_retries(monkeypatch, expect_error):
    runtime = make_runtime()

    class FakeTensor:
        pass

    tensor = FakeTensor()
    attempts = []
    monkeypatch.setattr(decode_module.ttnn, "Tensor", FakeTensor)

    def deallocate(value):
        attempts.append(value)
        if len(attempts) == 1:
            raise RuntimeError("release failed")

    monkeypatch.setattr(decode_module.ttnn, "deallocate", deallocate)

    failures = runtime._release_or_retain_transient(tensor)
    assert [str(error) for error in failures] == ["release failed"]
    assert runtime.transient_orphan_count == 1
    with expect_error(RuntimeError, "unreleased transient"):
        prepare(runtime)

    runtime.cleanup_transients()
    assert attempts == [tensor, tensor]
    assert runtime.transient_orphan_count == 0
    assert prepare(runtime).sampling_path == "logits"


def test_decode_runtime_is_plain_constructor_injected_orchestration():
    assert not hasattr(decode_module, "DecodeRuntimeConfig")
    assert not hasattr(DecodeRuntime, "from_config")
    assert not issubclass(DecodeRuntime, LightweightModule)
    assert DecodeRuntime.__bases__ == (object,)
