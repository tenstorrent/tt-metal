# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import models.common.llm_runtime.decode as decode_module
import ttnn
from models.common.llm_runtime.config import PageTableLayout
from models.common.llm_runtime.decode import (
    DecodeDeviceInputs,
    DecodePersistentInputs,
    DecodeProgramSignature,
    DecodeRuntime,
    DecodeRuntimeConfig,
    DecodeTraceSignature,
    InvocationResult,
)
from models.common.llm_runtime.output_reader import OutputReader, PendingRead
from models.common.sampling import SamplingParams


class FakeMesh:
    shape = (1, 1)


class FakeSampling:
    config = SimpleNamespace(allow_force_argmax=True, max_batch_size=2)

    def decode_forward(self, logits, *, k=None, p=None, temp=None, tt_out_tok=None):
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
    config = DecodeRuntimeConfig.resolve(
        model=model,
        output_reader=OutputReader(mesh),
        lane_capacity=2,
        page_table_layout=page_table_layout(),
        device_sampling_enabled=sampling,
        force_greedy_top_k=force_greedy_top_k,
    )
    return DecodeRuntime(config)


def page_table_layout(*, raw_width=8, block_size=32):
    return PageTableLayout(
        block_size=block_size,
        raw_capacity_width=raw_width,
        prefill_width=((raw_width + 7) // 8) * 8,
        decode_width=((raw_width + 7) // 8) * 8,
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
        sampling_params=sampling_params,
        reset_batch=reset,
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        (
            DecodeRuntime.prepare,
            (
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("tokens", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("start_pos", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("page_table", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("sampling_params", inspect.Parameter.KEYWORD_ONLY, None),
                ("reset_batch", inspect.Parameter.KEYWORD_ONLY, False),
            ),
        ),
        (
            DecodeRuntime.read_decode_output,
            (
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("tt_out", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("async_read", inspect.Parameter.KEYWORD_ONLY, False),
            ),
        ),
        (
            DecodeRuntime.process_decode_output_host,
            (
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("tt_out", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("is_tokens", inspect.Parameter.KEYWORD_ONLY, False),
            ),
        ),
    ),
)
def test_runtime_api_signatures_are_exact(method, expected):
    parameters = inspect.signature(method).parameters.values()

    assert tuple((parameter.name, parameter.kind, parameter.default) for parameter in parameters) == expected


def test_runtime_api_preserves_positional_prefixes_and_rejects_extra_fields(expect_error):
    cases = (
        (DecodeRuntime.prepare, ("tokens", "start_pos", "page_table"), "sampling_params"),
        (DecodeRuntime.read_decode_output, ("tt_out",), "async_read"),
        (DecodeRuntime.process_decode_output_host, ("tt_out",), "is_tokens"),
    )

    for method, positional_prefix, keyword_name in cases:
        signature = inspect.signature(method)
        signature.bind(None, *positional_prefix, **{keyword_name: True})
        with expect_error(TypeError, "too many positional arguments"):
            signature.bind(None, *positional_prefix, True)
        with expect_error(TypeError, "unexpected keyword argument 'unknown'"):
            signature.bind(None, *positional_prefix, unknown=True)

    assert inspect.get_annotations(DecodeRuntime.prepare, eval_str=True)["sampling_params"] is Any


def test_config_resolves_canonical_static_capabilities_and_is_frozen(expect_error):
    mesh = FakeMesh()
    model = FakeModel()
    config = DecodeRuntimeConfig.resolve(
        model=model,
        output_reader=OutputReader(mesh),
        lane_capacity=2,
        page_table_layout=page_table_layout(),
        device_sampling_enabled=True,
    )

    assert config.cluster_shape == (1, 1)
    assert config.num_devices == 1
    assert config.vocab_size == 8
    assert config.allow_force_argmax
    assert config.position_feedback_capable
    with expect_error(dataclasses.FrozenInstanceError, "cannot assign to field"):
        config.lane_capacity = 1
    with expect_error(TypeError, "DecodeRuntimeConfig"):
        DecodeRuntime(config=None)


def test_config_rejects_inconsistent_collaborators_and_dimensions(expect_error):
    mesh = FakeMesh()
    model = FakeModel()
    model.config.mesh_device = mesh
    with expect_error(ValueError, "same mesh_device"):
        DecodeRuntimeConfig.resolve(
            model=model,
            output_reader=OutputReader(FakeMesh()),
            lane_capacity=2,
            page_table_layout=page_table_layout(),
            device_sampling_enabled=True,
        )
    with expect_error(ValueError, "positive integer"):
        DecodeRuntimeConfig.resolve(
            model=model,
            output_reader=OutputReader(mesh),
            lane_capacity=True,
            page_table_layout=page_table_layout(),
            device_sampling_enabled=True,
        )
    with expect_error(TypeError, "PageTableLayout"):
        DecodeRuntimeConfig.resolve(
            model=model,
            output_reader=OutputReader(mesh),
            lane_capacity=2,
            page_table_layout=SimpleNamespace(raw_capacity_width=8, decode_width=8, block_size=32),
            device_sampling_enabled=True,
        )


def test_layout_replacement_is_immutable_and_bounded(expect_error):
    runtime = make_runtime()
    original = runtime.config
    replacement = page_table_layout(raw_width=4)

    runtime.configure_page_table_layout(replacement)

    assert runtime.config is not original
    assert original.page_table_layout.raw_capacity_width == 8
    assert runtime.config.page_table_layout is replacement
    assert runtime.config.page_table_layout_ceiling is original.page_table_layout
    with expect_error(ValueError, "block_size"):
        runtime.configure_page_table_layout(page_table_layout(raw_width=4, block_size=16))
    with expect_error(ValueError, "ceiling"):
        runtime.configure_page_table_layout(page_table_layout(raw_width=9))
    with expect_error(ValueError, "decode width"):
        runtime.configure_page_table_layout(PageTableLayout(32, 4, 8, 1024))


def test_sampling_admission_follows_resolved_configuration(expect_error):
    runtime = make_runtime(sampling=False)
    with expect_error(ValueError, "device sampling is disabled"):
        prepare(runtime, sampling_params=greedy_sampling())


def test_feedback_and_sampling_path_follow_resolved_capabilities():
    argmax_runtime = make_runtime()
    argmax_prepared = prepare(argmax_runtime, sampling_params=greedy_sampling())
    assert argmax_prepared.device_feedback
    assert argmax_prepared.sampling_path == "argmax"

    model = FakeModel()
    model.increment_positions = None
    mesh = FakeMesh()
    no_feedback = DecodeRuntime(
        DecodeRuntimeConfig.resolve(
            model=model,
            output_reader=OutputReader(mesh),
            lane_capacity=2,
            page_table_layout=page_table_layout(),
            device_sampling_enabled=True,
            force_greedy_top_k=True,
        )
    )
    prepared = prepare(no_feedback, sampling_params=greedy_sampling())

    assert not prepared.device_feedback
    assert prepared.sampling_path == "topk"


def test_single_and_multi_device_logits_conversion(monkeypatch):
    single = make_runtime()
    logits = torch.arange(16, dtype=torch.float32).reshape(1, 1, 2, 8)
    monkeypatch.setattr(ttnn, "to_torch", lambda value: logits)
    converted, _ = single._normalize_host_output("single-device", is_tokens=False)
    assert converted.shape == (2, 1, 8)

    mesh = SimpleNamespace(shape=(1, 2))
    model = FakeModel()
    model.num_devices = 2
    multi = DecodeRuntime(
        DecodeRuntimeConfig.resolve(
            model=model,
            output_reader=OutputReader(mesh),
            lane_capacity=2,
            page_table_layout=page_table_layout(),
            device_sampling_enabled=True,
        )
    )
    calls = []
    monkeypatch.setattr(
        decode_module,
        "_concat_host_output",
        lambda value, shape: calls.append((value, shape)) or logits,
    )
    converted, _ = multi._normalize_host_output("multi-device", is_tokens=False)
    assert converted.shape == (2, 1, 8)
    assert calls == [("multi-device", (1, 2))]


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

    def formatter_spy(
        sampling_params: SamplingParams,
        batch_size: int,
    ) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], bool]:
        calls.append((sampling_params, batch_size))
        return formatter(sampling_params, batch_size)

    monkeypatch.setattr(
        decode_module,
        "_formatted_sampling_values",
        formatter_spy,
    )
    prepared = prepare(runtime, sampling_params=greedy_sampling())
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: "mapper")
    monkeypatch.setattr(ttnn, "from_torch", lambda value, **kwargs: value)

    runtime._make_host_kpt(prepared)
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

    assert not prepare(runtime, page_table=back_to_baseline.page_table).page_table_changed
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

    def run_body(inputs, sampling_params, kpt, *, device_feedback):
        return "captured"

    monkeypatch.setattr(runtime, "_run_body", run_body)

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

    def fail_refresh_kpt(device_kpt, prepared):
        pytest.fail("unchanged KPT was refreshed")

    monkeypatch.setattr(runtime, "_refresh_kpt", fail_refresh_kpt)

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
    assert not prepare(runtime, page_table=prepared.page_table).page_table_changed


def test_blocking_consume_normalizes_logits_and_releases_owned_values(monkeypatch):
    runtime = make_runtime()
    host_logits = torch.arange(16, dtype=torch.float32).reshape(1, 1, 2, 8)
    released = []
    monkeypatch.setattr(runtime.config.output_reader, "read", lambda value, *, blocking: (host_logits, "probs"))
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda value: released.append(value) or [])
    result = InvocationResult(value="raw", owned="owned", is_tokens=False)

    logits, log_probs = runtime.consume(result)

    assert logits.shape == (2, 1, 8)
    assert log_probs == "probs"
    assert released == ["owned"]


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
    monkeypatch.setattr(runtime.config.output_reader, "read", lambda value, *, blocking: "first-host")
    assert runtime.read_decode_output(first.value) == "first-host"

    second = InvocationResult(value=object(), owned="second-owned", is_tokens=True)
    runtime.consume(second, read_from_device=False)
    host_tokens = torch.tensor([[[[7], [8]]]], dtype=torch.int32)
    pending = PendingRead(value=(host_tokens, None), events=("event",), sequence=4, _owner=object())
    monkeypatch.setattr(runtime.config.output_reader, "submit", lambda value: pending)
    monkeypatch.setattr(runtime.config.output_reader, "complete", lambda value: pending.value)

    host, events = runtime.read_decode_output(second.value, async_read=True)
    assert host is pending.value
    assert events == ["event"]
    tokens, log_probs = runtime.process_decode_output_host(host, is_tokens=True)
    assert tokens.tolist() == [7, 8]
    assert tokens.dtype == torch.int64
    assert log_probs is None
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
    monkeypatch.setattr(runtime.config.output_reader, "submit", lambda value: pending)
    monkeypatch.setattr(runtime.config.output_reader, "complete", lambda value: pending.value)

    result = InvocationResult(value=raw, owned=None, is_tokens=True)
    assert runtime.consume(result, read_from_device=False) is raw
    host, events = runtime.read_decode_output(raw, async_read=True)
    assert events == ["event"]
    tokens, log_probs = runtime.process_decode_output_host(host, is_tokens=True)

    assert tokens.tolist() == [7, 8]
    assert log_probs is None
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
