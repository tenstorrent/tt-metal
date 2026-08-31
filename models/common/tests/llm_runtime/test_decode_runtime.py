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
from models.common.modules.sampling.params import PreparedSamplingParams
from models.common.sampling.sampling_params import SamplingParams


class FakeMesh:
    shape = (1, 1)


class FakeSampling:
    def __init__(self, seed_buffer=None):
        self.config = SimpleNamespace(allow_force_argmax=True, max_batch_size=2, max_top_k=32, seeds=seed_buffer)

    def decode_forward(
        self,
        logits,
        *,
        k=None,
        p=None,
        temp=None,
        seeds=None,
        tt_out_tok=None,
        enable_log_probs=False,
    ):
        return logits, None


class FakeRope:
    def get_rot_idxs(self, positions, *, on_host):
        assert on_host
        return ("rotary", positions.clone())

    def get_rot_mats(self, rotary_indices):
        return ("cos", "sin")


class FakeModel:
    def __init__(self, seed_buffer=None):
        self.config = SimpleNamespace(max_batch_size=2)
        self.sampling = FakeSampling(seed_buffer)
        self.rope_setup = FakeRope()
        self.vocab_size = 8
        self.num_devices = 1

    def iter_executor_named_modules(self):
        return iter(())

    def increment_positions(self, positions, rotary_indices):
        return None


class FakeLazySeedBuffer:
    def __init__(self):
        self.source = torch.arange(2, dtype=torch.int64)
        self._value = object()
        self.updates = []

    def update(self, source):
        self.source = source
        self.updates.append(source.clone())

    def get_device_buffer(self):
        return self._value


def make_runtime(*, sampling=True, force_greedy_top_k=False, seed_buffer=None):
    mesh = FakeMesh()
    model = FakeModel(seed_buffer)
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


def test_prepare_uses_resolved_sampler_capacity_and_neutral_inactive_rows():
    prepared = prepare(
        make_runtime(),
        positions=(0, -1),
        sampling_params=SamplingParams(temperature=1.0, top_k=32, top_p=0.08),
    ).prepared_sampling

    assert isinstance(prepared, PreparedSamplingParams)
    assert prepared.batch_size == 2
    assert prepared.active_rows == 1
    assert prepared.active_mask == (True, False)
    assert prepared.top_k == (32, 1)
    assert prepared.top_p == pytest.approx((0.08, 0.0))
    assert prepared.temperature == (1.0, 1.0)
    assert prepared.row_paths == ("topk", "inactive")


def test_prepare_accepts_vector_tensor_fields_for_full_lane():
    prepared = prepare(
        make_runtime(),
        positions=(0, 0),
        sampling_params=SamplingParams(
            temperature=torch.ones(2),
            top_k=torch.full((2,), 32, dtype=torch.int32),
            top_p=torch.full((2,), 0.08),
        ),
    ).prepared_sampling

    assert isinstance(prepared, PreparedSamplingParams)
    assert prepared.active_mask == (True, True)
    assert prepared.top_k == (32, 32)
    assert prepared.top_p == pytest.approx((0.08, 0.08))
    assert prepared.temperature == (1.0, 1.0)


def prepare(
    runtime,
    *,
    positions=(0, -1),
    page_table=None,
    sampling_params=None,
    prompt_tokens=None,
    output_tokens=None,
    slot_remap=None,
    reset=False,
):
    if page_table is None:
        page_table = torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.int32)
    return runtime.prepare(
        torch.tensor([11, 0]),
        torch.tensor(positions),
        page_table,
        sampling_params=sampling_params,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        slot_remap=slot_remap,
        reset_batch=reset,
    )


def test_prepare_places_only_start_pos_active_rows_and_neutralizes_gap_sentinels():
    runtime = make_runtime()
    sampling = SamplingParams(
        temperature=[0.8, 0.8],
        top_k=[999, 7],
        top_p=[0.9, 0.8],
        seed=[-1, -1],
        enable_log_probs=[False, False],
        num_logprobs=[-2, -2],
    )

    prepared = prepare(
        runtime,
        positions=(-1, 4),
        sampling_params=sampling,
        prompt_tokens=torch.tensor([[10, -1], [20, 21]]),
        output_tokens=[[30, -1], [40, 41]],
        slot_remap=torch.tensor([1, 0]),
        reset=True,
    ).prepared_sampling

    assert prepared is not None
    assert prepared.active_mask == (False, True)
    assert prepared.row_paths == ("inactive", "topk")
    assert prepared.top_k == (1, 7)
    assert prepared.top_p == pytest.approx((0.0, 0.8))
    assert prepared.seeds == (None, None)
    assert prepared.enable_log_probs == (False, False)
    assert prepared.num_logprobs == (0, 0)
    assert prepared.prompt_tokens.tolist() == [[-1, -1], [20, 21]]
    assert prepared.output_tokens == [[-1, -1], [40, 41]]
    assert prepared.slot_remap.tolist() == [1, 0]


def seeded_sampling(seed0, seed1=None):
    return SamplingParams(
        temperature=[0.8, 0.8],
        top_k=[32, 32],
        top_p=[0.95, 0.95],
        seed=[seed0, seed1],
    )


def stochastic_sampling(seed=None):
    return SamplingParams(
        temperature=[0.8, 0.8],
        top_k=[32, 32],
        top_p=[0.95, 0.95],
        seed=seed,
    )


def make_four_slot_seed_runtime():
    seed_buffer = FakeLazySeedBuffer()
    seed_buffer.source = torch.arange(4, dtype=torch.int64)
    model = FakeModel(seed_buffer)
    model.config.max_batch_size = 4
    model.sampling.config.max_batch_size = 4
    runtime = DecodeRuntime(
        DecodeRuntimeConfig.resolve(
            model=model,
            output_reader=OutputReader(FakeMesh()),
            lane_capacity=4,
            page_table_layout=page_table_layout(),
            device_sampling_enabled=True,
        )
    )
    return runtime, seed_buffer


def four_slot_sampled_warmup(runtime):
    return runtime.prepare(
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        torch.zeros((4, 8), dtype=torch.int32),
        sampling_params=SamplingParams(
            temperature=torch.ones(4),
            top_k=torch.full((4,), 32, dtype=torch.int32),
            top_p=torch.full((4,), 0.08),
            seed=[11, 22, 33, 44],
        ),
    )


def _stub_compile_only_decode(runtime, monkeypatch, run_body):
    monkeypatch.setattr(runtime, "_prepare_inputs_host", lambda prepared: "host")
    monkeypatch.setattr(
        runtime,
        "_stage_inputs_and_kpt",
        lambda host, prepared: (DecodeDeviceInputs(None, None, None, None), None),
    )
    monkeypatch.setattr(runtime, "_run_body", run_body)


def test_compile_only_sampled_decode_temporarily_admits_and_resets_fallback_seed_slots(monkeypatch, expect_error):
    runtime, seed_buffer = make_four_slot_seed_runtime()
    defaults = seed_buffer.source.clone()
    prepared = four_slot_sampled_warmup(runtime)
    during_compile = []

    def run_body(*args, **kwargs):
        during_compile.append(runtime._seed_state.snapshot())
        assert not kwargs["count_tokens"]
        assert not kwargs["advance_seeds"]
        return object()

    _stub_compile_only_decode(runtime, monkeypatch, run_body)

    runtime.invoke(prepared, count_tokens=False)

    assert during_compile[0].active_slots == (0, 1, 2, 3)
    assert not during_compile[0].buffer_is_default
    reset = runtime._seed_state.snapshot()
    assert reset.active_slots == ()
    assert reset.buffer_is_default
    assert torch.equal(seed_buffer.source, defaults)
    with expect_error(RuntimeError, "reset_batch=True"):
        runtime._refresh_sampling_seeds(prepared)


def test_compile_only_sampled_decode_resets_fallback_seed_slots_after_failure(monkeypatch, expect_error):
    runtime, seed_buffer = make_four_slot_seed_runtime()
    defaults = seed_buffer.source.clone()
    prepared = four_slot_sampled_warmup(runtime)

    def run_body(*args, **kwargs):
        assert runtime._seed_state.snapshot().active_slots == (0, 1, 2, 3)
        raise RuntimeError("compile boom")

    _stub_compile_only_decode(runtime, monkeypatch, run_body)
    monkeypatch.setattr(runtime, "_release_or_retain_transient", lambda owned: [])

    with expect_error(RuntimeError, "compile boom"):
        runtime.invoke(prepared, count_tokens=False)

    reset = runtime._seed_state.snapshot()
    assert reset.active_slots == ()
    assert reset.buffer_is_default
    assert torch.equal(seed_buffer.source, defaults)


def test_runtime_seed_same_request_and_absolute_position_are_cardinality_independent():
    first_buffer = FakeLazySeedBuffer()
    first = make_runtime(seed_buffer=first_buffer)
    first_prepared = prepare(first, positions=(41, -1), sampling_params=seeded_sampling(1234), reset=True)
    first._refresh_sampling_seeds(first_prepared)

    remapped_buffer = FakeLazySeedBuffer()
    remapped = make_runtime(seed_buffer=remapped_buffer)
    remapped_prepared = prepare(
        remapped,
        positions=(-1, 41),
        sampling_params=seeded_sampling(None, 1234),
        reset=True,
    )
    remapped._refresh_sampling_seeds(remapped_prepared)

    assert int(first_buffer.updates[-1][0]) == int(remapped_buffer.updates[-1][1])
    assert first._seed_state.snapshot().active == (True, False)
    assert remapped._seed_state.snapshot().active == (False, True)


@pytest.mark.parametrize("seed", [1234, torch.tensor(1234)])
def test_runtime_scalar_seed_belongs_to_one_request_and_is_not_broadcast(seed):
    seed_buffer = FakeLazySeedBuffer()
    runtime = make_runtime(seed_buffer=seed_buffer)
    sampling_params = SamplingParams(
        temperature=[0.8, 0.8],
        top_k=[32, 32],
        top_p=[0.95, 0.95],
        seed=seed,
    )

    prepared = prepare(runtime, positions=(17, 17), sampling_params=sampling_params, reset=True)
    runtime._refresh_sampling_seeds(prepared)

    assert prepared.prepared_sampling.seeds == (1234, None)
    snapshot = runtime._seed_state.snapshot()
    assert snapshot.request_seeds == (1234, None)
    assert snapshot.active == (True, True)
    assert snapshot.current_device_seeds[0] is not None
    assert snapshot.current_device_seeds[1] is not None


@pytest.mark.parametrize("seed", [[111, 222], torch.tensor([111, 222])])
def test_runtime_vector_seed_remains_slot_indexed(seed):
    runtime = make_runtime(seed_buffer=FakeLazySeedBuffer())
    sampling_params = SamplingParams(
        temperature=[0.8, 0.8],
        top_k=[32, 32],
        top_p=[0.95, 0.95],
        seed=seed,
    )

    prepared = prepare(runtime, positions=(5, 5), sampling_params=sampling_params, reset=True)

    assert prepared.prepared_sampling.seeds == (111, 222)


def test_runtime_simultaneous_equal_request_seeds_receive_distinct_salts():
    runtime = make_runtime(seed_buffer=FakeLazySeedBuffer())
    prepared = prepare(
        runtime,
        positions=(5, 5),
        sampling_params=seeded_sampling(77, 77),
        reset=True,
    )

    runtime._refresh_sampling_seeds(prepared)

    snapshot = runtime._seed_state.snapshot()
    assert snapshot.request_seeds == (77, 77)
    assert snapshot.salts == (0, 1)
    assert snapshot.current_device_seeds[0] != snapshot.current_device_seeds[1]


def test_runtime_slot_remap_moves_complete_seed_stream_before_refresh():
    seed_buffer = FakeLazySeedBuffer()
    runtime = make_runtime(seed_buffer=seed_buffer)
    initial = prepare(
        runtime,
        positions=(5, -1),
        sampling_params=seeded_sampling(77),
        reset=True,
    )
    runtime._refresh_sampling_seeds(initial)
    original_device_seed = int(seed_buffer.updates[-1][0])
    original_state = runtime._seed_state.snapshot()

    moved = prepare(
        runtime,
        positions=(-1, 5),
        sampling_params=seeded_sampling(None, 77),
        slot_remap=torch.tensor([0, 0]),
        reset=False,
    )
    runtime._refresh_sampling_seeds(moved)

    state = runtime._seed_state.snapshot()
    assert state.active == (False, True)
    assert state.request_seeds == (None, 77)
    assert state.token_counters[1] == original_state.token_counters[0]
    assert int(seed_buffer.updates[-1][1]) == original_device_seed


def test_runtime_seed_changes_with_request_seed_and_decode_position():
    seed_buffer = FakeLazySeedBuffer()
    runtime = make_runtime(seed_buffer=seed_buffer)

    runtime._refresh_sampling_seeds(
        prepare(runtime, positions=(7, -1), sampling_params=seeded_sampling(101), reset=True)
    )
    position_7 = int(seed_buffer.updates[-1][0])
    runtime._refresh_sampling_seeds(
        prepare(runtime, positions=(8, -1), sampling_params=seeded_sampling(101), reset=False)
    )
    position_8 = int(seed_buffer.updates[-1][0])
    runtime._refresh_sampling_seeds(
        prepare(runtime, positions=(7, -1), sampling_params=seeded_sampling(202), reset=True)
    )
    different_request = int(seed_buffer.updates[-1][0])

    assert len({position_7, position_8, different_request}) == 3


def test_runtime_explicit_seed_absolute_position_is_stable_across_reset_boundaries():
    seed_buffer = FakeLazySeedBuffer()
    runtime = make_runtime(seed_buffer=seed_buffer)
    request = seeded_sampling(909)

    runtime._refresh_sampling_seeds(prepare(runtime, positions=(19, -1), sampling_params=request, reset=True))
    original = seed_buffer.updates[-1].clone()
    runtime._refresh_sampling_seeds(prepare(runtime, positions=(20, -1), sampling_params=request, reset=False))
    continued = seed_buffer.updates[-1].clone()
    runtime._refresh_sampling_seeds(prepare(runtime, positions=(19, -1), sampling_params=request, reset=True))
    restarted = seed_buffer.updates[-1].clone()
    runtime._refresh_sampling_seeds(prepare(runtime, positions=(20, -1), sampling_params=request, reset=True))
    resumed = seed_buffer.updates[-1].clone()

    assert torch.equal(original, restarted)
    assert torch.equal(continued, resumed)


def test_runtime_seed_refreshes_before_eager_model_invocation(monkeypatch):
    events = []
    seed_buffer = FakeLazySeedBuffer()
    original_update = seed_buffer.update

    def record_update(source):
        events.append("seed")
        original_update(source)

    seed_buffer.update = record_update
    runtime = make_runtime(seed_buffer=seed_buffer)
    prepared = prepare(runtime, positions=(3, -1), sampling_params=seeded_sampling(77), reset=True)
    monkeypatch.setattr(runtime, "_prepare_inputs_host", lambda prepared: object())
    monkeypatch.setattr(
        runtime,
        "_stage_inputs_and_kpt",
        lambda host, prepared: (DecodeDeviceInputs(None, None, None, None), None),
    )

    def run_body(*args, **kwargs):
        events.append("invoke")
        return object()

    monkeypatch.setattr(runtime, "_run_body", run_body)

    runtime.invoke(prepared)

    assert events[-1] == "invoke"
    assert events[:-1]
    assert set(events[:-1]) == {"seed"}


def test_runtime_trace_captures_stable_seed_handle_and_refreshes_before_replay(monkeypatch):
    events = []
    seed_buffer = FakeLazySeedBuffer()
    original_update = seed_buffer.update

    def record_update(source):
        events.append("seed")
        original_update(source)

    seed_buffer.update = record_update
    runtime = make_runtime(seed_buffer=seed_buffer)
    prepared = prepare(runtime, positions=(12, -1), sampling_params=seeded_sampling(88), reset=True)
    monkeypatch.setattr(runtime, "_prepare_inputs_host", lambda prepared: object())
    monkeypatch.setattr(
        runtime,
        "_stage_inputs_and_kpt",
        lambda host, prepared: (DecodeDeviceInputs(None, None, None, None), None),
    )

    persistent = runtime.capture_plan(prepared).prepare_inputs()
    assert persistent.seed_buffer is seed_buffer.get_device_buffer()
    sampling = prepared.prepared_sampling
    assert sampling is not None
    persistent = dataclasses.replace(
        persistent,
        kpt_signature=[(sampling.top_k, sampling.top_p, sampling.temperature)],
    )
    runtime.refresh_trace(persistent, prepared, SimpleNamespace(full=False, page_table=False))
    events.append("replay")

    assert events[-1] == "replay"
    assert events[:-1]
    assert set(events[:-1]) == {"seed"}


def test_runtime_seed_handling_does_not_mutate_sampling_params():
    seed_buffer = FakeLazySeedBuffer()
    runtime = make_runtime(seed_buffer=seed_buffer)
    sampling_params = seeded_sampling(11, 22)
    before = dataclasses.asdict(sampling_params)

    prepared = prepare(runtime, positions=(4, 9), sampling_params=sampling_params, reset=True)
    runtime._refresh_sampling_seeds(prepared)

    assert dataclasses.asdict(sampling_params) == before


def test_runtime_unseeded_stream_varies_and_same_absolute_position_is_idempotent():
    seed_buffer = FakeLazySeedBuffer()
    defaults = seed_buffer.source.clone()
    runtime = make_runtime(seed_buffer=seed_buffer)

    unseeded = prepare(runtime, positions=(5, -1), sampling_params=stochastic_sampling(), reset=True)
    runtime._refresh_sampling_seeds(unseeded)
    first = seed_buffer.updates[-1].clone()
    first_state = runtime._seed_state.snapshot()
    assert int(first[0]) != int(defaults[0])
    assert int(first[1]) == int(defaults[1])

    runtime._refresh_sampling_seeds(dataclasses.replace(unseeded, reset_batch=False))
    repeated = seed_buffer.updates[-1].clone()
    assert torch.equal(repeated, first)

    advanced = prepare(
        runtime,
        positions=(6, -1),
        sampling_params=stochastic_sampling(),
        reset=False,
    )
    runtime._refresh_sampling_seeds(advanced)
    snapshot = runtime._seed_state.snapshot()
    assert snapshot.request_seeds == (None, None)
    assert snapshot.active == (True, False)
    assert snapshot.token_counters[0] == 2
    assert snapshot.unseeded_rng_states[0] != first_state.unseeded_rng_states[0]


def test_runtime_seed_change_requires_reset_and_preserves_unseeded_survivor(expect_error):
    seed_buffer = FakeLazySeedBuffer()
    runtime = make_runtime(seed_buffer=seed_buffer)
    unseeded = prepare(
        runtime,
        positions=(10, 10),
        sampling_params=stochastic_sampling(seed=[None, None]),
        reset=True,
    )
    runtime._refresh_sampling_seeds(unseeded)
    initial = runtime._seed_state.snapshot()

    mixed = prepare(
        runtime,
        positions=(11, 11),
        sampling_params=stochastic_sampling(seed=[None, 42]),
        reset=False,
    )
    with expect_error(RuntimeError, "reset_batch=True"):
        runtime._refresh_sampling_seeds(mixed)

    runtime._refresh_sampling_seeds(dataclasses.replace(mixed, reset_batch=True))
    admitted = runtime._seed_state.snapshot()
    assert admitted.request_seeds == (None, 42)
    assert admitted.active == (True, True)
    assert admitted.token_counters[0] == initial.token_counters[0] + 1

    continued = dataclasses.replace(mixed, start_pos=torch.tensor([12, 12]))
    runtime._refresh_sampling_seeds(continued)
    continued_state = runtime._seed_state.snapshot()
    assert continued_state.token_counters[0] == admitted.token_counters[0] + 1
    assert continued_state.request_seeds == (None, 42)


def test_runtime_inactive_peer_is_cleaned_up_without_readmitting_survivor():
    seed_buffer = FakeLazySeedBuffer()
    defaults = seed_buffer.source.clone()
    runtime = make_runtime(seed_buffer=seed_buffer)
    initial = prepare(
        runtime,
        positions=(20, 20),
        sampling_params=stochastic_sampling(seed=[None, 42]),
        reset=True,
    )
    runtime._refresh_sampling_seeds(initial)
    initial_state = runtime._seed_state.snapshot()

    peer_left = prepare(
        runtime,
        positions=(21, -1),
        sampling_params=stochastic_sampling(seed=[None, 42]),
        reset=False,
    )
    runtime._refresh_sampling_seeds(peer_left)

    state = runtime._seed_state.snapshot()
    assert state.active == (True, False)
    assert state.request_seeds == (None, None)
    assert state.token_counters[0] == initial_state.token_counters[0] + 1
    assert int(seed_buffer.updates[-1][1]) == int(defaults[1])


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
                ("prompt_tokens", inspect.Parameter.KEYWORD_ONLY, None),
                ("output_tokens", inspect.Parameter.KEYWORD_ONLY, None),
                ("slot_remap", inspect.Parameter.KEYWORD_ONLY, None),
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
    assert config.max_device_top_k == 32
    assert config.sampling_batch_size == 2
    assert config.sampling_state_controller is None
    assert config.sampling_state is None
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


def test_signature_tracks_native_penalty_and_sampled_logprob_program_modes():
    runtime = make_runtime()
    sampling = SamplingParams(
        temperature=[0.0, 0.0],
        top_k=[1, 1],
        top_p=[1.0, 1.0],
        presence_penalty=[0.5, 0.0],
        enable_log_probs=[True, False],
        num_logprobs=[0, -2],
    )

    prepared = prepare(runtime, sampling_params=sampling)
    native = prepared.prepared_sampling
    signature = runtime.program_signature(prepared)

    assert native is not None
    assert native.logprob_modes == ("sampled_token", "none")
    assert native.penalties_enabled
    assert native.log_probs_enabled
    assert prepared.sampling_path == "topk"
    assert signature.penalties_enabled
    assert signature.logprobs_enabled
    assert signature.key_material()[-2:] == (
        ("penalties_enabled", True),
        ("logprobs_enabled", True),
    )


def test_configured_topk_policy_is_not_collapsed_to_argmax_by_greedy_temperature():
    runtime = make_runtime(force_greedy_top_k=True)
    sampling = SamplingParams(temperature=[0.0, 0.0], top_k=[32, 32], top_p=[0.08, 0.08])

    prepared = prepare(runtime, sampling_params=sampling)

    assert prepared.sampling_path == "topk"


def test_unconfigured_topk_values_use_argmax_for_greedy_temperature():
    runtime = make_runtime()
    sampling = SamplingParams(temperature=[0.0, 0.0], top_k=[32, 32], top_p=[0.08, 0.08])

    assert prepare(runtime, sampling_params=sampling).sampling_path == "argmax"


def test_sampling_params_are_prepared_once_and_reused_for_kpt(monkeypatch):
    runtime = make_runtime(force_greedy_top_k=True)
    calls = []
    formatter = decode_module.prepare_sampling_params

    def formatter_spy(*args, **kwargs):
        calls.append((args, kwargs))
        return formatter(*args, **kwargs)

    monkeypatch.setattr(
        decode_module,
        "prepare_sampling_params",
        formatter_spy,
    )
    prepared = prepare(runtime, sampling_params=greedy_sampling())
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: "mapper")
    monkeypatch.setattr(ttnn, "from_torch", lambda value, **kwargs: value)

    runtime._make_host_kpt(prepared)
    assert len(calls) == 1


def test_tile_padded_sampler_preserves_one_semantic_lane_and_neutralizes_inactive_rows(monkeypatch):
    mesh = FakeMesh()
    model = FakeModel()
    model.config.max_batch_size = 1
    model.sampling.config.max_batch_size = 32
    runtime = DecodeRuntime(
        DecodeRuntimeConfig.resolve(
            model=model,
            output_reader=OutputReader(mesh),
            lane_capacity=1,
            page_table_layout=page_table_layout(),
            device_sampling_enabled=True,
            force_greedy_top_k=True,
        )
    )
    prepared = runtime.prepare(
        torch.tensor([11]),
        torch.tensor([0]),
        torch.tensor([[3, 4, 5]], dtype=torch.int32),
        sampling_params=SamplingParams(
            temperature=0.7,
            top_k=32,
            top_p=0.08,
            presence_penalty=0.25,
            frequency_penalty=0.5,
            repetition_penalty=1.2,
            seed=17,
            enable_log_probs=True,
            num_logprobs=1,
        ),
    )
    sampling = prepared.prepared_sampling

    assert runtime.config.lane_capacity == 1
    assert runtime.config.sampling_batch_size == 32
    assert sampling.batch_size == 32
    assert sampling.active_rows == 1
    assert sampling.active_mask == (True,) + (False,) * 31
    assert sampling.top_k == (32,) + (1,) * 31
    assert sampling.top_p == pytest.approx((0.08,) + (0.0,) * 31)
    assert sampling.temperature == pytest.approx((1.0 / 0.7,) + (1.0,) * 31)
    assert sampling.seeds == (17,) + (None,) * 31
    assert sampling.presence_penalty == pytest.approx((0.25,) + (0.0,) * 31)
    assert sampling.frequency_penalty == pytest.approx((0.5,) + (0.0,) * 31)
    assert sampling.repetition_penalty == pytest.approx((1.2,) + (1.0,) * 31)
    assert sampling.enable_log_probs == (True,) + (False,) * 31
    assert sampling.num_logprobs == (1,) + (0,) * 31
    assert sampling.logprob_modes[1:] == ("none",) * 31

    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh_device: "mapper")
    monkeypatch.setattr(ttnn, "from_torch", lambda value, **kwargs: value)
    k, p, temperature = runtime._make_host_kpt(prepared)

    assert tuple(k.shape) == tuple(p.shape) == tuple(temperature.shape) == (32,)
    assert k.tolist() == [32] + [1] * 31
    assert p.tolist() == pytest.approx([0.08] + [0.0] * 31)
    assert temperature.tolist() == pytest.approx([1.0 / 0.7] + [1.0] * 31)


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

    def run_body(
        inputs,
        prepared,
        kpt,
        *,
        device_feedback,
        count_tokens=True,
        advance_seeds=True,
    ):
        return "captured"

    monkeypatch.setattr(runtime, "_run_body", run_body)

    plan = runtime.capture_plan(prepared)
    persistent = plan.prepare_inputs()

    assert persistent.device_inputs is device
    assert persistent.kpt == "kpt"
    sampling = prepared.prepared_sampling
    assert sampling is not None
    assert persistent.kpt_signature == [(sampling.top_k, sampling.top_p, sampling.temperature)]
    assert plan.capture(persistent) == "captured"
    assert plan.refresh_policy.every_replay == ("sampling",)
    assert plan.refresh_policy.full_on_batch_reset
    assert plan.refresh_policy.full_on_graph_switch
    assert plan.refresh_policy.full_without_device_feedback
    assert plan.refresh_policy.refresh_page_table_on_change


def test_trace_refresh_skips_unchanged_sampling_values(monkeypatch):
    runtime = make_runtime(force_greedy_top_k=True)
    prepared = prepare(runtime, sampling_params=greedy_sampling())
    sampling = prepared.prepared_sampling
    assert sampling is not None
    persistent = DecodePersistentInputs(
        device_inputs=DecodeDeviceInputs("tokens", "positions", "rotary", "page_table"),
        kpt="kpt",
        kpt_signature=[(sampling.top_k, sampling.top_p, sampling.temperature)],
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
        lambda inputs, prepared, kpt, *, device_feedback, **kwargs: calls.append(device_feedback) or ("raw", None),
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


def test_sampled_token_logprobs_are_flattened_to_lane_row_order():
    runtime = make_runtime()
    host_tokens = torch.tensor([[[[7], [8]]]], dtype=torch.int32)
    host_log_probs = torch.tensor([[[[-0.25, -0.75]]]], dtype=torch.bfloat16)

    tokens, log_probs = runtime._normalize_host_output(
        (host_tokens, host_log_probs),
        is_tokens=True,
    )

    assert tokens.tolist() == [7, 8]
    assert tokens.dtype == torch.int64
    assert log_probs.tolist() == pytest.approx([-0.25, -0.75])
    assert log_probs.dtype == torch.float32


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
