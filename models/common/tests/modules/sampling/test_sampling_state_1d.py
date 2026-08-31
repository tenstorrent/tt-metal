# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host/fake contract tests for SamplingState1D.

These tests deliberately avoid TT device construction.  They verify controller
ordering, ownership, topology/config agreement, and exactly-once bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from models.common.modules.lazy_buffer import LazyBuffer
from models.common.modules.sampling import penalties_1d as penalties_module
from models.common.modules.sampling.params import PreparedSamplingParams, place_prepared_sampling_params
from models.common.modules.sampling.penalties_1d import Penalties1D
from models.common.modules.sampling.sampling_state_1d import SamplingState1D


class FakeMesh:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def get_num_devices(self):
        return self.shape[0] * self.shape[1]


class FakeBuffer:
    def __init__(self, source):
        self.source = source.clone()
        self.handle = object()
        self.updates = []
        self.released = False

    def get_device_buffer(self):
        return self.handle

    def update(self, source):
        self.source = source.clone()
        self.updates.append(source.clone())

    def release(self):
        self.released = True


@dataclass
class FakeSeedState:
    capacity: int
    active: list[bool]
    seeds: list[int | None]

    @property
    def request_seeds(self):
        return self.seeds


class FakeSeedManager:
    def __init__(self, config, events):
        self.max_batch_size = config.max_batch_size
        self.seed_buffer = config.seeds
        self.events = events

    def create_state(self):
        return FakeSeedState(
            capacity=self.max_batch_size,
            active=[False] * self.max_batch_size,
            seeds=[None] * self.max_batch_size,
        )

    def reset(self, state):
        self.events.append(("seed.reset",))
        state.active[:] = [False] * state.capacity
        state.seeds[:] = [None] * state.capacity

    def admit(self, state, seeds, slots):
        slots = tuple(slots)
        seeds = tuple(seeds)
        self.events.append(("seed.admit", seeds, slots))
        for slot, seed in zip(slots, seeds):
            state.active[slot] = True
            state.seeds[slot] = seed

    def synchronize(self, state, seeds, active_slots, *, reset_batch):
        active_slots = tuple(active_slots)
        self.events.append(("seed.synchronize", active_slots, reset_batch))
        changed = [slot for slot in active_slots if not state.active[slot] or state.seeds[slot] != seeds[slot]]
        if changed and not reset_batch:
            raise RuntimeError(
                "new or changed active seed slots require reset_batch=True or an explicit admit() call: " f"{changed}"
            )
        active = set(active_slots)
        for slot in range(state.capacity):
            state.active[slot] = slot in active
            state.seeds[slot] = seeds[slot] if slot in active else None

    def refresh(self, state, active_slots, *, positions=None):
        self.events.append(("seed.refresh", tuple(active_slots), positions))

    def refresh_prefill_replicated(self, state, slot, *, position=None):
        self.events.append(("seed.prefill_replicated", int(slot), position))
        return 123

    def restore_defaults(self, state):
        self.events.append(("seed.restore_defaults",))

    def apply_slot_remap(self, state, remap):
        remap = tuple(remap)
        self.events.append(("seed.remap", remap))
        old_active = tuple(state.active)
        old_seeds = tuple(state.seeds)
        state.active[:] = [old_active[source] for source in remap]
        state.seeds[:] = [old_seeds[source] for source in remap]

    def cleanup(self, state, live_slots):
        live_slots = tuple(live_slots)
        self.events.append(("seed.cleanup", live_slots))
        live = set(live_slots)
        for slot in range(state.capacity):
            if slot not in live:
                state.active[slot] = False
                state.seeds[slot] = None

    def get_seed_device_buffer(self):
        self.events.append(("seed.handle",))
        return self.seed_buffer.handle


class FakePenalties:
    _BUFFER_NAMES = (
        "prompt_mask",
        "output_mask",
        "output_counts",
        "output_counts_gathered",
        "zeros",
        "decode_src",
        "presence_penalties",
        "frequency_penalties",
        "repetition_penalties",
        "inverse_repetition_penalties",
    )

    def __init__(self, config, events, *, config_updates=None):
        buffers = {}
        for name in self._BUFFER_NAMES:
            shape = (
                (config.max_batch_size, config.vocab_size)
                if name
                in {
                    "prompt_mask",
                    "output_mask",
                    "output_counts",
                    "output_counts_gathered",
                    "zeros",
                }
                else (config.max_batch_size, 1)
            )
            buffers[name] = FakeBuffer(torch.zeros(shape))
        self.config = replace(config, **buffers, **(config_updates or {}))
        self.events = events
        self.loaded = False
        self.released = False

    def load_device_buffers(self):
        self.events.append(("penalty.load",))
        self.loaded = True

    def init_prompt_penalties(self, params, accum, prompt_tokens):
        self.events.append(("penalty.prompt", prompt_tokens.clone()))

    def reset_output_tokens(self, accum, tokens=None):
        self.events.append(("penalty.output_reset", None if tokens is None else tokens.clone()))

    def decode_forward(self, logits, params, accum):
        self.events.append(("penalty.decode", logits))
        return f"penalized:{logits}"

    def update_output_tokens(self, accum, tokens):
        self.events.append(("penalty.update", tokens))

    def release(self):
        self.events.append(("penalty.release",))
        self.released = True
        for name in self._BUFFER_NAMES:
            getattr(self.config, name).release()


class FakeSampling:
    def __init__(self, config, events):
        self.config = config
        self.events = events
        self.released = False
        self.raise_on_decode = False

    def decode_forward(self, logits, **kwargs):
        self.events.append(("sampling.decode", logits, kwargs))
        if self.raise_on_decode:
            raise RuntimeError("sampling failed")
        return "sampled-tokens", "sampled-logprobs"

    def release(self):
        self.released = True


def _make_controller(shape=(1, 4), *, penalty_config_updates=None):
    events = []
    mesh = FakeMesh(shape)
    seed_buffer = FakeBuffer(torch.arange(4, dtype=torch.int32))
    sub_core_grids = object()
    config = SimpleNamespace(
        vocab_size=128,
        valid_vocab_size=127,
        mesh_device=mesh,
        max_batch_size=4,
        max_top_k=32,
        sub_core_grids=sub_core_grids,
        seeds=seed_buffer,
    )
    sampling = FakeSampling(config, events)
    penalties = None

    def penalties_factory(penalties_config):
        nonlocal penalties
        penalties = FakePenalties(
            penalties_config,
            events,
            config_updates=penalty_config_updates,
        )
        return penalties

    controller = SamplingState1D(
        sampling,
        penalties_factory=penalties_factory,
        seed_manager_factory=lambda sampling_config: FakeSeedManager(sampling_config, events),
    )
    return controller, sampling, penalties, events


def _prepared(
    *,
    penalties=True,
    sampling_path="topk",
    log_probs=False,
    slot_remap=None,
    presence=(0.5, 0.0, 0.0, 0.0),
    seeds=(11, None, None, None),
):
    repetition = (1.5, 1.0, 1.0, 1.0) if penalties else (1.0, 1.0, 1.0, 1.0)
    frequency = (0.25, 0.0, 0.0, 0.0) if penalties else (0.0, 0.0, 0.0, 0.0)
    presence = presence if penalties else (0.0, 0.0, 0.0, 0.0)
    row_path = "argmax" if sampling_path == "argmax" else "topk"
    logprob_modes = ("sampled_token", "none", "none", "none") if log_probs else ("none",) * 4
    return PreparedSamplingParams(
        top_k=(1, 5, 1, 1),
        top_p=(0.0, 0.9, 0.0, 0.0),
        temperature=(1.0, 1.25, 1.0, 1.0),
        presence_penalty=presence,
        frequency_penalty=frequency,
        repetition_penalty=repetition,
        seeds=seeds,
        enable_log_probs=(log_probs, False, False, False),
        num_logprobs=(0, 0, 0, 0),
        logprob_modes=logprob_modes,
        greedy_mask=(sampling_path == "argmax", sampling_path == "argmax", False, False),
        row_paths=(row_path, row_path, "inactive", "inactive"),
        active_mask=(True, True, False, False),
        sampling_path=sampling_path,
        active_rows=2,
        batch_size=4,
        max_device_top_k=32,
        prompt_tokens=torch.tensor([[1, 2], [3, -1]]),
        output_tokens=torch.tensor([[7, 8], [9, -1]]),
        slot_remap=slot_remap,
    )


def _event_names(events):
    return [event[0] for event in events]


def test_constructor_derives_exact_penalty_contract_from_borrowed_sampler():
    controller, sampling, penalties, _ = _make_controller()

    assert controller.sampling is sampling
    assert penalties.config.mesh_device is sampling.config.mesh_device
    assert penalties.config.vocab_size == sampling.config.vocab_size
    assert penalties.config.max_batch_size == sampling.config.max_batch_size
    assert penalties.config.sub_core_grids is sampling.config.sub_core_grids
    assert controller.seed_manager.seed_buffer is sampling.config.seeds


def test_constructor_rejects_non_1d_topology_before_constructing_state(expect_error):
    with expect_error(ValueError, "only supports 1D"):
        _make_controller(shape=(2, 2))


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"vocab_size": 256}, "vocab_size"),
        ({"max_batch_size": 8}, "max_batch_size"),
        ({"mesh_device": FakeMesh((1, 4))}, "mesh_device"),
        ({"sub_core_grids": object()}, "sub_core_grids"),
    ],
)
def test_constructor_rejects_penalty_contract_drift(updates, message, expect_error):
    with expect_error(ValueError, message):
        _make_controller(penalty_config_updates=updates)


def test_create_state_materializes_noop_params_and_clears_history(expect_error):
    controller, _, penalties, events = _make_controller()
    state = controller.create_state()

    assert state.seed_state.capacity == 4
    assert state.active_mask == (False,) * 4
    assert penalties.loaded
    assert torch.equal(penalties.config.presence_penalties.source, torch.zeros(4, 1))
    assert torch.equal(penalties.config.frequency_penalties.source, torch.zeros(4, 1))
    assert torch.equal(penalties.config.repetition_penalties.source, torch.ones(4, 1))
    assert torch.equal(penalties.config.inverse_repetition_penalties.source, torch.ones(4, 1))
    assert _event_names(events)[-3:] == ["seed.reset", "penalty.prompt", "penalty.output_reset"]

    with expect_error(RuntimeError, "already live"):
        controller.create_state()


def test_admit_updates_params_seeds_histories_and_static_identity():
    controller, _, penalties, events = _make_controller()
    state = controller.create_state()
    prepared = _prepared(log_probs=True)
    events.clear()

    controller.admit(state, prepared)

    assert state.active_slots == (0, 1)
    assert state.static_identity.sampling_path == "topk"
    assert state.static_identity.penalties_enabled
    assert state.static_identity.log_probs_enabled
    assert state.static_identity.logprob_modes == ("sampled_token", "none")
    assert torch.allclose(
        penalties.config.presence_penalties.source.reshape(-1),
        torch.tensor([0.5, 0.0, 0.0, 0.0]),
    )
    assert torch.allclose(
        penalties.config.inverse_repetition_penalties.source.reshape(-1),
        torch.tensor([1.0 / 1.5, 1.0, 1.0, 1.0]),
    )
    assert _event_names(events) == [
        "seed.admit",
        "seed.synchronize",
        "penalty.prompt",
        "penalty.output_reset",
    ]
    assert torch.equal(events[-2][1], prepared.prompt_tokens)
    assert torch.equal(events[-1][1], prepared.output_tokens)


def test_admit_validates_repetition_history_before_mutating_seed_state(expect_error):
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    prepared = replace(_prepared(), prompt_tokens=None)
    events.clear()

    with expect_error(ValueError, "prompt_tokens are required"):
        controller.admit(state, prepared)

    assert events == []
    assert not any(state.seed_state.active)


def test_prepared_capability_must_match_the_borrowed_sampler(expect_error):
    controller, _, _, _ = _make_controller()
    state = controller.create_state()
    prepared = replace(_prepared(), max_device_top_k=64)

    with expect_error(ValueError, "max_device_top_k"):
        controller.admit(state, prepared)


def test_decode_reset_validates_history_before_mutating_seed_state(expect_error):
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    prepared = replace(_prepared(), prompt_tokens=None)
    events.clear()

    with expect_error(ValueError, "prompt_tokens are required"):
        controller.synchronize_decode(state, prepared, reset_batch=True)

    assert events == []
    assert not any(state.seed_state.active)


def test_refresh_updates_dynamic_params_and_seeds_without_rebuilding_history():
    controller, _, penalties, events = _make_controller()
    state = controller.create_state()
    controller.admit(state, _prepared())
    changed = _prepared(presence=(0.75, 0.0, 0.0, 0.0))
    positions = (3, 4, -1, -1)
    events.clear()

    controller.refresh_dynamic_inputs(state, changed, positions=positions)

    assert _event_names(events) == ["seed.synchronize", "seed.refresh"]
    assert events[-1][2] is positions
    assert penalties.config.presence_penalties.source[0, 0].item() == pytest.approx(0.75)


def test_prefill_admission_keeps_seed_on_decode_slot_and_samples_request_order():
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    prepared = replace(
        _prepared(),
        active_mask=(True, False, False, False),
        active_rows=1,
        row_paths=("topk", "inactive", "inactive", "inactive"),
        seeds=(77, None, None, None),
    )
    events.clear()

    controller.admit_prefill(state, prepared, slots=(3,), positions=(12,))
    output = controller.prefill_forward(
        "logits",
        state,
        prepared,
        k=object(),
        p=object(),
        temp=object(),
    )

    assert state.seed_state.active == [False, False, False, True]
    assert state.seed_state.seeds[3] == 77
    assert ("seed.prefill_replicated", 3, 12) in events
    assert controller.penalties.config.presence_penalties.source.reshape(-1).tolist() == [0.5] * 4
    assert output == ("sampled-tokens", "sampled-logprobs")
    assert state.penalty_history_valid is False


def test_no_penalty_prefill_keeps_history_valid_for_gap_decode_without_reset():
    controller, _, _, _ = _make_controller()
    state = controller.create_state()
    prepared = replace(
        _prepared(penalties=False),
        active_mask=(True, False, False, False),
        active_rows=1,
        row_paths=("topk", "inactive", "inactive", "inactive"),
        seeds=(77, None, None, None),
    )

    controller.admit_prefill(state, prepared, slots=(3,), positions=(12,))
    controller.prefill_forward(
        "logits",
        state,
        prepared,
        k=object(),
        p=object(),
        temp=object(),
    )
    decode_prepared = place_prepared_sampling_params(prepared, (3,))

    assert state.penalty_history_valid is True
    assert controller.synchronize_decode(state, decode_prepared, reset_batch=False) is decode_prepared
    assert state.active_slots == (3,)


def test_decode_identity_remap_admits_new_slot_without_reset_and_preserves_survivor():
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    initial = replace(
        _prepared(penalties=False),
        active_mask=(True, False, False, False),
        active_rows=1,
        row_paths=("topk", "inactive", "inactive", "inactive"),
        seeds=(11, None, None, None),
    )
    controller.admit(state, initial)
    events.clear()
    expanded = _prepared(
        penalties=False,
        seeds=(11, 22, None, None),
        slot_remap=(0, 1, 2, 3),
    )

    consumed = controller.synchronize_decode(state, expanded, reset_batch=False)

    assert consumed.slot_remap is None
    assert state.seed_state.active == [True, True, False, False]
    assert state.seed_state.seeds == [11, 22, None, None]
    assert _event_names(events)[:3] == ["seed.remap", "seed.admit", "seed.synchronize"]
    assert events[1] == ("seed.admit", (22,), (1,))
    assert events[2] == ("seed.synchronize", (0, 1), False)


def test_decode_new_slot_without_remap_remains_strict_and_does_not_leak(expect_error):
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    initial = replace(
        _prepared(penalties=False),
        active_mask=(True, False, False, False),
        active_rows=1,
        row_paths=("topk", "inactive", "inactive", "inactive"),
        seeds=(11, None, None, None),
    )
    controller.admit(state, initial)
    before = (tuple(state.seed_state.active), tuple(state.seed_state.seeds))
    events.clear()
    expanded = _prepared(penalties=False, seeds=(11, 22, None, None))

    with expect_error(RuntimeError, "reset_batch=True"):
        controller.synchronize_decode(state, expanded, reset_batch=False)

    assert (tuple(state.seed_state.active), tuple(state.seed_state.seeds)) == before
    assert events == [("seed.synchronize", (0, 1), False)]


def test_decode_nonidentity_remap_changed_survivor_rejects_before_mutation(expect_error):
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    initial = replace(
        _prepared(penalties=False),
        active_mask=(True, False, False, False),
        active_rows=1,
        row_paths=("topk", "inactive", "inactive", "inactive"),
        seeds=(11, None, None, None),
    )
    controller.admit(state, initial)
    before = (tuple(state.seed_state.active), tuple(state.seed_state.seeds))
    events.clear()
    changed = _prepared(
        penalties=False,
        seeds=(22, 99, None, None),
        slot_remap=(1, 0, 2, 3),
    )

    with expect_error(RuntimeError, "changed active seed slots"):
        controller.synchronize_decode(state, changed, reset_batch=False)

    assert (tuple(state.seed_state.active), tuple(state.seed_state.seeds)) == before
    assert events == []


def test_decode_remap_validates_new_penalty_history_before_mutation(expect_error):
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    initial = replace(
        _prepared(penalties=False),
        active_mask=(True, False, False, False),
        active_rows=1,
        row_paths=("topk", "inactive", "inactive", "inactive"),
        seeds=(11, None, None, None),
    )
    controller.admit(state, initial)
    before = (tuple(state.seed_state.active), tuple(state.seed_state.seeds))
    events.clear()
    incomplete = replace(
        _prepared(seeds=(22, 11, None, None), slot_remap=(1, 0, 2, 3)),
        prompt_tokens=None,
    )

    with expect_error(ValueError, "prompt_tokens are required"):
        controller.synchronize_decode(state, incomplete, reset_batch=False)

    assert (tuple(state.seed_state.active), tuple(state.seed_state.seeds)) == before
    assert events == []


def test_decode_slot_remap_moves_survivor_then_admits_new_slot():
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    initial = replace(
        _prepared(penalties=False),
        active_mask=(True, False, False, False),
        active_rows=1,
        row_paths=("topk", "inactive", "inactive", "inactive"),
        seeds=(11, None, None, None),
    )
    controller.admit(state, initial)
    events.clear()
    remapped = _prepared(
        penalties=False,
        seeds=(22, 11, None, None),
        slot_remap=(1, 0, 2, 3),
    )

    consumed = controller.synchronize_decode(state, remapped, reset_batch=False)

    assert consumed.slot_remap is None
    assert state.seed_state.active == [True, True, False, False]
    assert state.seed_state.seeds == [22, 11, None, None]
    assert _event_names(events)[:3] == ["seed.remap", "seed.admit", "seed.synchronize"]
    assert events[1] == ("seed.admit", (22,), (0,))


def test_penalized_prefill_gap_decode_rebuilds_history_on_batch_reset():
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    prepared = replace(
        _prepared(),
        active_mask=(True, False, False, False),
        active_rows=1,
        row_paths=("topk", "inactive", "inactive", "inactive"),
        seeds=(77, None, None, None),
    )
    controller.admit_prefill(state, prepared, slots=(3,), positions=(12,))
    controller.prefill_forward(
        "logits",
        state,
        prepared,
        k=object(),
        p=object(),
        temp=object(),
    )
    decode_prepared = place_prepared_sampling_params(
        replace(prepared, output_tokens=torch.tensor([[7, 8, 99], [9, -1, -1]])),
        (3,),
    )
    events.clear()

    controller.synchronize_decode(state, decode_prepared, reset_batch=True)
    controller.refresh_dynamic_inputs(
        state,
        decode_prepared,
        positions=(-1, -1, -1, 13),
    )

    assert state.penalty_history_valid is True
    assert state.active_slots == (3,)
    assert _event_names(events) == [
        "seed.synchronize",
        "penalty.prompt",
        "penalty.output_reset",
        "seed.synchronize",
        "seed.refresh",
    ]


def test_refresh_rejects_static_identity_change_until_trace_is_reselected(expect_error):
    controller, _, _, _ = _make_controller()
    state = controller.create_state()
    controller.admit(state, _prepared())

    with expect_error(RuntimeError, "static identity changed"):
        controller.refresh_dynamic_inputs(state, _prepared(sampling_path="argmax"))


def test_slot_remap_moves_seed_state_and_rebuilds_penalty_history():
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    controller.admit(state, _prepared())
    remapped = _prepared(
        slot_remap=(1, 0, 2, 3),
        seeds=(None, 11, None, None),
    )
    events.clear()

    consumed = controller.apply_slot_remap(state, remapped)

    assert _event_names(events) == [
        "seed.remap",
        "seed.synchronize",
        "penalty.prompt",
        "penalty.output_reset",
    ]
    assert consumed.slot_remap is None
    events.clear()
    controller.refresh_dynamic_inputs(state, consumed, positions=(1, 1, -1, -1))
    assert _event_names(events) == ["seed.synchronize", "seed.refresh"]


def test_before_after_sampling_enforces_penalty_order_and_exactly_once_update(expect_error):
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    prepared = _prepared()
    controller.admit(state, prepared)
    events.clear()

    logits, sample_id = controller.before_sampling("logits", state, prepared, positions=(0, 0, -1, -1))
    assert logits == "penalized:logits"
    controller.after_sampling(state, "token", sample_id=sample_id)

    assert _event_names(events) == [
        "seed.synchronize",
        "seed.refresh",
        "penalty.decode",
        "penalty.update",
    ]
    with expect_error(RuntimeError, "not the pending"):
        controller.after_sampling(state, "token", sample_id=sample_id)


def test_atomic_decode_composes_penalties_sampling_seeds_and_history():
    controller, sampling, _, events = _make_controller()
    state = controller.create_state()
    prepared = _prepared(log_probs=True)
    controller.admit(state, prepared)
    events.clear()
    k, p, temp = object(), object(), object()

    output = controller.decode_forward(
        "logits",
        state,
        prepared,
        k=k,
        p=p,
        temp=temp,
        positions=(2, 2, -1, -1),
    )

    assert output == ("sampled-tokens", "sampled-logprobs")
    assert _event_names(events) == [
        "seed.synchronize",
        "seed.refresh",
        "penalty.decode",
        "seed.handle",
        "sampling.decode",
        "penalty.update",
    ]
    kwargs = events[-2][2]
    assert kwargs["k"] is k and kwargs["p"] is p and kwargs["temp"] is temp
    assert kwargs["seeds"] is sampling.config.seeds.handle
    assert kwargs["enable_log_probs"] == [True, False, False, False]
    assert state.pending_sample_id is None


def test_sampling_failure_cancels_pending_step_without_updating_history(expect_error):
    controller, sampling, _, events = _make_controller()
    state = controller.create_state()
    prepared = _prepared()
    controller.admit(state, prepared)
    sampling.raise_on_decode = True
    events.clear()

    with expect_error(RuntimeError, "sampling failed"):
        controller.decode_forward(
            "logits",
            state,
            prepared,
            k=object(),
            p=object(),
            temp=object(),
            positions=(2, 2, -1, -1),
        )

    assert state.pending_sample_id is None
    assert "penalty.update" not in _event_names(events)


def test_compile_only_decode_does_not_count_a_phantom_output_token():
    controller, _, penalties, events = _make_controller()
    state = controller.create_state()
    prepared = _prepared()
    controller.admit(state, prepared)
    penalty_param_updates = len(penalties.config.presence_penalties.updates)
    events.clear()

    controller.decode_forward(
        "logits",
        state,
        prepared,
        k=object(),
        p=object(),
        temp=object(),
        count_tokens=False,
        advance_seeds=False,
    )

    assert "sampling.decode" in _event_names(events)
    assert "seed.restore_defaults" not in _event_names(events)
    assert "seed.refresh" not in _event_names(events)
    assert "seed.synchronize" not in _event_names(events)
    assert "penalty.update" not in _event_names(events)
    assert len(penalties.config.presence_penalties.updates) == penalty_param_updates
    assert state.pending_sample_id is None


def test_trace_capture_body_records_penalty_update_without_capturing_dynamic_writes():
    controller, _, penalties, events = _make_controller()
    state = controller.create_state()
    prepared = _prepared()
    controller.admit(state, prepared)
    penalty_param_updates = len(penalties.config.presence_penalties.updates)
    events.clear()

    controller.decode_forward(
        "logits",
        state,
        prepared,
        k=object(),
        p=object(),
        temp=object(),
        count_tokens=True,
        advance_seeds=False,
    )

    assert _event_names(events) == [
        "penalty.decode",
        "seed.handle",
        "sampling.decode",
        "penalty.update",
    ]
    assert len(penalties.config.presence_penalties.updates) == penalty_param_updates


def test_argmax_restores_default_seeds_and_does_not_require_kpt():
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    prepared = _prepared(penalties=False, sampling_path="argmax")
    controller.admit(state, prepared)
    events.clear()

    controller.decode_forward("logits", state, prepared)

    assert "seed.restore_defaults" in _event_names(events)
    assert "seed.handle" not in _event_names(events)
    assert "penalty.decode" not in _event_names(events)
    assert "penalty.update" not in _event_names(events)


def test_cleanup_validates_before_mutating_and_rebuilds_live_history(expect_error):
    controller, _, _, events = _make_controller()
    state = controller.create_state()
    prepared = _prepared()
    controller.admit(state, prepared)
    events.clear()

    with expect_error(ValueError, "prepared sampling state is required"):
        controller.cleanup(state, (0, 1))
    assert events == []

    controller.cleanup(state, (0, 1), prepared=prepared)
    assert _event_names(events) == ["seed.cleanup", "penalty.prompt", "penalty.output_reset"]


def test_release_resets_caller_state_and_releases_only_owned_penalties(expect_error):
    controller, sampling, penalties, events = _make_controller()
    state = controller.create_state()
    controller.admit(state, _prepared())
    events.clear()

    controller.release(state)

    assert state.released
    assert penalties.released
    assert not sampling.released
    assert _event_names(events) == ["seed.reset", "penalty.release"]
    with expect_error(RuntimeError, "released"):
        controller.reset(state)
    controller.release(state)


def test_penalties_release_deallocates_owned_lazy_buffers_and_slice_tensors(monkeypatch):
    released = []
    monkeypatch.setattr(penalties_module.ttnn, "deallocate", released.append)
    names = FakePenalties._BUFFER_NAMES
    buffers = {}
    values = []
    for name in names:
        buffer = LazyBuffer(source=torch.zeros(1))
        value = object()
        buffer._value = value
        buffers[name] = buffer
        values.append(value)

    penalties = object.__new__(Penalties1D)
    penalties.config = SimpleNamespace(**buffers)
    penalties._slice_start = object()
    penalties._slice_end = object()
    values.extend((penalties._slice_start, penalties._slice_end))
    penalties._decode_src = buffers["decode_src"]._value
    penalties._zeros = buffers["zeros"]._value
    penalties._device_buffers_loaded = True

    penalties.release()
    penalties.release()

    assert released == values
    assert all(buffer._value is None for buffer in buffers.values())
    assert penalties._slice_start is None and penalties._slice_end is None
    assert penalties._decode_src is None and penalties._zeros is None
    assert not penalties._device_buffers_loaded
