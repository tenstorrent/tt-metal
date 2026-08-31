# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Eager host contract tests for the TTTv2-native seed lifecycle."""

import ast
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from models.common.modules.lazy_buffer import LazyBuffer
from models.common.modules.sampling.seed_manager_1d import SeedManager1D, SeedState, _hash_request_seed_to_device_seed


class _EntropySequence:
    def __init__(self, start=1000):
        self._next = start

    def __call__(self, _bits):
        value = self._next
        self._next += 1
        return value


class _TrackingSeedBuffer(LazyBuffer):
    """LazyBuffer test double that models a materialized stable handle."""

    def __init__(self, defaults):
        super().__init__(source=defaults)
        self.handle = object()
        self.materialize_calls = 0
        self.updates = []

    def get_device_buffer(self):
        self.materialize_calls += 1
        return self.handle

    def update(self, new_source):
        self.source = new_source
        self.updates.append(new_source.detach().clone())


def _make_manager(capacity=4):
    defaults = torch.arange(capacity, dtype=torch.int32)
    buffer = _TrackingSeedBuffer(defaults)
    config = SimpleNamespace(max_batch_size=capacity, seeds=buffer)
    manager = SeedManager1D(config, entropy_factory=_EntropySequence())
    return manager, manager.create_state(), buffer, defaults


def test_seed_state_requires_consistent_caller_owned_slot_storage(expect_error):
    with expect_error(ValueError, "same capacity"):
        SeedState(
            active=[False],
            request_seeds=[],
            token_counters=[0],
            salts=[0],
            unseeded_rngs=[random.Random(1)],
            last_absolute_positions=[None],
            current_device_seeds=[None],
        )


def test_seed_manager_requires_sampling_config_lazy_buffer_and_matching_capacity(expect_error):
    with expect_error(TypeError, "mutable LazyBuffer-compatible"):
        SeedManager1D(SimpleNamespace(max_batch_size=4, seeds=torch.arange(4)))

    buffer = _TrackingSeedBuffer(torch.arange(3, dtype=torch.int32))
    with expect_error(ValueError, "does not match"):
        SeedManager1D(SimpleNamespace(max_batch_size=4, seeds=buffer))


def test_prefill_admission_resets_counters_and_assigns_simultaneous_equal_seed_salts():
    manager, state, _, _ = _make_manager()

    manager.admit(state, [42, 42, None], [0, 2, 3])
    snapshot = state.snapshot()

    assert snapshot.active_slots == (0, 2, 3)
    assert snapshot.request_seeds == (42, None, 42, None)
    assert snapshot.token_counters == (0, 0, 0, 0)
    assert snapshot.salts == (0, 0, 1, 0)

    manager.refresh(state, [0, 2], positions={0: 8, 2: 8})
    manager.admit(state, [42], [2])
    assert state.snapshot().token_counters[2] == 0
    assert state.snapshot().salts[2] == 1


def test_absolute_position_refresh_is_idempotent_and_sequential_refresh_resumes_after_it():
    manager, state, buffer, defaults = _make_manager()
    manager.admit(state, [707], [1])

    first = manager.refresh(state, [1], positions={1: 13})
    assert first[1] == _hash_request_seed_to_device_seed(707, 14)
    assert state.snapshot().token_counters[1] == 15

    repeated = manager.refresh(state, [1], positions={1: 13})
    assert repeated == first
    assert state.snapshot().token_counters[1] == 15

    next_position = manager.refresh(state, [1], positions={1: 14})
    assert next_position[1] == _hash_request_seed_to_device_seed(707, 15)
    assert state.snapshot().token_counters[1] == 16

    sequential = manager.refresh(state, [1])
    assert sequential[1] == _hash_request_seed_to_device_seed(707, 16)
    assert state.snapshot().token_counters[1] == 17
    assert torch.equal(buffer.source, defaults)


def test_equal_request_seeds_produce_distinct_salted_streams():
    manager, state, _, _ = _make_manager()
    manager.admit(state, [1234, 1234, 1234, 1234], [0, 1, 2, 3])

    values = manager.refresh(state, [0, 1, 2, 3], positions=[7, 7, 7, 7])

    assert state.snapshot().salts == (0, 1, 2, 3)
    assert len(set(values)) == 4


def test_unseeded_rng_state_is_diverse_and_same_position_trace_refresh_does_not_advance_twice():
    manager, state, _, _ = _make_manager()
    manager.admit(state, [None, None], [0, 1])

    first = manager.refresh(state, [0, 1], positions=[20, 20])
    after_first = state.snapshot()
    repeated = manager.refresh(state, [0, 1], positions=[20, 20])
    after_repeated = state.snapshot()

    assert first == repeated
    assert first[0] != first[1]
    assert after_first.token_counters[:2] == (1, 1)
    assert after_repeated.token_counters == after_first.token_counters
    assert after_repeated.unseeded_rng_states == after_first.unseeded_rng_states

    second = manager.refresh(state, [0, 1], positions=[21, 21])
    assert second[0] != first[0]
    assert second[1] != first[1]
    assert state.snapshot().token_counters[:2] == (2, 2)


def test_synchronize_requires_reset_for_admission_and_preserves_survivor_state(expect_error):
    manager, state, _, _ = _make_manager()
    slot_seeds = [42, None, None, None]

    with expect_error(RuntimeError, "reset_batch=True"):
        manager.synchronize(state, slot_seeds, [0], reset_batch=False)

    manager.synchronize(state, slot_seeds, [0], reset_batch=True)
    manager.refresh(state, [0], positions=[9, -1, -1, -1])
    survivor = state.snapshot()

    manager.synchronize(state, [42, 42, None, None], [0, 1], reset_batch=True)
    updated = state.snapshot()
    assert updated.token_counters[0] == survivor.token_counters[0]
    assert updated.salts[0] == survivor.salts[0] == 0
    assert updated.current_device_seeds[0] == survivor.current_device_seeds[0]
    assert updated.salts[1] == 1

    with expect_error(RuntimeError, "reset_batch=True"):
        manager.synchronize(state, [77, 42, None, None], [0, 1], reset_batch=False)
    assert state.snapshot() == updated


def test_slot_remap_moves_counter_salt_rng_and_current_buffer_value_then_vacates_source():
    manager, state, buffer, defaults = _make_manager()
    manager.admit(state, [55, 55], [0, 3])
    manager.refresh(state, [0, 3], positions={0: 8, 3: 8})
    source = state.snapshot()

    manager.apply_slot_remap(state, torch.tensor([0, 3, 2, 3], dtype=torch.int32))
    moved = state.snapshot()

    assert moved.request_seeds[1] == source.request_seeds[3] == 55
    assert moved.token_counters[1] == source.token_counters[3]
    assert moved.salts[1] == source.salts[3] == 1
    assert moved.unseeded_rng_states[1] == source.unseeded_rng_states[3]
    assert moved.current_device_seeds[1] == source.current_device_seeds[3]
    assert moved.active[3] is False
    assert buffer.updates[-1][1].item() == source.current_device_seeds[3]
    assert buffer.updates[-1][3].item() == defaults[3].item()


def test_suspend_resume_preserves_unseeded_rng_stream_across_slot_movement():
    manager, state, _, _ = _make_manager()
    manager.admit(state, [None], [3])
    manager.refresh(state, [3], positions={3: 10})

    checkpoint = manager.suspend(state, 3)
    expected_rng = random.Random()
    expected_rng.setstate(checkpoint.unseeded_rng_state)
    expected_next = expected_rng.randint(1, 1_000_000)

    manager.resume(state, 1, checkpoint)
    resumed = manager.refresh(state, [1], positions={1: 11})

    assert resumed[1] == expected_next
    assert state.snapshot().token_counters[1] == checkpoint.token_counter + 1
    assert state.snapshot().active_slots == (1,)


def test_cleanup_removes_ghost_seed_and_restores_default_buffer_rows():
    manager, state, buffer, defaults = _make_manager()
    manager.admit(state, [42], [3])
    manager.refresh(state, [3], positions={3: 4})

    manager.cleanup(state, [])

    assert state.snapshot().active_slots == ()
    assert state.snapshot().request_seeds == (None, None, None, None)
    assert torch.equal(buffer.updates[-1], defaults)

    manager.admit(state, [42], [1])
    assert state.snapshot().salts[1] == 0


def test_restore_defaults_keeps_request_state_and_reset_discards_it():
    manager, state, buffer, defaults = _make_manager()
    manager.admit(state, [99], [2])
    request_values = manager.refresh(state, [2], positions={2: 6})
    before_restore = state.snapshot()

    manager.restore_defaults(state)

    assert torch.equal(buffer.updates[-1], defaults)
    assert state.snapshot().active_slots == (2,)
    assert state.snapshot().token_counters == before_restore.token_counters
    assert state.snapshot().buffer_is_default is True

    replay_values = manager.refresh(state, [2], positions={2: 6})
    assert replay_values == request_values
    assert state.snapshot().token_counters == before_restore.token_counters

    manager.reset(state)
    reset = state.snapshot()
    assert reset.active_slots == ()
    assert reset.request_seeds == (None, None, None, None)
    assert reset.token_counters == (0, 0, 0, 0)
    assert reset.salts == (0, 0, 0, 0)
    assert reset.current_device_seeds == (None, None, None, None)
    assert reset.buffer_is_default is True
    assert torch.equal(buffer.updates[-1], defaults)


def test_refresh_updates_one_stable_handle_and_never_promotes_request_values_to_defaults():
    manager, state, buffer, defaults = _make_manager()
    handle = manager.get_seed_device_buffer()
    manager.admit(state, [123], [0])

    manager.refresh(state, [0], positions=[1, -1, -1, -1])
    manager.refresh(state, [0], positions=[2, -1, -1, -1])

    assert manager.get_seed_device_buffer() is handle
    assert buffer.materialize_calls >= 4
    assert len(buffer.updates) == 2
    assert not torch.equal(buffer.updates[0], buffer.updates[1])
    assert torch.equal(buffer.source, defaults)


def test_slot_validation_rejects_ambiguous_or_out_of_capacity_lifecycle_updates(expect_error):
    manager, state, _, _ = _make_manager()

    with expect_error(ValueError, "unique"):
        manager.admit(state, [1, 2], [0, 0])
    with expect_error(ValueError, "outside"):
        manager.admit(state, [1], [4])
    with expect_error(ValueError, "expected 2 request seeds"):
        manager.admit(state, [1], [0, 1])
    with expect_error(ValueError, "must contain 4"):
        manager.apply_slot_remap(state, [0, 1])

    manager.admit(state, [1], [3])
    with expect_error(ValueError, "do not cover active seed slot 3"):
        manager.refresh(state, [3], positions=[0])
    with expect_error(ValueError, "multiple destinations"):
        manager.apply_slot_remap(state, [3, 3, 2, 3])


def test_seed_manager_1d_imports_no_legacy_sampling_state_or_generator():
    module_path = Path(__file__).parents[3] / "modules" / "sampling" / "seed_manager_1d.py"
    tree = ast.parse(module_path.read_text())
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "models.common.sampling.generator" not in imported_modules
    assert "models.common.sampling.tt_sampling" not in imported_modules
    assert "models.common.sampling.tt_penalties" not in imported_modules
