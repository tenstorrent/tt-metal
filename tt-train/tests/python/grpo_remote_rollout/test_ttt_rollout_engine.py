# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from threading import Event, Thread
from unittest.mock import Mock

from utils.rollout_engine import (
    EngineFailed,
    EngineState,
    PolicyActivated,
    PromptGroupLease,
    ResultReady,
    WeightsStaged,
)
from utils.ttt_rollout_engine import TttRolloutEngine


def _lease(*, version: int = 1, payload=None) -> object:
    return PromptGroupLease(
        lease_id="lease-1",
        group_id="group-1",
        behavior_version=version,
        payload=payload if payload is not None else [[11, 12], [21]],
    )


def _dependencies(*, completions=None, weights=None):
    worker = Mock()
    worker.models = [object()]
    worker.generate.return_value = completions if completions is not None else [[31], [41, 42]]
    bridge = Mock()
    bridge.receive_weights.return_value = weights if weights is not None else [{"weight": object()}]
    return worker, bridge


def _engine(*, worker, bridge, events, **kwargs):
    return TttRolloutEngine(
        engine_id="tt-0",
        active_version=1,
        worker=worker,
        weight_bridge=bridge,
        event_sink=events.append,
        **kwargs,
    )


def test_generation_delegates_to_worker_and_publishes_versioned_result():
    events = []
    worker, bridge = _dependencies()
    engine = _engine(
        worker=worker,
        bridge=bridge,
        events=events,
        max_new_tokens=64,
        enable_trace=False,
        stop_at_eos=False,
    )
    lease = _lease()

    engine.start_rollout(lease)

    worker.generate.assert_called_once_with(
        lease.payload,
        max_new_tokens=64,
        enable_trace=False,
        stop_at_eos=False,
    )
    assert engine.snapshot().state is EngineState.READY
    assert len(events) == 1
    assert isinstance(events[0], ResultReady)
    assert events[0].result.behavior_version == 1
    assert events[0].result.payload == worker.generate.return_value


def test_idle_cutover_receives_fences_and_activates_one_staged_version():
    events = []
    worker, bridge = _dependencies()
    engine = _engine(worker=worker, bridge=bridge, events=events)

    engine.quiesce(2)
    engine.stage_weights(2, source="trainer-rank")

    bridge.receive_weights.assert_called_once_with()
    bridge.barrier.assert_called_once_with()
    worker.update_weights.assert_called_once_with(bridge.receive_weights.return_value)
    assert engine.snapshot().state is EngineState.READY
    assert engine.snapshot().active_version == 2
    assert [type(event) for event in events] == [WeightsStaged, PolicyActivated]


def test_weight_receive_overlaps_final_rollout_but_activation_waits_for_it():
    events = []
    generation_started = Event()
    finish_generation = Event()
    worker, bridge = _dependencies()

    def blocking_generate(*_args, **_kwargs):
        generation_started.set()
        assert finish_generation.wait(timeout=5)
        return [[31], [41, 42]]

    worker.generate.side_effect = blocking_generate
    engine = _engine(worker=worker, bridge=bridge, events=events)
    rollout_thread = Thread(target=engine.start_rollout, args=(_lease(),))

    rollout_thread.start()
    assert generation_started.wait(timeout=5)
    engine.stage_weights(2, source=None)

    assert engine.snapshot().state is EngineState.DRAINING
    assert engine.snapshot().staged_version == 2
    worker.update_weights.assert_not_called()

    finish_generation.set()
    rollout_thread.join(timeout=5)

    assert not rollout_thread.is_alive()
    worker.update_weights.assert_called_once_with(bridge.receive_weights.return_value)
    assert engine.snapshot().active_version == 2
    assert [type(event) for event in events] == [WeightsStaged, ResultReady, PolicyActivated]


def test_rollout_may_finish_while_weight_receive_is_still_in_progress():
    events = []
    generation_started = Event()
    finish_generation = Event()
    receive_started = Event()
    finish_receive = Event()
    worker, bridge = _dependencies()
    received_weights = [{"weight": object()}]

    def blocking_generate(*_args, **_kwargs):
        generation_started.set()
        assert finish_generation.wait(timeout=5)
        return [[31], [41, 42]]

    def blocking_receive():
        receive_started.set()
        assert finish_receive.wait(timeout=5)
        return received_weights

    worker.generate.side_effect = blocking_generate
    bridge.receive_weights.side_effect = blocking_receive
    engine = _engine(worker=worker, bridge=bridge, events=events)
    rollout_thread = Thread(target=engine.start_rollout, args=(_lease(),))
    stage_thread = Thread(target=engine.stage_weights, args=(2, None))

    rollout_thread.start()
    assert generation_started.wait(timeout=5)
    stage_thread.start()
    assert receive_started.wait(timeout=5)

    finish_generation.set()
    rollout_thread.join(timeout=5)
    assert not rollout_thread.is_alive()
    assert engine.snapshot().state is EngineState.WAITING_FOR_STAGE
    worker.update_weights.assert_not_called()

    finish_receive.set()
    stage_thread.join(timeout=5)

    assert not stage_thread.is_alive()
    worker.update_weights.assert_called_once_with(received_weights)
    assert engine.snapshot().active_version == 2
    assert [type(event) for event in events] == [ResultReady, WeightsStaged, PolicyActivated]


def test_invalid_received_target_count_fails_staging(expect_error):
    events = []
    worker, bridge = _dependencies(weights=[])
    engine = _engine(worker=worker, bridge=bridge, events=events)

    with expect_error(ValueError, "worker expects 1"):
        engine.stage_weights(2, source=None)

    assert engine.snapshot().state is EngineState.FAILED
    worker.update_weights.assert_not_called()
    assert isinstance(events[-1], EngineFailed)
    assert events[-1].operation == "stage_weights"


def test_activation_failure_drops_private_staging_and_preserves_active_version(expect_error):
    events = []
    worker, bridge = _dependencies()
    worker.update_weights.side_effect = RuntimeError("device copy failed")
    engine = _engine(worker=worker, bridge=bridge, events=events)

    with expect_error(RuntimeError, "device copy failed"):
        engine.stage_weights(2, source=None)

    snapshot = engine.snapshot()
    assert snapshot.state is EngineState.FAILED
    assert snapshot.active_version == 1
    assert snapshot.staged_version is None
    assert engine._staged is None
    assert isinstance(events[-1], EngineFailed)
    assert events[-1].operation == "activate"


def test_constructor_rejects_invalid_worker_or_generation_limit(expect_error):
    events = []
    worker, bridge = _dependencies()
    worker.models = []

    with expect_error(ValueError, "at least one model"):
        _engine(worker=worker, bridge=bridge, events=events)

    worker.models = [object()]
    with expect_error(ValueError, "non-negative"):
        _engine(worker=worker, bridge=bridge, events=events, max_new_tokens=-1)
