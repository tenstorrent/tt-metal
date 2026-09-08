# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-free contract tests for the rollout-engine lifecycle."""

from __future__ import annotations

from unittest.mock import Mock, call

import pytest

from utils.rollout_engine import (
    EngineFailed,
    EngineState,
    InvalidTransitionError,
    PolicyActivated,
    PromptGroupLease,
    ResultReady,
    RolloutEngine,
    RolloutOutput,
    RolloutResult,
    WeightsStaged,
)


class MockRolloutEngine(RolloutEngine):
    def __init__(self, *, engine_id: str = "engine-0", event_sink: Mock | None = None) -> None:
        self.actions = Mock()
        super().__init__(engine_id=engine_id, active_version=1, event_sink=event_sink)

    def _start_rollout_action(self, lease):
        self.actions.start_rollout(lease)

    def _stage_weights_action(self, version, source):
        self.actions.stage_weights(version, source)

    def _activate_staged_action(self, version):
        self.actions.activate(version)

    def _discard_staged_action(self, version):
        self.actions.discard(version)

    def _shutdown_action(self):
        self.actions.shutdown()


def lease(version: int = 1, lease_id: str = "lease-1") -> PromptGroupLease:
    return PromptGroupLease(
        lease_id=lease_id,
        group_id="group-1",
        behavior_version=version,
        attempt_id=2,
        payload={"prompt": [1, 2]},
    )


def test_rollout_output_requires_one_logprob_per_generated_token(expect_error):
    output = RolloutOutput.from_sequences([[3, 4], [5]], [[-0.1, -0.2], [-0.3]])

    assert output.tokens == ((3, 4), (5,))
    assert output.logprobs == ((-0.1, -0.2), (-0.3,))
    with expect_error(ValueError, "same batch size"):
        RolloutOutput.from_sequences([[3]], [])
    with expect_error(ValueError, "same length for row 0"):
        RolloutOutput.from_sequences([[3, 4]], [[-0.1]])


def test_rollout_result_keeps_lease_identity_and_behavior_version():
    sink = Mock()
    engine = MockRolloutEngine(event_sink=sink)
    active_lease = lease()

    engine.start_rollout(active_lease)
    assert engine.snapshot().state is EngineState.GENERATING
    output = RolloutOutput.from_sequences([[3]], [[-0.25]])
    engine.rollout_completed(active_lease.lease_id, output)

    assert engine.snapshot().state is EngineState.READY
    result = sink.call_args.args[0]
    assert result == ResultReady(
        result=RolloutResult(
            engine_id="engine-0",
            lease_id="lease-1",
            group_id="group-1",
            attempt_id=2,
            behavior_version=1,
            output=output,
            request_payload=active_lease.payload,
        )
    )


@pytest.mark.parametrize("stage_first", [False, True], ids=["drain-first", "stage-first"])
def test_activation_waits_for_both_rollout_completion_and_staging(stage_first):
    sink = Mock()
    engine = MockRolloutEngine(event_sink=sink)
    active_lease = lease()
    weights = object()
    engine.start_rollout(active_lease)

    if stage_first:
        engine.stage_weights(2, weights)
        engine.weights_staged(2)
        assert engine.snapshot().state is EngineState.DRAINING
        engine.rollout_completed(active_lease.lease_id, RolloutOutput.from_sequences([[3]], [[-0.25]]))
    else:
        engine.quiesce(2)
        engine.rollout_completed(active_lease.lease_id, RolloutOutput.from_sequences([[3]], [[-0.25]]))
        assert engine.snapshot().state is EngineState.WAITING_FOR_STAGE
        engine.stage_weights(2, weights)
        engine.weights_staged(2)

    assert engine.snapshot().state is EngineState.ACTIVATING
    engine.actions.activate.assert_called_once_with(2)
    engine.activation_completed(2)

    snapshot = engine.snapshot()
    assert snapshot.state is EngineState.READY
    assert snapshot.active_version == 2
    assert snapshot.target_version is None
    assert snapshot.staged_version is None
    expected_events = (
        [WeightsStaged, ResultReady, PolicyActivated] if stage_first else [ResultReady, WeightsStaged, PolicyActivated]
    )
    assert [type(args.args[0]) for args in sink.call_args_list] == expected_events


def test_stage_request_itself_stops_new_old_version_work(expect_error):
    engine = MockRolloutEngine()

    engine.stage_weights(2, object())

    assert engine.snapshot().state is EngineState.WAITING_FOR_STAGE
    with expect_error(InvalidTransitionError, "expected READY"):
        engine.start_rollout(lease())


def test_quiesce_is_idempotent_and_activation_occurs_once():
    engine = MockRolloutEngine()

    engine.quiesce(2)
    engine.quiesce(2)
    engine.stage_weights(2, object())
    engine.weights_staged(2)
    engine.quiesce(2)

    engine.actions.activate.assert_called_once_with(2)


def test_staging_failure_preserves_active_version_and_fails_with_context():
    sink = Mock()
    engine = MockRolloutEngine(event_sink=sink)
    engine.stage_weights(2, object())

    engine.weight_staging_failed(2, RuntimeError("transfer incomplete"))

    snapshot = engine.snapshot()
    assert snapshot.state is EngineState.FAILED
    assert snapshot.active_version == 1
    engine.actions.discard.assert_called_once_with(2)
    assert sink.call_args.args[0] == EngineFailed(
        engine_id="engine-0",
        operation="stage_weights",
        message="transfer incomplete",
        active_version=1,
        target_version=2,
        lease_id=None,
    )


def test_activation_failure_preserves_active_version_and_stops_engine():
    engine = MockRolloutEngine()
    engine.stage_weights(2, object())
    engine.weights_staged(2)

    engine.activation_failed(2, "copy failed")

    assert engine.snapshot().state is EngineState.FAILED
    assert engine.snapshot().active_version == 1
    assert engine.snapshot().staged_version is None
    engine.actions.discard.assert_called_once_with(2)


def test_backend_action_exception_is_reported_and_reraised(expect_error):
    sink = Mock()
    engine = MockRolloutEngine(event_sink=sink)
    engine.actions.start_rollout.side_effect = RuntimeError("generator failed")

    with expect_error(RuntimeError, "generator failed"):
        engine.start_rollout(lease())

    assert engine.snapshot().state is EngineState.FAILED
    failure = sink.call_args.args[0]
    assert isinstance(failure, EngineFailed)
    assert failure.operation == "start_rollout"
    assert failure.lease_id == "lease-1"


@pytest.mark.parametrize("operation", ["stage_weights", "activate"])
def test_weight_action_exception_discards_staging_and_preserves_active_version(operation, expect_error):
    engine = MockRolloutEngine()
    getattr(engine.actions, operation).side_effect = RuntimeError("device operation failed")

    with expect_error(RuntimeError, "device operation failed"):
        engine.stage_weights(2, object())
        if operation == "activate":
            engine.weights_staged(2)

    assert engine.snapshot().state is EngineState.FAILED
    assert engine.snapshot().active_version == 1
    assert engine.snapshot().staged_version is None
    engine.actions.discard.assert_called_once_with(2)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda engine: engine.start_rollout(lease(version=0)), "does not match"),
        (lambda engine: engine.quiesce(1), "must be newer"),
        (lambda engine: engine.weights_staged(2), "does not match"),
        (lambda engine: engine.activation_completed(2), "expected ACTIVATING"),
    ],
)
def test_invalid_or_stale_commands_do_not_change_ready_state(operation, message, expect_error):
    engine = MockRolloutEngine()

    with expect_error(InvalidTransitionError, message):
        operation(engine)

    assert engine.snapshot().state is EngineState.READY


def test_only_one_staged_version_is_allowed_and_targets_cannot_change_mid_cutover(expect_error):
    engine = MockRolloutEngine()
    engine.stage_weights(2, object())

    with expect_error(InvalidTransitionError, "conflicts"):
        engine.quiesce(3)
    with expect_error(InvalidTransitionError, "already occupies staging"):
        engine.stage_weights(2, object())


def test_wrong_lease_completion_is_rejected_without_losing_active_lease(expect_error):
    engine = MockRolloutEngine()
    engine.start_rollout(lease())

    with expect_error(InvalidTransitionError, "does not match"):
        engine.rollout_completed("another-lease", RolloutOutput.from_sequences([[3]], [[-0.25]]))

    assert engine.snapshot().active_lease_id == "lease-1"
    assert engine.snapshot().state is EngineState.GENERATING


def test_failed_and_stopped_engines_reject_further_work(expect_error):
    failed = MockRolloutEngine()
    failed.start_rollout(lease())
    failed.rollout_failed("lease-1", "timeout")

    with expect_error(InvalidTransitionError, "FAILED"):
        failed.quiesce(2)

    stopped = MockRolloutEngine()
    stopped.shutdown()
    assert stopped.snapshot().state is EngineState.STOPPED
    stopped.actions.assert_has_calls([call.shutdown()])
    with expect_error(InvalidTransitionError, "STOPPED"):
        stopped.stage_weights(2, object())


def test_shutdown_rejects_active_rollout(expect_error):
    engine = MockRolloutEngine()
    engine.start_rollout(lease())

    with expect_error(InvalidTransitionError, "GENERATING"):
        engine.shutdown()


def test_shutdown_rejects_inflight_staging(expect_error):
    engine = MockRolloutEngine()
    engine.stage_weights(2, object())

    with expect_error(InvalidTransitionError, "WAITING_FOR_STAGE"):
        engine.shutdown()


def test_engine_instances_are_isolated_for_future_multi_worker_coordinator():
    engine_0 = MockRolloutEngine(engine_id="engine-0")
    engine_1 = MockRolloutEngine(engine_id="engine-1")

    engine_0.start_rollout(lease(lease_id="lease-0"))
    engine_1.stage_weights(2, object())

    assert engine_0.snapshot().state is EngineState.GENERATING
    assert engine_0.snapshot().active_lease_id == "lease-0"
    assert engine_1.snapshot().state is EngineState.WAITING_FOR_STAGE
    assert engine_1.snapshot().active_lease_id is None
