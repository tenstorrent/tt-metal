# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-free tests for single-worker trainer-side rollout coordination."""

from __future__ import annotations

from unittest.mock import Mock, call

from utils.rollout_coordinator import SingleWorkerRolloutCoordinator
from utils.rollout_engine import (
    EngineFailed,
    PolicyActivated,
    PromptGroupLease,
    ResultReady,
    RolloutOutput,
    RolloutResult,
    WeightsStaged,
)
from utils.rollout_transport import RemoteRolloutError, create_in_memory_rollout_transports


def _lease(lease_id: str = "lease-1", *, version: int = 3, attempt_id: int = 0) -> PromptGroupLease:
    return PromptGroupLease(
        lease_id=lease_id,
        group_id="group-1",
        behavior_version=version,
        attempt_id=attempt_id,
        payload=[[11, 12]],
    )


def _result(lease: PromptGroupLease) -> RolloutResult:
    return RolloutResult(
        engine_id="engine-0",
        lease_id=lease.lease_id,
        group_id=lease.group_id,
        behavior_version=lease.behavior_version,
        attempt_id=lease.attempt_id,
        output=RolloutOutput.from_sequences([[21, 22]], [[-0.1, -0.2]]),
    )


def _coordinator(*, capacity: int = 8):
    transports = create_in_memory_rollout_transports(capacity=capacity)
    bridge = Mock()
    coordinator = SingleWorkerRolloutCoordinator(
        engine_id="engine-0",
        active_version=3,
        transport=transports.trainer,
        weight_bridge=bridge,
    )
    return coordinator, transports.worker, bridge


def test_submit_tracks_identity_and_delivers_matching_result():
    coordinator, worker, _bridge = _coordinator()
    lease = _lease()

    coordinator.submit(lease)
    assert worker.receive() == lease
    worker.publish_event(ResultReady(_result(lease)))

    assert coordinator.receive_result() == _result(lease)
    assert coordinator.outstanding_count == 0


def test_cutover_buffers_drained_result_and_waits_for_activation(expect_error):
    coordinator, worker, bridge = _coordinator()
    lease = _lease()
    weights = {"model.weight": object()}
    coordinator.submit(lease)

    coordinator.begin_policy_cutover(4, weights)

    assert worker.receive() == lease
    assert worker.receive().target_version == 4
    assert worker.receive().version == 4
    assert bridge.method_calls == [call.send_weights(weights), call.barrier()]
    assert coordinator.active_version == 3
    with expect_error(RuntimeError, "cutover to version 4"):
        coordinator.submit(_lease("lease-2", version=4))

    worker.publish_event(ResultReady(_result(lease)))
    worker.publish_event(WeightsStaged("engine-0", 4))
    worker.publish_event(PolicyActivated("engine-0", 4))

    coordinator.await_policy_activation(4)

    assert coordinator.active_version == 4
    assert coordinator.receive_result() == _result(lease)
    new_lease = _lease("lease-2", version=4)
    coordinator.submit(new_lease)
    assert worker.receive() == new_lease


def test_cutover_rejects_activation_before_staging_ack(expect_error):
    coordinator, worker, _bridge = _coordinator()
    coordinator.begin_policy_cutover(4, {"weight": object()})
    worker.publish_event(PolicyActivated("engine-0", 4))

    with expect_error(RuntimeError, "before weights-staged"):
        coordinator.await_policy_activation(4)


def test_result_identity_mismatch_is_rejected(expect_error):
    coordinator, worker, _bridge = _coordinator()
    lease = _lease()
    coordinator.submit(lease)
    mismatched = RolloutResult(
        engine_id="engine-0",
        lease_id=lease.lease_id,
        group_id=lease.group_id,
        behavior_version=lease.behavior_version,
        attempt_id=lease.attempt_id + 1,
        output=_result(lease).output,
    )
    worker.publish_event(ResultReady(mismatched))

    with expect_error(RuntimeError, "does not match submitted lease"):
        coordinator.receive_result()


def test_duplicate_result_is_rejected_after_first_delivery(expect_error):
    coordinator, worker, _bridge = _coordinator()
    lease = _lease()
    result = _result(lease)
    coordinator.submit(lease)
    worker.publish_event(ResultReady(result))
    assert coordinator.receive_result() == result

    worker.publish_event(ResultReady(result))
    with expect_error(RuntimeError, "unknown or completed lease"):
        coordinator.receive_event()


def test_weight_transfer_failure_makes_coordinator_terminal(expect_error):
    coordinator, _worker, bridge = _coordinator()
    bridge.send_weights.side_effect = RuntimeError("weight send failed")

    with expect_error(RuntimeError, "weight send failed"):
        coordinator.begin_policy_cutover(4, {"weight": object()})
    with expect_error(RuntimeError, "weight send failed"):
        coordinator.submit(_lease())


def test_remote_failure_is_terminal_and_preserves_context(expect_error):
    coordinator, worker, _bridge = _coordinator()
    failure = EngineFailed(
        engine_id="engine-0",
        operation="activate",
        message="device copy failed",
        active_version=3,
        target_version=4,
        lease_id=None,
    )
    worker.publish_event(failure)

    with expect_error(RemoteRolloutError, "device copy failed") as error:
        coordinator.receive_result()
    assert error.value.failure == failure

    with expect_error(RemoteRolloutError, "device copy failed"):
        coordinator.submit(_lease())


def test_multi_worker_extension_is_isolated_by_engine_id(expect_error):
    coordinator, worker, _bridge = _coordinator()
    worker.publish_event(WeightsStaged("engine-1", 4))

    with expect_error(RuntimeError, "expected engine-0"):
        coordinator.receive_event()
