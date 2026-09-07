# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-free tests for bounded rollout transport semantics."""

from __future__ import annotations

from queue import Empty, Full
from threading import Event, Thread

from utils.rollout_engine import EngineFailed, PromptGroupLease, RolloutOutput, RolloutResult
from utils.rollout_transport import (
    QuiescePolicy,
    RemoteRolloutError,
    RolloutTransportClosed,
    StagePolicyWeights,
    create_in_memory_rollout_transports,
)


def _lease(lease_id: str = "lease-1") -> PromptGroupLease:
    return PromptGroupLease(
        lease_id=lease_id,
        group_id="group-1",
        behavior_version=7,
        attempt_id=2,
        payload=[[11, 12]],
    )


def _result(lease_id: str = "lease-1") -> RolloutResult:
    return RolloutResult(
        engine_id="engine-0",
        lease_id=lease_id,
        group_id="group-1",
        behavior_version=7,
        attempt_id=2,
        output=RolloutOutput.from_sequences([[21, 22]], [[-0.1, -0.2]]),
    )


def test_queue_preserves_lease_identity_policy_version_tokens_and_logprobs():
    transports = create_in_memory_rollout_transports(capacity=1)

    transports.trainer.submit(_lease())
    received = transports.worker.receive()
    transports.worker.publish(_result(received.lease_id))

    assert received == _lease()
    assert transports.trainer.receive_result() == _result()


def test_bounded_request_and_result_queues_apply_backpressure(expect_error):
    transports = create_in_memory_rollout_transports(capacity=1)
    transports.trainer.submit(_lease("lease-1"))

    with expect_error(Full):
        transports.trainer.submit(_lease("lease-2"), timeout=0)

    transports.worker.receive()
    transports.worker.publish(_result("lease-1"))
    with expect_error(Full):
        transports.worker.publish(_result("lease-2"), timeout=0)


def test_policy_control_commands_share_ordered_request_lane():
    transports = create_in_memory_rollout_transports(capacity=2)

    transports.trainer.quiesce(8)
    transports.trainer.request_weight_stage(8)

    assert transports.worker.receive() == QuiescePolicy(8)
    assert transports.worker.receive() == StagePolicyWeights(8)


def test_engine_failure_is_delivered_with_remote_context(expect_error):
    transports = create_in_memory_rollout_transports(capacity=1)
    failure = EngineFailed(
        engine_id="engine-0",
        operation="rollout",
        message="decode failed",
        active_version=7,
        target_version=None,
        lease_id="lease-1",
    )
    transports.worker.publish_failure(failure)

    with expect_error(RemoteRolloutError, "decode failed") as error:
        transports.trainer.receive_result()

    assert error.value.failure == failure


def test_close_wakes_blocked_consumer_and_preserves_already_accepted_work(expect_error):
    transports = create_in_memory_rollout_transports(capacity=1)
    transports.trainer.submit(_lease())
    blocked = Event()
    closed = Event()

    def consume_until_closed():
        assert transports.worker.receive() == _lease()
        blocked.set()
        with expect_error(RolloutTransportClosed, "closed"):
            transports.worker.receive()
        closed.set()

    thread = Thread(target=consume_until_closed)
    thread.start()
    assert blocked.wait(timeout=5)
    transports.trainer.close()
    thread.join(timeout=5)

    assert closed.is_set()
    assert not thread.is_alive()
    with expect_error(RolloutTransportClosed, "closed"):
        transports.trainer.submit(_lease("lease-2"))
    with expect_error(RolloutTransportClosed, "closed"):
        transports.trainer.receive_result()


def test_empty_queue_timeout_uses_standard_empty_exception(expect_error):
    transports = create_in_memory_rollout_transports(capacity=1)

    with expect_error(Empty):
        transports.worker.receive(timeout=0)
