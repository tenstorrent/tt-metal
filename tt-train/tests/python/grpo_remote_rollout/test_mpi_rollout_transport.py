# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-free tests for MPI framing and progress-thread behavior."""

from __future__ import annotations

from collections import defaultdict
from queue import Queue
from threading import Event, Thread
from unittest.mock import Mock

import pytest

from utils.mpi_rollout_transport import (
    MPIRolloutTrainerTransport,
    MPIRolloutWorkerTransport,
    _decode_failure,
    _decode_result,
    _encode_failure,
    _encode_result,
)
from utils.rollout_engine import EngineFailed, PromptGroupLease, RolloutOutput, RolloutResult
from utils.rollout_service import RolloutWorkerService
from utils.ttt_rollout_engine import TttRolloutEngine


class _Channel:
    def __init__(self, incoming, outgoing):
        self._incoming = incoming
        self._outgoing = outgoing

    def send(self, data, _peer_rank, tag):
        self._outgoing[tag].put(bytes(data))

    def recv(self, size, _peer_rank, tag):
        data = self._incoming[tag].get(timeout=5)
        assert len(data) == size
        return data


def _channels():
    trainer_to_worker = defaultdict(Queue)
    worker_to_trainer = defaultdict(Queue)
    return (
        _Channel(worker_to_trainer, trainer_to_worker),
        _Channel(trainer_to_worker, worker_to_trainer),
    )


def _lease() -> PromptGroupLease:
    return PromptGroupLease(
        lease_id="lease-1",
        group_id="group-1",
        behavior_version=3,
        attempt_id=4,
        payload=[[11, 12], [13]],
    )


def _result() -> RolloutResult:
    return RolloutResult(
        engine_id="engine-0",
        lease_id="lease-1",
        group_id="group-1",
        behavior_version=3,
        attempt_id=4,
        output=RolloutOutput.from_sequences([[21, 22], [23]], [[-0.1, -0.2], [-0.3]]),
    )


def test_binary_result_codec_round_trips_ragged_host_arrays():
    decoded = _decode_result(_encode_result(_result()))

    assert decoded.engine_id == "engine-0"
    assert decoded.lease_id == "lease-1"
    assert decoded.group_id == "group-1"
    assert decoded.behavior_version == 3
    assert decoded.attempt_id == 4
    assert decoded.output.tokens == ((21, 22), (23,))
    for actual, expected in zip(decoded.output.logprobs, ((-0.1, -0.2), (-0.3,))):
        assert actual == pytest.approx(expected)


def test_failure_codec_preserves_remote_context():
    failure = EngineFailed(
        engine_id="engine-0",
        operation="rollout",
        message="decode failed",
        active_version=3,
        target_version=4,
        lease_id="lease-1",
    )

    assert _decode_failure(_encode_failure(failure)) == failure


def test_progress_threads_connect_transport_to_rollout_engine():
    trainer_channel, worker_channel = _channels()
    trainer = MPIRolloutTrainerTransport(peer_rank=1, capacity=2, channel=trainer_channel)
    worker_transport = MPIRolloutWorkerTransport(peer_rank=0, capacity=2, channel=worker_channel)
    service = RolloutWorkerService(worker_transport)
    generation_worker = Mock()
    generation_worker.models = [object()]
    generation_started = Event()
    finish_generation = Event()

    def generate(*_args, **_kwargs):
        generation_started.set()
        assert finish_generation.wait(timeout=5)
        return _result().output

    generation_worker.generate.side_effect = generate
    weight_bridge = Mock()
    staging_started = Event()
    received_weights = [{"weight": object()}]

    def receive_weights():
        staging_started.set()
        return received_weights

    weight_bridge.receive_weights.side_effect = receive_weights
    engine = TttRolloutEngine(
        engine_id="engine-0",
        active_version=3,
        worker=generation_worker,
        weight_bridge=weight_bridge,
        event_sink=service.handle_event,
    )
    service.bind_engine(engine)
    worker_transport.start()
    trainer.start()
    service_thread = Thread(target=service.serve_forever)
    service_thread.start()

    trainer.submit(_lease())
    assert generation_started.wait(timeout=5)
    trainer.quiesce(4)
    trainer.request_weight_stage(4, timeout=5)
    assert staging_started.wait(timeout=5)
    generation_worker.update_weights.assert_not_called()
    finish_generation.set()

    assert trainer.receive_result(timeout=5) == _decode_result(_encode_result(_result()))
    generation_worker.generate.assert_called_once_with(
        [[11, 12], [13]],
        max_new_tokens=128,
        enable_trace=True,
        stop_at_eos=True,
    )

    trainer.close()
    service_thread.join(timeout=5)
    assert not service_thread.is_alive()
    weight_bridge.receive_weights.assert_called_once_with()
    weight_bridge.barrier.assert_called_once_with()
    generation_worker.update_weights.assert_called_once_with(received_weights)
    assert engine.snapshot().active_version == 4
