# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-free integration tests for the trainer-facing rollout client."""

from __future__ import annotations

from queue import Queue
from threading import Barrier, Event, Thread
from unittest.mock import Mock

from utils.coordinator_rollout_client import CoordinatorRolloutClient
from utils.rollout_coordinator import SingleWorkerRolloutCoordinator
from utils.rollout_engine import RolloutOutput
from utils.rollout_service import RolloutWorkerService
from utils.rollout_transport import create_in_memory_rollout_transports
from utils.ttt_rollout_engine import TttRolloutEngine


class _WeightChannel:
    def __init__(self) -> None:
        self.values: Queue[list[dict[str, object]]] = Queue()
        self.fence = Barrier(2)

    def send_weights(self, weights):
        self.values.put([weights])

    def receive_weights(self):
        return self.values.get()

    def barrier(self):
        self.fence.wait(timeout=1.0)


class _ControlledWorker:
    def __init__(self) -> None:
        self.models = [object()]
        self.generation_started = Event()
        self.finish_generation = Event()
        self.weight_updates: list[list[dict[str, object]]] = []

    def generate(self, prompts, **_kwargs):
        self.generation_started.set()
        assert self.finish_generation.wait(1.0)
        return RolloutOutput.from_sequences([[91, 92] for _ in prompts], [[-0.1, -0.2] for _ in prompts])

    def update_weights(self, per_submesh):
        self.weight_updates.append(per_submesh)


def _running_stack():
    transports = create_in_memory_rollout_transports(capacity=8)
    bridge = _WeightChannel()
    worker = _ControlledWorker()
    service = RolloutWorkerService(transports.worker)
    engine = TttRolloutEngine(
        engine_id="engine-0",
        active_version=0,
        worker=worker,
        weight_bridge=bridge,
        max_new_tokens=2,
        event_sink=service.handle_event,
    )
    service.bind_engine(engine)
    service_thread = Thread(target=service.serve_forever, name="test-rollout-service")
    service_thread.start()
    coordinator = SingleWorkerRolloutCoordinator(
        engine_id="engine-0",
        active_version=0,
        transport=transports.trainer,
        weight_bridge=bridge,
    )
    client = CoordinatorRolloutClient(coordinator=coordinator, max_new_tokens=2)
    return client, worker, service_thread


def test_full_result_survives_weight_staging_during_active_rollout():
    client, worker, service_thread = _running_stack()
    try:
        assert client.send_weights({"weight": "theta-0"}) == 1
        client.submit_remote_generate([[11, 12]], max_new_tokens=2)
        assert worker.generation_started.wait(1.0)

        # Transfer theta-1 while the theta-0 rollout is still active. The
        # engine cannot activate it until generation reaches the safe boundary.
        assert client.begin_weight_update({"weight": "theta-1"}) == 2
        worker.finish_generation.set()
        assert client.await_weight_update() == 2

        result = client.await_rollout()
        assert result.behavior_version == 1
        assert result.output.tokens == ((91, 92),)
        assert result.output.logprobs == ((-0.1, -0.2),)
        assert client.last_result == result
        assert worker.weight_updates == [[{"weight": "theta-0"}], [{"weight": "theta-1"}]]

        client.submit_remote_generate([[21]], max_new_tokens=2)
        assert client.await_remote_generate() == [[91, 92]]
        assert client.last_result is not None
        assert client.last_result.behavior_version == 2
        assert client.last_result.output.logprobs == ((-0.1, -0.2),)
    finally:
        worker.finish_generation.set()
        client.close()
        service_thread.join(timeout=1.0)
    assert not service_thread.is_alive()


def test_client_rejects_overlapping_leases_and_mismatched_generation_config(expect_error):
    coordinator = Mock()
    coordinator.engine_id = "engine-0"
    coordinator.active_version = 3
    client = CoordinatorRolloutClient(coordinator=coordinator, max_new_tokens=2, id_factory=lambda: "lease-1")

    with expect_error(ValueError, "does not match rollout engine"):
        client.submit_remote_generate([[1]], max_new_tokens=3)

    client.submit_remote_generate([[1]], max_new_tokens=2)
    lease = coordinator.submit.call_args.args[0]
    assert lease.lease_id == lease.group_id == "lease-1"
    assert lease.behavior_version == 3
    assert lease.payload == [[1]]
    with expect_error(RuntimeError, "still pending"):
        client.submit_remote_generate([[2]], max_new_tokens=2)


def test_two_phase_weight_update_validates_pending_version(expect_error):
    coordinator = Mock()
    coordinator.engine_id = "engine-0"
    coordinator.active_version = 7
    client = CoordinatorRolloutClient(coordinator=coordinator, max_new_tokens=2)

    assert client.begin_weight_update({"weight": object()}) == 8
    with expect_error(RuntimeError, "already activating"):
        client.begin_weight_update({"weight": object()})
    with expect_error(ValueError, "version 8 is pending"):
        client.await_weight_update(9)

    assert client.await_weight_update() == 8
    coordinator.await_policy_activation.assert_called_once_with(8, timeout=None)
