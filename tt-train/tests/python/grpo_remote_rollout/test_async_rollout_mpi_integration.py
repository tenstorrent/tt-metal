# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Two-rank integration test for the asynchronous rollout control plane.

This test uses the real MPI rollout transports and ``HostWeightBridge`` with
small device tensors.  Generation remains deterministic and lightweight so
the test validates orchestration rather than model accuracy.
"""

from __future__ import annotations

import gc
import os
import time
from threading import Event, Lock

import pytest

_WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
if _WORLD_SIZE != 2:
    pytest.skip(
        "test_async_rollout_mpi_integration must run under tt-run with world_size == 2 "
        "(use runner_async_rollout_mpi_integration.sh)",
        allow_module_level=True,
    )

_MPI_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])

import torch  # noqa: E402
import ttnn  # noqa: E402

from utils.mpi_rollout_transport import MPIRolloutTrainerTransport, MPIRolloutWorkerTransport  # noqa: E402
from utils.rollout_coordinator import SingleWorkerRolloutCoordinator  # noqa: E402
from utils.rollout_engine import PromptGroupLease, RolloutOutput  # noqa: E402
from utils.rollout_service import RolloutWorkerService  # noqa: E402
from utils.ttt_rollout_engine import TttRolloutEngine  # noqa: E402
from utils.weight_bridge import RECEIVER_RANK, SENDER_RANK, HostWeightBridge  # noqa: E402

ENGINE_ID = "mpi-integration-engine"
INITIAL_VERSION = 0
UPDATED_VERSION = 1
MESH_SHAPE = (1, 1)


def _ensure_distributed_context() -> None:
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()


def _open_mesh():
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE), offset=ttnn.MeshCoordinate(0, 0))


def _version_weights(mesh, version: int) -> dict[str, "ttnn.Tensor"]:
    host = torch.full((1, 1, 32, 32), float(version), dtype=torch.bfloat16)
    return {
        "policy.version": ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
        )
    }


def _expected_output(version: int, batch_size: int) -> RolloutOutput:
    tokens = [[100 * version + 10 * row + 1, 100 * version + 10 * row + 2] for row in range(batch_size)]
    logprobs = [[-(version + 1.0), -(version + 1.5)] for _ in range(batch_size)]
    return RolloutOutput.from_sequences(tokens, logprobs)


class _DeterministicGenerationWorker:
    """Minimal generation target whose behavior changes with received weights."""

    def __init__(self) -> None:
        self.models = [object()]
        self.generation_inflight = Event()
        self._version = INITIAL_VERSION
        self._generation_count = 0
        self._lock = Lock()

    def generate(self, prompts, *, max_new_tokens: int, enable_trace: bool, stop_at_eos: bool) -> RolloutOutput:
        assert max_new_tokens == 2
        assert enable_trace is False
        assert stop_at_eos is True
        with self._lock:
            version = self._version
            generation_index = self._generation_count
            self._generation_count += 1

        self.generation_inflight.set()
        try:
            # Leave enough time for the stage command and small host transfer
            # to overlap the first rollout on an unloaded CI node.
            if generation_index == 0:
                time.sleep(2)
            return _expected_output(version, len(prompts))
        finally:
            self.generation_inflight.clear()

    def update_weights(self, per_submesh) -> None:
        assert not self.generation_inflight.is_set(), "policy activated before the old rollout drained"
        assert len(per_submesh) == 1
        tensor = per_submesh[0]["policy.version"]
        host = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
        version = int(host.flatten()[0].item())
        assert torch.all(host == version)
        with self._lock:
            self._version = version


class _RecordingReceiverBridge:
    def __init__(self, bridge: HostWeightBridge, generation_inflight: Event) -> None:
        self._bridge = bridge
        self._generation_inflight = generation_inflight
        self.received_during_generation = False

    def receive_weights(self):
        weights = self._bridge.receive_weights()
        self.received_during_generation = self._generation_inflight.is_set()
        return weights

    def barrier(self) -> None:
        self._bridge.barrier()


def _lease(lease_id: str, version: int) -> PromptGroupLease:
    return PromptGroupLease(
        lease_id=lease_id,
        group_id=f"group-{lease_id}",
        behavior_version=version,
        attempt_id=0,
        payload=[[11, 12], [21]],
    )


def _trainer_side(mesh) -> None:
    bridge = HostWeightBridge.init_sender(mesh=mesh, peer_rank=RECEIVER_RANK)
    transport = MPIRolloutTrainerTransport(peer_rank=RECEIVER_RANK, capacity=2)
    bridge.connect()
    transport.start()
    coordinator = SingleWorkerRolloutCoordinator(
        engine_id=ENGINE_ID,
        active_version=INITIAL_VERSION,
        transport=transport,
        weight_bridge=bridge,
    )

    first = _lease("lease-0", INITIAL_VERSION)
    coordinator.submit(first, timeout=10)
    coordinator.begin_policy_cutover(UPDATED_VERSION, _version_weights(mesh, UPDATED_VERSION), timeout=30)
    coordinator.await_policy_activation(UPDATED_VERSION, timeout=30)

    old_result = coordinator.receive_result(timeout=10)
    assert old_result.behavior_version == INITIAL_VERSION
    assert old_result.output == _expected_output(INITIAL_VERSION, 2)

    second = _lease("lease-1", UPDATED_VERSION)
    coordinator.submit(second, timeout=10)
    new_result = coordinator.receive_result(timeout=10)
    assert new_result.behavior_version == UPDATED_VERSION
    assert new_result.output == _expected_output(UPDATED_VERSION, 2)
    assert coordinator.active_version == UPDATED_VERSION
    assert coordinator.outstanding_count == 0
    coordinator.close()


def _worker_side(mesh) -> None:
    worker = _DeterministicGenerationWorker()
    raw_bridge = HostWeightBridge.init_receiver(mesh=mesh, peer_rank=SENDER_RANK, submeshes=[mesh])
    bridge = _RecordingReceiverBridge(raw_bridge, worker.generation_inflight)
    transport = MPIRolloutWorkerTransport(peer_rank=SENDER_RANK, capacity=2)
    service = RolloutWorkerService(transport)
    engine = TttRolloutEngine(
        engine_id=ENGINE_ID,
        active_version=INITIAL_VERSION,
        worker=worker,
        weight_bridge=bridge,
        max_new_tokens=2,
        enable_trace=False,
        event_sink=service.handle_event,
    )
    service.bind_engine(engine)
    raw_bridge.connect()
    transport.start()
    service.serve_forever()

    assert bridge.received_during_generation, "weight receipt did not overlap the old-policy rollout"
    assert engine.snapshot().active_version == UPDATED_VERSION


@pytest.mark.timeout(120)
def test_async_rollout_mpi_cutover() -> None:
    """Exercise rollout, overlapped staging, activation, and clean shutdown."""
    _ensure_distributed_context()
    mesh = _open_mesh()
    try:
        if _MPI_RANK == SENDER_RANK:
            _trainer_side(mesh)
        elif _MPI_RANK == RECEIVER_RANK:
            _worker_side(mesh)
        else:
            raise RuntimeError(f"unexpected MPI rank {_MPI_RANK}")
    finally:
        gc.collect()
        ttnn.close_mesh_device(mesh)
