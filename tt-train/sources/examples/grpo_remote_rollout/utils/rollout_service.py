# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Connect a :class:`RolloutEngine` state machine to rollout queues."""

from __future__ import annotations

from threading import Lock, Thread

from .rollout_engine import EngineEvent, EngineFailed, RolloutEngine
from .rollout_transport import (
    QuiescePolicy,
    RolloutTransportClosed,
    StagePolicyWeights,
    WorkerRolloutTransport,
)


class RolloutWorkerService:
    """Thin process wrapper around one engine and one transport endpoint.

    Construct this service first, pass :meth:`handle_event` as the engine's
    ``event_sink``, then call :meth:`bind_engine`.  Keeping transport mechanics
    outside ``RolloutEngine`` leaves all lifecycle transitions backend-neutral.

    MULTI-WORKER EXTENSION: instantiate one service per worker endpoint.  A
    trainer-side coordinator maps ``engine_id`` to endpoints; this class does
    not require new state transitions.
    """

    def __init__(self, transport: WorkerRolloutTransport) -> None:
        self._transport = transport
        self._engine: RolloutEngine | None = None
        self._failure: EngineFailed | None = None
        self._lock = Lock()
        self._generation_thread: Thread | None = None
        self._staging_thread: Thread | None = None

    def bind_engine(self, engine: RolloutEngine) -> None:
        with self._lock:
            if self._engine is not None:
                raise RuntimeError("rollout worker service already has an engine")
            self._engine = engine

    def handle_event(self, event: EngineEvent) -> None:
        """Engine event sink: route all events to the trainer in emission order."""
        if isinstance(event, EngineFailed):
            with self._lock:
                self._failure = event
        self._transport.publish_event(event)

    def serve_forever(self) -> None:
        engine = self._require_engine()
        try:
            while True:
                try:
                    command = self._transport.receive()
                except RolloutTransportClosed:
                    return
                if isinstance(command, QuiescePolicy):
                    engine.quiesce(command.target_version)
                elif isinstance(command, StagePolicyWeights):
                    self._join_action(self._staging_thread)
                    self._staging_thread = self._start_action(
                        engine.stage_weights,
                        command.version,
                        None,
                        name="rollout-weight-stage",
                    )
                else:
                    # Preserve submission order if the bounded queue is
                    # configured deeper than this first implementation's one
                    # active rollout. Policy commands can still pass an active
                    # generation when no earlier lease is waiting.
                    self._join_action(self._generation_thread)
                    self._generation_thread = self._start_action(
                        engine.start_rollout,
                        command,
                        name="rollout-generation",
                    )
                with self._lock:
                    failure = self._failure
                if failure is not None:
                    raise RuntimeError(
                        f"engine {failure.engine_id} failed during {failure.operation}: {failure.message}"
                    )
        finally:
            self._join_action(self._generation_thread)
            self._join_action(self._staging_thread)
            self._transport.close()

    def _require_engine(self) -> RolloutEngine:
        with self._lock:
            if self._engine is None:
                raise RuntimeError("bind an engine before serving rollouts")
            return self._engine

    def _start_action(self, action, *args, name: str) -> Thread:
        def run() -> None:
            try:
                action(*args)
            except Exception as error:
                # Backend action failures normally emit EngineFailed before
                # re-raising. Convert validation/dispatch failures too, rather
                # than silently losing an exception on this service thread.
                with self._lock:
                    reported = self._failure is not None
                if not reported:
                    snapshot = self._require_engine().snapshot()
                    self.handle_event(
                        EngineFailed(
                            engine_id=snapshot.engine_id,
                            operation=name,
                            message=str(error),
                            active_version=snapshot.active_version,
                            target_version=snapshot.target_version,
                            lease_id=snapshot.active_lease_id,
                        )
                    )

        thread = Thread(target=run, name=name)
        thread.start()
        return thread

    @staticmethod
    def _join_action(thread: Thread | None) -> None:
        if thread is not None:
            thread.join()


__all__ = ["RolloutWorkerService"]
