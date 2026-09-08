# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bounded asynchronous rollout queues over TT-Metal's MPI context.

Only host ``bytes`` cross this boundary.  One progress thread owns each
blocking direction because the current Python bindings expose blocking
``send_bytes``/``recv_bytes`` calls.  TT-Metal initializes MPI with
``MPI_THREAD_MULTIPLE``; no rollout-engine or device-worker thread calls MPI.

Weights deliberately remain on their existing :class:`WeightBridge` lane.
They are not serialized into these comparatively small prompt/result frames.
"""

from __future__ import annotations

import json
import struct
from collections import deque
from dataclasses import dataclass
from threading import BoundedSemaphore, Event, Lock, Thread
from time import monotonic
from typing import Final, Protocol

from .rollout_engine import (
    EngineEvent,
    EngineFailed,
    PolicyActivated,
    PolicyVersion,
    PromptGroupLease,
    ResultReady,
    RolloutOutput,
    RolloutResult,
    WeightsStaged,
)
from .rollout_transport import (
    QuiescePolicy,
    RemoteRolloutError,
    RolloutCommand,
    RolloutTransportClosed,
    StagePolicyWeights,
    TrainerRolloutTransport,
    WorkerRolloutTransport,
    _ClosableQueue,
)

TTML_RANK: Final = 0
TTT_RANK: Final = 1


_REQUEST_HEADER_TAG: Final = 22200
_REQUEST_BODY_TAG: Final = 22201
_RESULT_HEADER_TAG: Final = 22202
_RESULT_BODY_TAG: Final = 22203

_FRAME_FORMAT: Final = "<4sBBHQ"
_FRAME_SIZE: Final = struct.calcsize(_FRAME_FORMAT)
_MAGIC: Final = b"TRLQ"
_VERSION: Final = 1
_REQUEST: Final = 1
_RESULT: Final = 2
_CLOSE: Final = 3
_CLOSE_ACK: Final = 4
_QUIESCE: Final = 5
_STAGE_WEIGHTS: Final = 6
_FAILURE: Final = 7
_WEIGHTS_STAGED: Final = 8
_POLICY_ACTIVATED: Final = 9
_RESULT_ACK: Final = 10


class _ByteChannel(Protocol):
    def send(self, data: bytes, peer_rank: int, tag: int) -> None:
        ...

    def recv(self, size: int, peer_rank: int, tag: int) -> bytes:
        ...


class _TtnnByteChannel:
    def send(self, data: bytes, peer_rank: int, tag: int) -> None:
        from ttnn import distributed_context_send_bytes

        distributed_context_send_bytes(data, peer_rank, tag)

    def recv(self, size: int, peer_rank: int, tag: int) -> bytes:
        from ttnn import distributed_context_recv_bytes

        return distributed_context_recv_bytes(size, peer_rank, tag)


def _require_distributed_context(owner: str) -> int:
    import ttnn

    if not ttnn.distributed_context_is_initialized():
        raise RuntimeError(f"{owner}: ttnn distributed context is not initialized")
    return int(ttnn.distributed_context_get_rank())


@dataclass(frozen=True)
class _Frame:
    kind: int
    body: bytes


@dataclass(frozen=True)
class _PendingCommand:
    command: RolloutCommand | _ResultAck
    sent: Event


@dataclass(frozen=True)
class _ResultAck:
    """Credit returned only after the trainer consumes a completed rollout."""


def _send_frame(channel: _ByteChannel, peer_rank: int, header_tag: int, body_tag: int, frame: _Frame) -> None:
    header = struct.pack(_FRAME_FORMAT, _MAGIC, _VERSION, frame.kind, 0, len(frame.body))
    channel.send(header, peer_rank, header_tag)
    if frame.body:
        channel.send(frame.body, peer_rank, body_tag)


def _receive_frame(channel: _ByteChannel, peer_rank: int, header_tag: int, body_tag: int) -> _Frame:
    header = channel.recv(_FRAME_SIZE, peer_rank, header_tag)
    magic, version, kind, _reserved, body_size = struct.unpack(_FRAME_FORMAT, header)
    if magic != _MAGIC:
        raise RuntimeError(f"invalid rollout frame magic {magic!r}")
    if version != _VERSION:
        raise RuntimeError(f"unsupported rollout protocol version {version}")
    body = channel.recv(body_size, peer_rank, body_tag) if body_size else b""
    return _Frame(kind, body)


def _encode_lease(lease: PromptGroupLease) -> bytes:
    return json.dumps(
        {
            "lease_id": lease.lease_id,
            "group_id": lease.group_id,
            "behavior_version": lease.behavior_version,
            "attempt_id": lease.attempt_id,
            "payload": lease.payload,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _decode_lease(body: bytes) -> PromptGroupLease:
    value = json.loads(body.decode("utf-8"))
    return PromptGroupLease(
        lease_id=str(value["lease_id"]),
        group_id=str(value["group_id"]),
        behavior_version=int(value["behavior_version"]),
        attempt_id=int(value["attempt_id"]),
        payload=value["payload"],
    )


def _encode_result(result: RolloutResult) -> bytes:
    row_lengths = [len(row) for row in result.output.tokens]
    token_count = sum(row_lengths)
    metadata = json.dumps(
        {
            "engine_id": result.engine_id,
            "lease_id": result.lease_id,
            "group_id": result.group_id,
            "behavior_version": result.behavior_version,
            "attempt_id": result.attempt_id,
            "row_lengths": row_lengths,
            "request_payload": result.request_payload,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    tokens = (token for row in result.output.tokens for token in row)
    logprobs = (logprob for row in result.output.logprobs for logprob in row)
    token_bytes = struct.pack(f"<{token_count}i", *tokens)
    logprob_bytes = struct.pack(f"<{token_count}f", *logprobs)
    return struct.pack("<I", len(metadata)) + metadata + token_bytes + logprob_bytes


def _decode_result(body: bytes) -> RolloutResult:
    if len(body) < 4:
        raise RuntimeError("rollout result frame is missing its metadata length")
    metadata_size = struct.unpack_from("<I", body)[0]
    metadata_end = 4 + metadata_size
    if metadata_end > len(body):
        raise RuntimeError("rollout result metadata exceeds frame length")
    metadata = json.loads(body[4:metadata_end].decode("utf-8"))
    row_lengths = [int(length) for length in metadata["row_lengths"]]
    if any(length < 0 for length in row_lengths):
        raise RuntimeError("rollout result contains a negative row length")
    token_count = sum(row_lengths)
    arrays_size = token_count * 8
    if len(body) - metadata_end != arrays_size:
        raise RuntimeError("rollout result token/logprob arrays do not match row lengths")
    tokens_flat = struct.unpack_from(f"<{token_count}i", body, metadata_end)
    logprobs_flat = struct.unpack_from(f"<{token_count}f", body, metadata_end + token_count * 4)
    tokens: list[tuple[int, ...]] = []
    logprobs: list[tuple[float, ...]] = []
    offset = 0
    for length in row_lengths:
        tokens.append(tokens_flat[offset : offset + length])
        logprobs.append(logprobs_flat[offset : offset + length])
        offset += length
    return RolloutResult(
        engine_id=str(metadata["engine_id"]),
        lease_id=str(metadata["lease_id"]),
        group_id=str(metadata["group_id"]),
        behavior_version=int(metadata["behavior_version"]),
        attempt_id=int(metadata["attempt_id"]),
        output=RolloutOutput(tuple(tokens), tuple(logprobs)),
        request_payload=metadata.get("request_payload"),
    )


def _encode_failure(failure: EngineFailed) -> bytes:
    return json.dumps(
        {
            "engine_id": failure.engine_id,
            "operation": failure.operation,
            "message": failure.message,
            "active_version": failure.active_version,
            "target_version": failure.target_version,
            "lease_id": failure.lease_id,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _decode_failure(body: bytes) -> EngineFailed:
    value = json.loads(body.decode("utf-8"))
    return EngineFailed(
        engine_id=str(value["engine_id"]),
        operation=str(value["operation"]),
        message=str(value["message"]),
        active_version=int(value["active_version"]),
        target_version=int(value["target_version"]) if value["target_version"] is not None else None,
        lease_id=str(value["lease_id"]) if value["lease_id"] is not None else None,
    )


def _encode_lifecycle_event(event: WeightsStaged | PolicyActivated) -> bytes:
    if isinstance(event, WeightsStaged):
        kind = "weights_staged"
    elif isinstance(event, PolicyActivated):
        kind = "policy_activated"
    else:
        raise TypeError(f"unsupported rollout lifecycle event {type(event).__name__}")
    return json.dumps(
        {"kind": kind, "engine_id": event.engine_id, "version": event.version},
        separators=(",", ":"),
    ).encode("utf-8")


def _decode_lifecycle_event(body: bytes) -> WeightsStaged | PolicyActivated:
    value = json.loads(body.decode("utf-8"))
    kind = value["kind"]
    if kind == "weights_staged":
        event_type = WeightsStaged
    elif kind == "policy_activated":
        event_type = PolicyActivated
    else:
        raise RuntimeError(f"unexpected rollout lifecycle event kind {kind!r}")
    return event_type(engine_id=str(value["engine_id"]), version=int(value["version"]))


class _ProgressFailure:
    def __init__(self) -> None:
        self._error: BaseException | None = None
        self._lock = Lock()

    def record(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error

    def raise_if_set(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError("rollout MPI progress thread failed") from error


class MPIRolloutTrainerTransport(TrainerRolloutTransport):
    """Trainer endpoint with bounded submission and result queues."""

    def __init__(self, *, peer_rank: int, capacity: int = 1, channel: _ByteChannel | None = None) -> None:
        if channel is None:
            local_rank = _require_distributed_context(type(self).__name__)
            if local_rank != TTML_RANK or int(peer_rank) != TTT_RANK:
                raise RuntimeError("trainer rollout transport must connect TTML_RANK to TTT_RANK")
        self._peer_rank = int(peer_rank)
        self._channel = channel or _TtnnByteChannel()
        self._outbound = _ClosableQueue[_PendingCommand](capacity)
        self._results = _ClosableQueue[EngineEvent](capacity)
        self._deferred_events: deque[EngineEvent] = deque()
        self._failure = _ProgressFailure()
        self._started = False
        self._sender = Thread(target=self._send_loop, name="rollout-mpi-request", daemon=True)
        self._receiver = Thread(target=self._receive_loop, name="rollout-mpi-result", daemon=True)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._receiver.start()
        self._sender.start()

    def submit(self, lease: PromptGroupLease, *, timeout: float | None = None) -> None:
        self._enqueue(lease, timeout)

    def quiesce(self, target_version: PolicyVersion, *, timeout: float | None = None) -> None:
        self._enqueue(QuiescePolicy(target_version), timeout)

    def request_weight_stage(self, version: PolicyVersion, *, timeout: float | None = None) -> None:
        """Ensure staging intent is on the wire before weight transfer starts."""
        started = monotonic()
        pending = self._enqueue(StagePolicyWeights(version), timeout)
        while not pending.sent.wait(0.05):
            self._failure.raise_if_set()
            if timeout is not None and monotonic() - started >= timeout:
                raise TimeoutError("timed out waiting to send the weight-stage command")
        self._failure.raise_if_set()

    def _enqueue(self, command: RolloutCommand, timeout: float | None) -> _PendingCommand:
        self._failure.raise_if_set()
        if not self._started:
            raise RuntimeError("call start() before submitting rollouts")
        pending = _PendingCommand(command, Event())
        self._outbound.put(pending, timeout)
        return pending

    def receive_event(self, *, timeout: float | None = None) -> EngineEvent:
        if self._deferred_events:
            return self._deferred_events.popleft()
        return self._receive_raw_event(timeout)

    def _receive_raw_event(self, timeout: float | None) -> EngineEvent:
        self._failure.raise_if_set()
        try:
            event = self._results.get(timeout)
            if isinstance(event, EngineFailed):
                raise RemoteRolloutError(event)
            return event
        except RolloutTransportClosed:
            self._failure.raise_if_set()
            raise

    def receive_result(self, *, timeout: float | None = None) -> RolloutResult:
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - monotonic())
            event = self._receive_raw_event(remaining)
            if isinstance(event, ResultReady):
                # Returning the credit here, rather than when the MPI receiver
                # buffers the frame, makes capacity an end-to-end bound on
                # completed rollouts (including data eagerly buffered by MPI).
                self._enqueue(_ResultAck(), remaining)
                return event.result
            self._deferred_events.append(event)

    def close(self) -> None:
        if not self._started:
            self._outbound.close()
            self._results.close()
            return
        self._outbound.close()
        self._sender.join()
        self._receiver.join()
        self._failure.raise_if_set()

    def _send_loop(self) -> None:
        try:
            while True:
                try:
                    pending = self._outbound.get(None)
                except RolloutTransportClosed:
                    break
                command = pending.command
                if isinstance(command, PromptGroupLease):
                    frame = _Frame(_REQUEST, _encode_lease(command))
                elif isinstance(command, QuiescePolicy):
                    frame = _Frame(_QUIESCE, struct.pack("<q", command.target_version))
                elif isinstance(command, StagePolicyWeights):
                    frame = _Frame(_STAGE_WEIGHTS, struct.pack("<q", command.version))
                else:
                    frame = _Frame(_RESULT_ACK, b"")
                _send_frame(
                    self._channel,
                    self._peer_rank,
                    _REQUEST_HEADER_TAG,
                    _REQUEST_BODY_TAG,
                    frame,
                )
                pending.sent.set()
            _send_frame(
                self._channel,
                self._peer_rank,
                _REQUEST_HEADER_TAG,
                _REQUEST_BODY_TAG,
                _Frame(_CLOSE, b""),
            )
        except BaseException as error:
            self._failure.record(error)
            if "pending" in locals():
                pending.sent.set()
            self._results.close()

    def _receive_loop(self) -> None:
        try:
            while True:
                frame = _receive_frame(self._channel, self._peer_rank, _RESULT_HEADER_TAG, _RESULT_BODY_TAG)
                if frame.kind == _CLOSE_ACK:
                    self._results.close()
                    return
                if frame.kind == _RESULT:
                    value: EngineEvent = ResultReady(_decode_result(frame.body))
                elif frame.kind == _FAILURE:
                    value = _decode_failure(frame.body)
                elif frame.kind in (_WEIGHTS_STAGED, _POLICY_ACTIVATED):
                    value = _decode_lifecycle_event(frame.body)
                    expected_type = WeightsStaged if frame.kind == _WEIGHTS_STAGED else PolicyActivated
                    if not isinstance(value, expected_type):
                        raise RuntimeError("rollout lifecycle frame kind does not match its body")
                else:
                    raise RuntimeError(f"unexpected rollout result frame kind {frame.kind}")
                self._results.put(value, None)
        except BaseException as error:
            self._failure.record(error)
            self._results.close()


class MPIRolloutWorkerTransport(WorkerRolloutTransport):
    """Rollout-rank endpoint; device work remains outside its progress threads."""

    def __init__(self, *, peer_rank: int, capacity: int = 1, channel: _ByteChannel | None = None) -> None:
        if channel is None:
            local_rank = _require_distributed_context(type(self).__name__)
            if local_rank != TTT_RANK or int(peer_rank) != TTML_RANK:
                raise RuntimeError("worker rollout transport must connect TTT_RANK to TTML_RANK")
        self._peer_rank = int(peer_rank)
        self._channel = channel or _TtnnByteChannel()
        self._requests = _ClosableQueue[RolloutCommand](capacity)
        self._outbound = _ClosableQueue[EngineEvent](capacity)
        self._result_credits = BoundedSemaphore(capacity)
        self._failure = _ProgressFailure()
        self._started = False
        self._receiver = Thread(target=self._receive_loop, name="rollout-mpi-request", daemon=True)
        self._sender = Thread(target=self._send_loop, name="rollout-mpi-result", daemon=True)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._sender.start()
        self._receiver.start()

    def receive(self, *, timeout: float | None = None) -> RolloutCommand:
        self._failure.raise_if_set()
        if not self._started:
            raise RuntimeError("call start() before receiving rollouts")
        try:
            return self._requests.get(timeout)
        except RolloutTransportClosed:
            self._failure.raise_if_set()
            raise

    def publish_event(self, event: EngineEvent, *, timeout: float | None = None) -> None:
        self._failure.raise_if_set()
        owns_credit = isinstance(event, ResultReady)
        if owns_credit and not self._result_credits.acquire(timeout=timeout):
            raise TimeoutError("timed out waiting for completed-rollout queue credit")
        try:
            self._outbound.put(event, timeout)
        except BaseException:
            if owns_credit:
                self._result_credits.release()
            raise

    def publish(self, result: RolloutResult, *, timeout: float | None = None) -> None:
        self.publish_event(ResultReady(result), timeout=timeout)

    def publish_failure(self, failure: EngineFailed) -> None:
        self.publish_event(failure)

    def close(self) -> None:
        self._outbound.close()
        if self._started:
            self._receiver.join()
            self._sender.join()
        self._failure.raise_if_set()

    def _receive_loop(self) -> None:
        try:
            while True:
                frame = _receive_frame(self._channel, self._peer_rank, _REQUEST_HEADER_TAG, _REQUEST_BODY_TAG)
                if frame.kind == _CLOSE:
                    self._requests.close()
                    return
                if frame.kind == _REQUEST:
                    command: RolloutCommand = _decode_lease(frame.body)
                elif frame.kind == _QUIESCE:
                    command = QuiescePolicy(struct.unpack("<q", frame.body)[0])
                elif frame.kind == _STAGE_WEIGHTS:
                    command = StagePolicyWeights(struct.unpack("<q", frame.body)[0])
                elif frame.kind == _RESULT_ACK:
                    self._result_credits.release()
                    continue
                else:
                    raise RuntimeError(f"unexpected rollout request frame kind {frame.kind}")
                self._requests.put(command, None)
        except BaseException as error:
            self._failure.record(error)
            self._requests.close()
            self._outbound.close()

    def _send_loop(self) -> None:
        try:
            while True:
                try:
                    value = self._outbound.get(None)
                except RolloutTransportClosed:
                    break
                if isinstance(value, EngineFailed):
                    frame = _Frame(_FAILURE, _encode_failure(value))
                elif isinstance(value, ResultReady):
                    frame = _Frame(_RESULT, _encode_result(value.result))
                elif isinstance(value, WeightsStaged):
                    frame = _Frame(_WEIGHTS_STAGED, _encode_lifecycle_event(value))
                elif isinstance(value, PolicyActivated):
                    frame = _Frame(_POLICY_ACTIVATED, _encode_lifecycle_event(value))
                else:
                    raise TypeError(f"unsupported rollout event {type(value).__name__}")
                _send_frame(
                    self._channel,
                    self._peer_rank,
                    _RESULT_HEADER_TAG,
                    _RESULT_BODY_TAG,
                    frame,
                )
            _send_frame(
                self._channel,
                self._peer_rank,
                _RESULT_HEADER_TAG,
                _RESULT_BODY_TAG,
                _Frame(_CLOSE_ACK, b""),
            )
        except BaseException as error:
            self._failure.record(error)
            self._requests.close()


__all__ = ["MPIRolloutTrainerTransport", "MPIRolloutWorkerTransport"]
