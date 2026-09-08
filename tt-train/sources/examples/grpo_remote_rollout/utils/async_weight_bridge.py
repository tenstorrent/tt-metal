# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Single-slot asynchronous host bridge for versioned model weights."""

from __future__ import annotations

import json
import struct
import threading
from dataclasses import dataclass
from typing import Any, Optional

import ttnn

from .weight_bridge import (
    RECEIVER_RANK,
    SENDER_RANK,
    _ROLE_RECEIVER,
    _ROLE_SENDER,
    _check_role_rank,
    _device0_to_host,
    _replicate_from_host,
    _require_distributed_context,
    _torch_load_bytes,
    _torch_save_bytes,
    _validate_source_tensor,
)

_HELLO = b"async-weight-v1"
_HELLO_SEND = 0
_HELLO_REPLY = 1
_VERSION = 2
_MANIFEST_LEN = 3
_MANIFEST = 4
_TENSOR_LEN = 5
_TENSOR = 6
_CLOSE_VERSION = -1


@dataclass(frozen=True)
class _SerializedWeights:
    version: int
    keys: tuple[str, ...]
    blobs: tuple[bytes, ...]


@dataclass(frozen=True)
class ReceivedWeights:
    version: int
    host_weights: dict[str, Any]


class AsyncHostWeightBridge:
    """Snapshot on the caller thread; move bytes on one private MPI thread.

    The sender pad is bounded to one snapshot and applies backpressure.  The
    receiver pad is also one slot but uses latest-wins replacement, allowing
    inference to skip obsolete intermediate policies when training is faster.
    """

    def __init__(self, *, role: str, peer_rank: int, targets: Optional[list[Any]] = None) -> None:
        local_rank = _require_distributed_context(type(self).__name__)
        _check_role_rank(type(self).__name__, role, local_rank)
        self._role = role
        self._peer_rank = int(peer_rank)
        self._targets = list(targets or [])
        if role == _ROLE_RECEIVER and not self._targets:
            raise ValueError("receiver requires at least one target submesh")
        self._ctx: Any = None
        self._slot: Any = None
        self._condition = threading.Condition()
        self._closing = False
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    @classmethod
    def init_sender(cls, *, peer_rank: int) -> "AsyncHostWeightBridge":
        if int(peer_rank) != RECEIVER_RANK:
            raise ValueError(f"sender peer must be rank {RECEIVER_RANK}")
        return cls(role=_ROLE_SENDER, peer_rank=peer_rank)

    @classmethod
    def init_receiver(cls, *, peer_rank: int, submeshes: list[Any]) -> "AsyncHostWeightBridge":
        if int(peer_rank) != SENDER_RANK:
            raise ValueError(f"receiver peer must be rank {SENDER_RANK}")
        return cls(role=_ROLE_RECEIVER, peer_rank=peer_rank, targets=submeshes)

    def connect(self) -> None:
        self._ctx = ttnn.distributed_context_duplicate()
        if self._role == _ROLE_SENDER:
            self._ctx.send(_HELLO, self._peer_rank, _HELLO_SEND)
            self._ctx.recv(len(_HELLO), self._peer_rank, _HELLO_REPLY)
            target = self._send_loop
        else:
            self._ctx.recv(len(_HELLO), self._peer_rank, _HELLO_SEND)
            self._ctx.send(_HELLO, self._peer_rank, _HELLO_REPLY)
            target = self._receive_loop
        self._thread = threading.Thread(target=target, name=f"async-weight-{self._role}", daemon=True)
        self._thread.start()

    def publish(self, version: int, weights: dict[str, Any]) -> None:
        """Copy a stable full-model snapshot to host and enqueue it."""
        if self._role != _ROLE_SENDER:
            raise RuntimeError("publish is sender-only")
        if version < 0:
            raise ValueError("weight version must be non-negative")
        keys = tuple(sorted(weights))
        if not keys:
            raise ValueError("cannot publish an empty weight dictionary")
        blobs = []
        for key in keys:
            _validate_source_tensor(key, weights[key])
            blobs.append(_torch_save_bytes(_device0_to_host(weights[key])))
        snapshot = _SerializedWeights(version, keys, tuple(blobs))
        with self._condition:
            while self._slot is not None and not self._closing:
                self._condition.wait()
            self._raise_if_failed()
            if self._closing:
                raise RuntimeError("weight bridge is closed")
            self._slot = snapshot
            self._condition.notify_all()

    def poll(self) -> ReceivedWeights | None:
        if self._role != _ROLE_RECEIVER:
            raise RuntimeError("poll is receiver-only")
        with self._condition:
            self._raise_if_failed()
            value = self._slot
            self._slot = None
            self._condition.notify_all()
            return value

    def wait(self) -> ReceivedWeights:
        if self._role != _ROLE_RECEIVER:
            raise RuntimeError("wait is receiver-only")
        with self._condition:
            while self._slot is None and not self._closing and self._error is None:
                self._condition.wait()
            self._raise_if_failed()
            if self._slot is None:
                raise RuntimeError("weight bridge closed before a snapshot arrived")
            value = self._slot
            self._slot = None
            self._condition.notify_all()
            return value

    def materialize(self, received: ReceivedWeights) -> list[dict[str, Any]]:
        if self._role != _ROLE_RECEIVER:
            raise RuntimeError("materialize is receiver-only")
        return [
            {key: _replicate_from_host(tensor, target) for key, tensor in received.host_weights.items()}
            for target in self._targets
        ]

    def close(self) -> None:
        with self._condition:
            if self._closing:
                return
            self._closing = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=60.0)

    def _send_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._slot is None and not self._closing:
                        self._condition.wait()
                    if self._slot is None and self._closing:
                        break
                    value = self._slot
                    self._slot = None
                    self._condition.notify_all()
                self._ctx.send(struct.pack("<q", value.version), self._peer_rank, _VERSION)
                manifest = json.dumps(value.keys, separators=(",", ":")).encode()
                self._ctx.send(struct.pack("<Q", len(manifest)), self._peer_rank, _MANIFEST_LEN)
                self._ctx.send(manifest, self._peer_rank, _MANIFEST)
                for blob in value.blobs:
                    self._ctx.send(struct.pack("<Q", len(blob)), self._peer_rank, _TENSOR_LEN)
                    self._ctx.send(blob, self._peer_rank, _TENSOR)
            self._ctx.send(struct.pack("<q", _CLOSE_VERSION), self._peer_rank, _VERSION)
        except BaseException as error:
            self._record_error(error)

    def _receive_loop(self) -> None:
        try:
            while True:
                version = struct.unpack("<q", self._ctx.recv(8, self._peer_rank, _VERSION))[0]
                if version == _CLOSE_VERSION:
                    with self._condition:
                        self._closing = True
                        self._condition.notify_all()
                    return
                manifest_len = struct.unpack("<Q", self._ctx.recv(8, self._peer_rank, _MANIFEST_LEN))[0]
                keys = json.loads(self._ctx.recv(manifest_len, self._peer_rank, _MANIFEST).decode())
                weights = {}
                for key in keys:
                    blob_len = struct.unpack("<Q", self._ctx.recv(8, self._peer_rank, _TENSOR_LEN))[0]
                    weights[str(key)] = _torch_load_bytes(self._ctx.recv(blob_len, self._peer_rank, _TENSOR))
                with self._condition:
                    self._slot = ReceivedWeights(int(version), weights)
                    self._condition.notify_all()
        except BaseException as error:
            self._record_error(error)

    def _record_error(self, error: BaseException) -> None:
        with self._condition:
            self._error = error
            self._condition.notify_all()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("asynchronous weight bridge failed") from self._error


__all__ = ["AsyncHostWeightBridge", "ReceivedWeights"]
