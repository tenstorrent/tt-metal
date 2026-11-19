# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Threaded cross-rank weight transport (``ThreadedWeightBridge``) on a private MPI communicator."""

from __future__ import annotations

import collections
import json
import threading
from contextlib import contextmanager
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

import ttnn

from ..distributed import exit_on_error
from .transport import duplicate_distributed_context, recv_frame, send_frame
from .weight_bridge import _manifest_entries, _torch_load_bytes, _torch_save_bytes, _validate_source_tensor

# Tags are matched only within this bridge's own communicator, so they cannot collide with other MPI traffic.
_TAG_MANIFEST_LEN, _TAG_MANIFEST_BODY, _TAG_WEIGHT_LEN, _TAG_WEIGHT_BODY = 0, 1, 2, 3


class _FifoLock:
    """A lock that threads get in the order that they ask for it."""

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._waiters: Deque[threading.Lock] = collections.deque()
        self._owner: Optional[threading.Lock] = None

    def acquire(self) -> bool:
        waiter = threading.Lock()
        waiter.acquire()
        try:
            with self._mutex:
                if self._owner is None:
                    self._owner = waiter
                    return True
                self._waiters.append(waiter)
            waiter.acquire()
            return True
        except BaseException:
            with self._mutex:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
            if self._owner is waiter:
                self.release()
            raise

    def release(self) -> None:
        with self._mutex:
            if self._owner is None:
                raise RuntimeError("release unlocked lock")
            if self._waiters:
                # No call between choosing the next owner and waking it, so KeyboardInterrupt cannot fall in between.
                self._owner = self._waiters[0]
                del self._waiters[0]
                self._owner.release()
            else:
                self._owner = None

    def locked(self) -> bool:
        return self._owner is not None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


# Race: while thread 0 captures a trace on CQ0, any command that thread 1 sends on CQ1 goes into the trace
# instead of to the device (https://github.com/tenstorrent/tt-metal/issues/65971). The capture holds this lock;
# the receiver thread takes it around each device call.
_capture_lock = _FifoLock()
_begin_trace_capture, _end_trace_capture = ttnn.begin_trace_capture, ttnn.end_trace_capture


def _begin_trace_capture_locked(*args, **kwargs):
    _capture_lock.acquire()
    try:
        return _begin_trace_capture(*args, **kwargs)
    except BaseException:
        _capture_lock.release()
        raise


def _end_trace_capture_locked(*args, **kwargs):
    try:
        return _end_trace_capture(*args, **kwargs)
    finally:
        _capture_lock.release()


def _lock_trace_capture(enable: bool) -> None:
    ttnn.begin_trace_capture = _begin_trace_capture_locked if enable else _begin_trace_capture
    ttnn.end_trace_capture = _end_trace_capture_locked if enable else _end_trace_capture


class ThreadedWeightBridge:
    """Moves an HF-keyed weight dict between ranks on a background thread over a private communicator.

    The sender copies the weights into on-device pads on CQ0 and returns; its thread reads the pads on CQ1.
    The receiver's thread writes them into pads on each submesh on CQ1 and keeps them until consumed.
    Both meshes need 2 command queues.
    """

    def __init__(self, *, peer_rank: int, mesh_device: Any, submeshes: Optional[List[Any]]) -> None:
        self._peer_rank = int(peer_rank)
        self._mesh = mesh_device
        self._submeshes = submeshes
        self._ctx: Any = None
        self._thread: Optional[threading.Thread] = None
        self._cv = threading.Condition()
        self._pending = False
        self._closed = False
        self._peer_closed = False
        self._version = 0
        self._events: List[Any] = []
        self._read_events: List[Any] = []
        self._pads: Optional[Dict[str, ttnn.Tensor]] = None
        self._recv_pads: Optional[List[Dict[str, ttnn.Tensor]]] = None

    @classmethod
    def sender(cls, *, peer_rank: int, mesh_device: Any) -> "ThreadedWeightBridge":
        return cls(peer_rank=peer_rank, mesh_device=mesh_device, submeshes=None)

    @classmethod
    def receiver(cls, *, peer_rank: int, submeshes: List[Any]) -> "ThreadedWeightBridge":
        if not submeshes:
            raise ValueError("ThreadedWeightBridge.receiver needs a non-empty submeshes list")
        return cls(peer_rank=peer_rank, mesh_device=None, submeshes=list(submeshes))

    def connect(self) -> None:
        self._ctx = duplicate_distributed_context()
        if self._submeshes is not None:
            _lock_trace_capture(True)
        loop = self._sender_loop if self._submeshes is None else self._receiver_loop
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join()
        if self._submeshes is not None:
            _lock_trace_capture(False)

    def send_weights(self, weights: Dict[str, ttnn.Tensor], version: int) -> None:
        for k in sorted(weights):
            _validate_source_tensor(k, weights[k])
        with self._cv:
            if self._closed:
                raise RuntimeError("ThreadedWeightBridge.send_weights called after close")
            while self._pending:
                self._cv.wait()
            self._ensure_send_pads(weights)
            for k in sorted(weights):
                ttnn.copy(weights[k], self._pads[k])
            self._events = [ttnn.record_event(self._mesh, 0)]
            self._version, self._pending = int(version), True
            self._cv.notify_all()

    @contextmanager
    def receive_weights(self, block: bool) -> Iterator[Optional[Tuple[int, List[Dict[str, ttnn.Tensor]]]]]:
        """Yields (version, one dict per submesh), or None if nothing arrived; pads stay untouched until exit."""
        with self._cv:
            while block and not self._pending and not self._peer_closed:
                self._cv.wait()
            if not self._pending:
                yield None
                return
            for event in self._events:
                ttnn.wait_for_event(0, event)
            yield self._version, [dict(pads) for pads in self._recv_pads]
            self._read_events = [ttnn.record_event(submesh, 0) for submesh in self._submeshes]
            self._pending = False
            self._cv.notify_all()

    @property
    def peer_closed(self) -> bool:
        with self._cv:
            return self._peer_closed

    def _ensure_send_pads(self, weights: Dict[str, ttnn.Tensor]) -> None:
        if self._pads is None:
            self._pads = {
                k: ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, self._mesh, t.memory_config())
                for k, t in weights.items()
            }
        elif set(weights) != set(self._pads):
            raise RuntimeError(f"ThreadedWeightBridge: key set changed from {sorted(self._pads)} to {sorted(weights)}")

    def _sender_loop(self) -> None:
        with exit_on_error():
            while self._wait_pending_or_closed():
                ttnn.wait_for_event(1, self._events[0])
                self._send_pads()
                with self._cv:
                    self._pending = False
                    self._cv.notify_all()
            self._send_blob(b"", _TAG_MANIFEST_LEN, _TAG_MANIFEST_BODY)

    def _wait_pending_or_closed(self) -> bool:
        with self._cv:
            while not self._pending and not self._closed:
                self._cv.wait()
            return self._pending

    def _send_pads(self) -> None:
        keys = sorted(self._pads)
        manifest = {"weight_version": self._version, "entries": _manifest_entries(self._pads, keys)}
        self._send_blob(json.dumps(manifest).encode("utf-8"), _TAG_MANIFEST_LEN, _TAG_MANIFEST_BODY)
        for k in keys:
            blob = _torch_save_bytes(ttnn.to_torch(self._pads[k], cq_id=1))
            self._send_blob(blob, _TAG_WEIGHT_LEN, _TAG_WEIGHT_BODY)

    def _send_blob(self, blob: bytes, len_tag: int, body_tag: int) -> None:
        send_frame(self._ctx.send, blob, self._peer_rank, len_tag, body_tag)

    def _receiver_loop(self) -> None:
        with exit_on_error():
            while (body := self._recv_blob(_TAG_MANIFEST_LEN, _TAG_MANIFEST_BODY)) is not None:
                manifest = json.loads(body.decode("utf-8"))
                with self._cv:
                    while self._pending:
                        self._cv.wait()
                self._recv_into_pads(manifest["entries"])
                with _capture_lock:
                    events = [ttnn.record_event(submesh, 1) for submesh in self._submeshes]
                with self._cv:
                    self._events = events
                    self._version, self._pending = int(manifest["weight_version"]), True
                    self._cv.notify_all()
            with self._cv:
                self._peer_closed = True
                self._cv.notify_all()

    def _recv_blob(self, len_tag: int, body_tag: int) -> Optional[bytes]:
        return recv_frame(self._ctx.recv, self._peer_rank, len_tag, body_tag)

    def _recv_into_pads(self, entries: List[dict]) -> None:
        if self._recv_pads is None:
            with _capture_lock:
                self._recv_pads = [
                    {
                        e["key"]: ttnn.allocate_tensor_on_device(
                            ttnn.Shape(e["shape"]),
                            getattr(ttnn.DataType, e["dtype"]),
                            getattr(ttnn.Layout, e["layout"]),
                            submesh,
                            ttnn.DRAM_MEMORY_CONFIG,
                        )
                        for e in entries
                    }
                    for submesh in self._submeshes
                ]
        with _capture_lock:
            for event in self._read_events:
                ttnn.wait_for_event(1, event)
        for e in entries:
            host = ttnn.from_torch(
                _torch_load_bytes(self._recv_blob(_TAG_WEIGHT_LEN, _TAG_WEIGHT_BODY)),
                dtype=getattr(ttnn.DataType, e["dtype"]),
                layout=getattr(ttnn.Layout, e["layout"]),
            )
            for pads in self._recv_pads:
                with _capture_lock:
                    ttnn.copy_host_to_device_tensor(host, pads[e["key"]], cq_id=1)
