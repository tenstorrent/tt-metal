# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Threaded cross-rank weight transport (``ThreadedWeightBridge``) on a private MPI communicator."""

from __future__ import annotations

import json
import struct
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import ttnn

from ..distributed import exit_on_error
from .weight_bridge import (
    _shape_to_list,
    _torch_load_bytes,
    _torch_save_bytes,
    _validate_source_tensor,
    duplicate_distributed_context,
)

# Tags are matched only within this bridge's own communicator, so they cannot collide with other MPI traffic.
_TAG_MANIFEST_LEN, _TAG_MANIFEST_BODY, _TAG_WEIGHT_LEN, _TAG_WEIGHT_BODY = 0, 1, 2, 3


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
        self._pads: Optional[Dict[str, ttnn.Tensor]] = None
        self._recv_pads: Optional[List[Dict[str, ttnn.Tensor]]] = None

    @classmethod
    def sender(cls, *, peer_rank: int, mesh_device: Any) -> "ThreadedWeightBridge":
        return cls(peer_rank=peer_rank, mesh_device=mesh_device, submeshes=None)

    @classmethod
    def receiver(cls, *, peer_rank: int, mesh_device: Any, submeshes: List[Any]) -> "ThreadedWeightBridge":
        if not submeshes:
            raise ValueError("ThreadedWeightBridge.receiver needs a non-empty submeshes list")
        return cls(peer_rank=peer_rank, mesh_device=mesh_device, submeshes=list(submeshes))

    def connect(self) -> None:
        self._ctx = duplicate_distributed_context()
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
            self._ctx.send(struct.pack("<Q", 0), self._peer_rank, _TAG_MANIFEST_LEN)

    def _wait_pending_or_closed(self) -> bool:
        with self._cv:
            while not self._pending and not self._closed:
                self._cv.wait()
            return self._pending

    def _send_pads(self) -> None:
        keys = sorted(self._pads)
        manifest = {
            "weight_version": self._version,
            "entries": [
                {
                    "key": k,
                    "shape": _shape_to_list(self._pads[k].shape),
                    "dtype": self._pads[k].dtype.name,
                    "layout": self._pads[k].layout.name,
                }
                for k in keys
            ],
        }
        self._send_blob(json.dumps(manifest).encode("utf-8"), _TAG_MANIFEST_LEN, _TAG_MANIFEST_BODY)
        for k in keys:
            blob = _torch_save_bytes(ttnn.to_torch(self._pads[k], cq_id=1))
            self._send_blob(blob, _TAG_WEIGHT_LEN, _TAG_WEIGHT_BODY)

    def _send_blob(self, blob: bytes, len_tag: int, body_tag: int) -> None:
        self._ctx.send(struct.pack("<Q", len(blob)), self._peer_rank, len_tag)
        self._ctx.send(blob, self._peer_rank, body_tag)

    def _receiver_loop(self) -> None:
        with exit_on_error():
            while (body := self._recv_blob(_TAG_MANIFEST_LEN, _TAG_MANIFEST_BODY)) is not None:
                manifest = json.loads(body.decode("utf-8"))
                with self._cv:
                    while self._pending:
                        self._cv.wait()
                self._recv_into_pads(manifest["entries"])
                with self._cv:
                    self._events = [ttnn.record_event(submesh, 1) for submesh in self._submeshes]
                    self._version, self._pending = int(manifest["weight_version"]), True
                    self._cv.notify_all()
            with self._cv:
                self._peer_closed = True
                self._cv.notify_all()

    def _recv_blob(self, len_tag: int, body_tag: int) -> Optional[bytes]:
        (n,) = struct.unpack("<Q", self._ctx.recv(8, self._peer_rank, len_tag))
        return self._ctx.recv(int(n), self._peer_rank, body_tag) if n else None

    def _recv_into_pads(self, entries: List[dict]) -> None:
        if self._recv_pads is None:
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
        for e in entries:
            host = ttnn.from_torch(
                _torch_load_bytes(self._recv_blob(_TAG_WEIGHT_LEN, _TAG_WEIGHT_BODY)),
                dtype=getattr(ttnn.DataType, e["dtype"]),
                layout=getattr(ttnn.Layout, e["layout"]),
            )
            for pads in self._recv_pads:
                ttnn.copy_host_to_device_tensor(host, pads[e["key"]], cq_id=1)
