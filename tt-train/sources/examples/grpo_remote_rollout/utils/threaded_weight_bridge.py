# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Threaded MPI weight bridge with a dict-of-tensors API and lazy on-device pads.

Public API mirrors :class:`WeightBridge` in ``utils/weight_bridge.py``:

  * ``connect()`` / ``close()``
  * ``send_weights(dict[str, ttnn.Tensor])``
  * ``receive_weights()`` -- context manager yielding
    ``List[dict[str, ttnn.Tensor]]``. Holds the recv pad lock across the
    caller's ``with`` block so the bridge cannot overwrite the pads.
  * ``poll_weights()`` -- non-blocking context manager. Yields ``None`` if
    no dict has arrived; behaves like ``receive_weights`` if one has.

Design summary
==============

Same on-device pad + cross-CQ event scheme as the single-tensor version
used to have, but the single pad is now a **dict of pads**, one per key.
Pads are lazy-init:

  * Sender: on the FIRST ``send_weights(weights)`` call, allocate
    ``self._send_pads[k]`` for each ``k`` from the caller's tensor spec.
    Subsequent calls assert the key set + per-key spec is stable, then
    do only ``ttnn.copy(weights[k], self._send_pads[k], queue_id=0)``.
  * Receiver: on the FIRST manifest received, allocate
    ``self._recv_pads[k]`` from the manifest entries. Subsequent
    manifests assert stability.

Freeze point
============

The caller's ``send_weights`` returns as soon as every ``ttnn.copy``
enqueue is done + one ``ttnn.record_event(mesh, 0)`` is recorded.
CQ0 is ordered within itself so one event covers the whole burst. The
caller is then free to mutate its source tensors immediately -- the
bridge thread's ``ttnn.to_torch(pad, cq_id=1)`` reads from the pads,
never from the caller's source tensors.

Wire format
===========

Mirrors :class:`HostWeightBridge` in ``utils/weight_bridge.py`` but on
the bridge's private duplicated MPI context, with dense tag numbers:

  * ``[u64 manifest_len]`` on tag :data:`_TAG_MANIFEST_LEN`.
    ``manifest_len == 0`` is the close message.
  * ``[manifest_bytes]`` on tag :data:`_TAG_MANIFEST_BODY`. JSON body:
    ``{"version": 1, "entries": [{"key", "shape", "dtype", "layout"}, ...]}``.
  * Per key in sorted order:
    ``[u64 blob_len]`` on :data:`_TAG_WEIGHT_LEN`,
    ``[torch.save(host_tensor)]`` on :data:`_TAG_WEIGHT_BODY`.

Concurrency
===========

Send pad lock is held end-to-end for a message: caller thread does the
``ttnn.copy``s and ``record_event`` under it; bridge thread's
``wait_for_event`` + per-key ``to_torch`` + serialize + MPI send all run
under the same lock. The bridge and caller alternate strictly on the
condvar.

Recv pad lock is symmetric: bridge thread's manifest recv + per-key
recv + ``copy_host_to_device_tensor`` + ``record_event`` all run under
it. The caller's ``with receive_weights() as dicts:`` block also holds
the same lock, so the bridge cannot overwrite the pads until the caller
exits the ``with``.

Relies on the GIL-release wrap on ``DistributedContext`` methods in
``tt-train/sources/ttml/nanobind/nb_core.cpp``; without it the bridge
thread's blocking MPI call would hold both the GIL and the pad lock and
starve the main thread.
"""

from __future__ import annotations

import io
import json
import struct
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import torch

import ttnn

# `ttnn.distributed_context_duplicate()` returns a
# `tt::tt_metal::distributed::multihost::DistributedContext`. Its nanobind
# class binding lives in the ttml module; importing ttml here registers
# the class so the duplicate call can hand its return value to Python.
import ttml  # noqa: F401


# Tag values scoped to the private duplicated MPI context.
_TAG_MANIFEST_LEN: int = 0
_TAG_MANIFEST_BODY: int = 1
_TAG_WEIGHT_LEN: int = 2
_TAG_WEIGHT_BODY: int = 3

_CLOSE_JOIN_TIMEOUT_S: float = 10.0


# ---- small helpers (mirroring utils/weight_bridge.py) ------------------------


def _shape_to_list(shape) -> List[int]:
    return [int(d) for d in shape]


def _torch_save_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _torch_load_bytes(blob: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(blob), weights_only=True)


def _validate_source_tensor(key: str, tensor: "ttnn.Tensor") -> None:
    """Cheap sanity checks. The test uses TILE / DRAM-interleaved."""
    if tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"ThreadedWeightBridge: tensor {key!r} has layout={tensor.layout}, expected ttnn.TILE_LAYOUT.")
    if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError(
            f"ThreadedWeightBridge: tensor {key!r} is not in DRAM_MEMORY_CONFIG "
            f"(memory_config={tensor.memory_config()})."
        )


def _dtype_from_name(name: str) -> "ttnn.DataType":
    dt = getattr(ttnn.DataType, name, None)
    if dt is None:
        raise ValueError(f"unknown ttnn.DataType name {name!r}")
    return dt


def _layout_from_name(name: str) -> "ttnn.Layout":
    lo = getattr(ttnn.Layout, name, None)
    if lo is None:
        raise ValueError(f"unknown ttnn.Layout name {name!r}")
    return lo


class ThreadedWeightBridge:
    """Threaded MPI weight bridge with a dict-of-tensors payload and
    lazy on-device pads.

    Construct via :meth:`sender` or :meth:`receiver`, call :meth:`connect`
    on both ranks, then use :meth:`send_weights` / :meth:`receive_weights`.
    """

    _ROLE_SENDER = "sender"
    _ROLE_RECEIVER = "receiver"

    def __init__(
        self,
        *,
        role: str,
        peer_rank: int,
        mesh_device: "ttnn.MeshDevice",
        submeshes: Optional[List["ttnn.MeshDevice"]] = None,
    ) -> None:
        if role not in (self._ROLE_SENDER, self._ROLE_RECEIVER):
            raise ValueError(f"role must be sender or receiver, got {role!r}")
        self._role: str = role
        self._peer_rank: int = int(peer_rank)
        self._mesh: "ttnn.MeshDevice" = mesh_device

        # Receiver-only: where to land the incoming weights. The receiver
        # allocates one pad per (submesh, key) so ``receive_weights`` can
        # yield a ``List[dict]`` shaped exactly for
        # ``TttGenerationWorker.update_weights(per_submesh)`` -- no main-thread
        # ``to_torch`` / ``from_torch`` needed to reshape the received tensors
        # per submesh. Sender must pass ``submeshes=None``.
        if role == self._ROLE_RECEIVER:
            if not submeshes:
                raise ValueError(
                    "ThreadedWeightBridge.receiver requires a non-empty ``submeshes`` list -- "
                    "pads must be allocated directly on the caller's per-submesh Transformer "
                    "targets so ``worker.update_weights`` can consume them without a host "
                    "round-trip."
                )
            self._recv_targets: List["ttnn.MeshDevice"] = list(submeshes)
        else:
            if submeshes is not None:
                raise ValueError("ThreadedWeightBridge.sender: ``submeshes`` must be None.")
            self._recv_targets = []

        # Populated by connect().
        self._ctx: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None

        # Sender-side state.
        self._send_pads: Optional[Dict[str, "ttnn.Tensor"]] = None  # lazy
        self._send_pad_lock: threading.Lock = threading.Lock()
        self._send_pad_cv: threading.Condition = threading.Condition(self._send_pad_lock)
        self._has_send_data: bool = False
        self._send_pending_event: Any = None  # ttnn.MeshEvent
        # Ordered key list for the CURRENT send. Populated by send_weights;
        # consumed by the sender loop for manifest generation.
        self._send_keys: List[str] = []

        # Receiver-side state. One dict per submesh; index i is the dict for
        # ``self._recv_targets[i]``. Lazy-allocated on the first manifest.
        self._recv_pads: Optional[List[Dict[str, "ttnn.Tensor"]]] = None
        self._recv_manifest_entries: Optional[List[dict]] = None
        self._recv_pad_lock: threading.Lock = threading.Lock()
        self._recv_pad_cv: threading.Condition = threading.Condition(self._recv_pad_lock)
        self._has_recv_data: bool = False
        self._recv_pending_event: Any = None
        self._recv_pad_version: int = 0
        self._shutdown_seen: bool = False

        self._shutdown: threading.Event = threading.Event()

    # ---- construction --------------------------------------------------------

    @classmethod
    def sender(cls, *, peer_rank: int, mesh_device: "ttnn.MeshDevice") -> "ThreadedWeightBridge":
        return cls(role=cls._ROLE_SENDER, peer_rank=peer_rank, mesh_device=mesh_device)

    @classmethod
    def receiver(
        cls,
        *,
        peer_rank: int,
        mesh_device: "ttnn.MeshDevice",
        submeshes: List["ttnn.MeshDevice"],
    ) -> "ThreadedWeightBridge":
        return cls(
            role=cls._ROLE_RECEIVER,
            peer_rank=peer_rank,
            mesh_device=mesh_device,
            submeshes=submeshes,
        )

    # ---- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Duplicate the world MPI context and start the transport thread.

        Collective on the world context; both ranks must call in the same
        order.
        """
        self._ctx = ttnn.distributed_context_duplicate()
        print(
            f"[weight-bridge {self._role}] duplicated MPI context "
            f"(rank={self._ctx.rank()}, size={self._ctx.size()})",
            flush=True,
        )

        if self._role == self._ROLE_SENDER:
            self._thread = threading.Thread(
                target=self._sender_loop,
                name="weight-bridge-sender",
                daemon=True,
            )
        else:
            self._thread = threading.Thread(
                target=self._receiver_loop,
                name="weight-bridge-receiver",
                daemon=True,
            )
        self._thread.start()

    def close(self) -> None:
        """Sender: signal shutdown; sender thread emits the length-0
        close message and exits. Receiver: signal shutdown; the receiver
        thread exits once its peer sends the length-0 close message.
        Idempotent."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()

        if self._role == self._ROLE_SENDER:
            with self._send_pad_cv:
                self._send_pad_cv.notify_all()
            if self._thread is not None:
                self._thread.join(timeout=_CLOSE_JOIN_TIMEOUT_S)
                if self._thread.is_alive():
                    print(
                        "[weight-bridge sender] WARNING: sender thread did not "
                        f"exit within {_CLOSE_JOIN_TIMEOUT_S}s",
                        flush=True,
                    )
        else:
            with self._recv_pad_cv:
                self._recv_pad_cv.notify_all()

    # ==== Sender ==============================================================

    # ---- sender public API ---------------------------------------------------

    def send_weights(self, weights: Dict[str, "ttnn.Tensor"]) -> None:
        """Send a dict of tensors. Under the send pad lock:

          1. Wait until the previous dict has been fully drained (bridge
             thread finished the MPI send).
          2. Lazy-allocate ``self._send_pads`` on the first call; assert
             stability on subsequent calls.
          3. ``ttnn.copy(weights[k], self._send_pads[k], queue_id=0)`` per
             key. Runs on CQ0. Freezes the value into the pad.
          4. ``ttnn.record_event(mesh, 0)`` once (CQ0 is ordered so one
             event covers the whole burst).
          5. Notify the bridge thread and return.

        The caller is free to mutate ``weights[k]`` values immediately
        after ``send_weights`` returns.
        """
        if self._role != self._ROLE_SENDER:
            raise RuntimeError("send_weights() called on a non-sender bridge")
        if self._shutdown.is_set():
            raise RuntimeError("send_weights() called after close()")
        if not weights:
            raise ValueError("send_weights: empty dict")

        for k, t in weights.items():
            _validate_source_tensor(k, t)

        with self._send_pad_cv:
            while self._has_send_data and not self._shutdown.is_set():
                self._send_pad_cv.wait()
            if self._shutdown.is_set():
                return

            self._ensure_send_pads(weights)
            keys = sorted(weights.keys())

            for k in keys:
                ttnn.copy(weights[k], self._send_pads[k])

            # One event covers every ttnn.copy above since CQ0 is ordered.
            self._send_pending_event = ttnn.record_event(self._mesh, 0)
            self._send_keys = keys
            self._has_send_data = True
            self._send_pad_cv.notify_all()

    def _ensure_send_pads(self, weights: Dict[str, "ttnn.Tensor"]) -> None:
        """Lazy-allocate send pads on first call; assert stability after."""
        if self._send_pads is None:
            pads: Dict[str, "ttnn.Tensor"] = {}
            for k, t in weights.items():
                pads[k] = ttnn.allocate_tensor_on_device(
                    t.shape,
                    t.dtype,
                    t.layout,
                    self._mesh,
                    t.memory_config(),
                )
            self._send_pads = pads
            print(
                f"[weight-bridge sender] lazy-alloc'd {len(pads)} send pad(s): {sorted(pads.keys())}",
                flush=True,
            )
            return
        # Stability check.
        if set(weights.keys()) != set(self._send_pads.keys()):
            raise RuntimeError(
                f"send_weights: key set changed. "
                f"previous={sorted(self._send_pads.keys())}, new={sorted(weights.keys())}"
            )
        for k, t in weights.items():
            pad = self._send_pads[k]
            if tuple(t.shape) != tuple(pad.shape):
                raise RuntimeError(f"send_weights: {k!r} shape changed. pad={tuple(pad.shape)}, new={tuple(t.shape)}")
            if t.dtype != pad.dtype:
                raise RuntimeError(f"send_weights: {k!r} dtype changed. pad={pad.dtype}, new={t.dtype}")

    # ---- sender transport thread ---------------------------------------------

    def _sender_loop(self) -> None:
        """Sender transport thread body. One iteration = one message. All
        MPI work stays under the send pad lock, so the caller's next
        ``send_weights`` cannot race with an in-flight transfer."""
        assert self._ctx is not None
        while True:
            with self._send_pad_cv:
                self._wait_send_pad_full_or_shutdown()

                if not self._has_send_data:
                    # Only reason to wake here is shutdown.
                    self._send_close_message()
                    return

                try:
                    self._send_one_dict_over_wire()
                except Exception as e:
                    print(
                        f"[weight-bridge sender] MPI send failed: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    return

                self._mark_send_pad_empty()

    def _wait_send_pad_full_or_shutdown(self) -> None:
        """Precondition: caller holds ``self._send_pad_cv``. Blocks until
        either a caller has populated the pad or ``close()`` fired."""
        while not self._has_send_data and not self._shutdown.is_set():
            self._send_pad_cv.wait()

    def _send_close_message(self) -> None:
        """Emit the length-0 manifest header the peer treats as
        end-of-stream. Best-effort; the sender thread exits either way."""
        assert self._ctx is not None
        try:
            self._ctx.send(struct.pack("<Q", 0), self._peer_rank, _TAG_MANIFEST_LEN)
        except Exception as e:
            print(
                f"[weight-bridge sender] close-message send failed: {type(e).__name__}: {e}",
                flush=True,
            )

    def _send_one_dict_over_wire(self) -> None:
        """Precondition: caller holds ``self._send_pad_cv`` AND
        ``self._has_send_data`` is True. Ships the current send-pad dict
        as manifest + per-key blob over MPI. Blocks on CQ1 while
        ``ttnn.to_torch`` runs."""
        assert self._ctx is not None
        assert self._send_pads is not None

        # Cross-CQ ordering: make CQ1 wait for CQ0's copies to retire.
        ttnn.wait_for_event(1, self._send_pending_event)

        keys = list(self._send_keys)

        # 1. Manifest.
        entries = [
            {
                "key": k,
                "shape": _shape_to_list(self._send_pads[k].shape),
                "dtype": self._send_pads[k].dtype.name,
                "layout": self._send_pads[k].layout.name,
            }
            for k in keys
        ]
        manifest = json.dumps({"version": 1, "entries": entries}).encode("utf-8")
        self._ctx.send(struct.pack("<Q", len(manifest)), self._peer_rank, _TAG_MANIFEST_LEN)
        self._ctx.send(manifest, self._peer_rank, _TAG_MANIFEST_BODY)

        # 2. Per-key D->H on CQ1 + torch.save + MPI send.
        for k in keys:
            host = ttnn.to_torch(self._send_pads[k], cq_id=1)
            blob = _torch_save_bytes(host)
            self._ctx.send(struct.pack("<Q", len(blob)), self._peer_rank, _TAG_WEIGHT_LEN)
            self._ctx.send(blob, self._peer_rank, _TAG_WEIGHT_BODY)

    def _mark_send_pad_empty(self) -> None:
        """Precondition: caller holds ``self._send_pad_cv``. Wakes any
        ``send_weights`` blocked on the pad-full backpressure."""
        self._has_send_data = False
        self._send_pending_event = None
        self._send_keys = []
        self._send_pad_cv.notify_all()

    # ==== Receiver ============================================================

    # ---- receiver public API -------------------------------------------------

    @contextmanager
    def receive_weights(self) -> Iterator[List[Dict[str, "ttnn.Tensor"]]]:
        """Blocking context manager. Yields ``[{key: pad_ref}, ...]`` (one
        dict per submesh target passed to the constructor) while holding
        the recv pad lock so the bridge cannot overwrite the pads. Shape
        matches ``TttGenerationWorker.update_weights(per_submesh)``. On
        peer shutdown yields ``[{}]``.
        """
        if self._role != self._ROLE_RECEIVER:
            raise RuntimeError("receive_weights called on a non-receiver bridge")

        self._recv_pad_lock.acquire()
        try:
            while not self._has_recv_data and not self._shutdown_seen:
                self._recv_pad_cv.wait()

            if self._shutdown_seen and not self._has_recv_data:
                yield [{}]
                return

            # CQ0 waits for CQ1's writes to retire before the caller reads.
            ttnn.wait_for_event(0, self._recv_pending_event)

            assert self._recv_pads is not None
            # Shallow copy of each per-target mapping so the caller cannot
            # accidentally mutate the bridge's own dicts. The outer list is
            # ordered to match ``self._recv_targets`` (submesh order at
            # construction), which is what ``worker.update_weights``
            # consumes directly.
            yield [dict(pad) for pad in self._recv_pads]

            # Caller exited the with block. Clear has_data and notify.
            self._has_recv_data = False
            self._recv_pending_event = None
            self._recv_pad_cv.notify_all()
        finally:
            self._recv_pad_lock.release()

    @contextmanager
    def poll_weights(self) -> Iterator[Optional[List[Dict[str, "ttnn.Tensor"]]]]:
        """Non-blocking context manager. Yields ``None`` (without holding
        the lock) if no dict is pending; otherwise behaves like
        :meth:`receive_weights`.
        """
        if self._role != self._ROLE_RECEIVER:
            raise RuntimeError("poll_weights called on a non-receiver bridge")

        with self._recv_pad_lock:
            has_data = self._has_recv_data

        if not has_data:
            yield None
            return

        with self.receive_weights() as dicts:
            yield dicts

    def latest_version(self) -> int:
        """Receiver-only. Monotonic counter of blobs the bridge thread has
        written into the recv pad. Useful for logging."""
        if self._role != self._ROLE_RECEIVER:
            raise RuntimeError("latest_version() called on a non-receiver bridge")
        with self._recv_pad_lock:
            return self._recv_pad_version

    def _ensure_recv_pads(self, entries: List[dict]) -> None:
        """Lazy-allocate one pad per (submesh_target, key) on first manifest;
        assert stability after."""
        if self._recv_pads is None:
            pads_per_target: List[Dict[str, "ttnn.Tensor"]] = []
            for target in self._recv_targets:
                per_target: Dict[str, "ttnn.Tensor"] = {}
                for e in entries:
                    per_target[e["key"]] = ttnn.allocate_tensor_on_device(
                        ttnn.Shape(list(e["shape"])),
                        _dtype_from_name(e["dtype"]),
                        _layout_from_name(e["layout"]),
                        target,
                        ttnn.DRAM_MEMORY_CONFIG,  # test-only assumption
                    )
                pads_per_target.append(per_target)
            self._recv_pads = pads_per_target
            self._recv_manifest_entries = list(entries)
            print(
                f"[weight-bridge receiver] lazy-alloc'd {len(entries)} keys x "
                f"{len(pads_per_target)} target(s): {sorted(pads_per_target[0].keys())}",
                flush=True,
            )
            return
        # Stability check.
        prev_keys = {e["key"] for e in (self._recv_manifest_entries or [])}
        new_keys = {e["key"] for e in entries}
        if prev_keys != new_keys:
            raise RuntimeError(
                f"receive_weights: manifest key set changed. previous={sorted(prev_keys)}, new={sorted(new_keys)}"
            )
        prev_by_key = {e["key"]: e for e in (self._recv_manifest_entries or [])}
        for e in entries:
            p = prev_by_key[e["key"]]
            if p["shape"] != e["shape"] or p["dtype"] != e["dtype"] or p["layout"] != e["layout"]:
                raise RuntimeError(f"receive_weights: {e['key']!r} spec drifted. prev={p}, new={e}")

    # ---- receiver transport thread -------------------------------------------

    def _receiver_loop(self) -> None:
        """Receiver transport thread body. One iteration = one message.
        Manifest recv runs OUTSIDE the recv pad lock (so ``close()`` on
        this rank can flip ``_shutdown_seen`` without contending), then
        the per-key blob recv + H->D copy + event record all run under
        the lock so the caller's ``with receive_weights()`` sees a
        coherent pad."""
        assert self._ctx is not None
        while True:
            manifest = self._recv_next_manifest_or_close()
            if manifest is None:
                return  # peer closed or MPI failure

            with self._recv_pad_cv:
                if not self._wait_recv_pad_empty_or_shutdown():
                    return
                try:
                    self._ensure_recv_pads(manifest["entries"])
                except Exception as e:
                    print(
                        f"[weight-bridge receiver] ensure_recv_pads failed: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    return
                try:
                    self._recv_blobs_into_pads(manifest["entries"])
                except Exception as e:
                    print(
                        f"[weight-bridge receiver] weight recv failed: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    return
                v = self._mark_recv_pad_full()

            print(
                f"[weight-bridge receiver] wrote pad v={v} ({len(manifest['entries'])} keys)",
                flush=True,
            )

    def _recv_next_manifest_or_close(self) -> Optional[dict]:
        """Recv the manifest length + body. Returns the parsed dict, or
        ``None`` if the peer sent the length-0 close message or an MPI
        failure occurred. On close/failure sets ``_shutdown_seen`` and
        notifies main so a blocking ``receive_weights`` returns.
        Called with NO lock held."""
        assert self._ctx is not None
        try:
            raw_len = self._ctx.recv(8, self._peer_rank, _TAG_MANIFEST_LEN)
        except Exception as e:
            print(
                f"[weight-bridge receiver] manifest-len recv failed: {type(e).__name__}: {e}",
                flush=True,
            )
            with self._recv_pad_cv:
                self._shutdown_seen = True
                self._recv_pad_cv.notify_all()
            return None
        (manifest_len,) = struct.unpack("<Q", raw_len)

        if manifest_len == 0:
            with self._recv_pad_cv:
                self._shutdown_seen = True
                self._recv_pad_cv.notify_all()
            return None

        try:
            manifest_bytes = self._ctx.recv(int(manifest_len), self._peer_rank, _TAG_MANIFEST_BODY)
        except Exception as e:
            print(
                f"[weight-bridge receiver] manifest-body recv failed: {type(e).__name__}: {e}",
                flush=True,
            )
            return None
        return json.loads(manifest_bytes.decode("utf-8"))

    def _wait_recv_pad_empty_or_shutdown(self) -> bool:
        """Precondition: caller holds ``self._recv_pad_cv``. Blocks until
        the caller has consumed the previous dict (via ``receive_weights``)
        or ``close()`` fires. Returns True to proceed, False if the
        receiver should exit."""
        while self._has_recv_data and not self._shutdown.is_set():
            self._recv_pad_cv.wait()
        return not self._shutdown.is_set()

    def _recv_blobs_into_pads(self, entries: List[dict]) -> None:
        """Precondition: caller holds ``self._recv_pad_cv``. Per-key
        length + blob recv + torch.load + wrap as ttnn host tensor +
        ``ttnn.copy_host_to_device_tensor`` into the pre-allocated recv
        pad on CQ1. Fans out to every submesh target so the receiver's
        yielded ``List[dict]`` matches ``worker.update_weights``'s
        expected per-submesh shape -- no main-thread H2D."""
        assert self._ctx is not None
        assert self._recv_pads is not None
        for entry in entries:
            (blob_len,) = struct.unpack("<Q", self._ctx.recv(8, self._peer_rank, _TAG_WEIGHT_LEN))
            blob = self._ctx.recv(int(blob_len), self._peer_rank, _TAG_WEIGHT_BODY)
            host_tensor = _torch_load_bytes(blob)
            # Wrap the torch host tensor as a ttnn host tensor (no device=
            # arg -> stays on host). ``from_torch`` here is zero-copy: the
            # returned HostBuffer views the torch storage.
            ttnn_host = ttnn.from_torch(
                host_tensor,
                dtype=_dtype_from_name(entry["dtype"]),
                layout=_layout_from_name(entry["layout"]),
            )
            # Fan out to every submesh target. On a single-submesh receiver
            # this is one H2D per key; a [1, N] parent with N submeshes is N
            # H2Ds per key. All on CQ1 on the bridge thread, so the main
            # inference thread never sees a to_torch / from_torch bounce.
            for per_target in self._recv_pads:
                ttnn.copy_host_to_device_tensor(ttnn_host, per_target[entry["key"]], cq_id=1)

    def _mark_recv_pad_full(self) -> int:
        """Precondition: caller holds ``self._recv_pad_cv``. Records the
        CQ1 event the caller's CQ0 read waits on, flips ``_has_recv_data``,
        bumps ``_recv_pad_version``, notifies. Returns the new version
        for logging."""
        self._recv_pending_event = ttnn.record_event(self._mesh, 1)
        self._has_recv_data = True
        self._recv_pad_version += 1
        self._recv_pad_cv.notify_all()
        return self._recv_pad_version
