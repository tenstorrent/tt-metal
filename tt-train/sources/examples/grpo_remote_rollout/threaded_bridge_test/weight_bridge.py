# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal threaded MPI weight bridge for the threaded_bridge_test example.

Data flows THROUGH a pre-allocated on-device `ttnn.Tensor` pad on each side.

Sender side (rank 0)
====================
  1. Main thread (CQ0) calls ``push_tensor(x)``. Under the send pad lock this
     issues ``ttnn.copy(x, send_pad)`` (D->D on CQ0) then records an event on
     CQ0.
  2. Sender bridge thread wakes on the condvar, still holding the lock. It
     ``ttnn.wait_for_event(1, event)`` (makes CQ1 wait for CQ0's write to
     retire), does a blocking ``ttnn.to_torch(send_pad, cq_id=1)`` (D->H on
     CQ1), serializes with ``torch.save``, then MPI-sends [u64 length][blob]
     on the duplicated context. Still holding the lock throughout, so the
     pad is "occupied" for the full send lifecycle.

Receiver side (rank 1)
======================
  1. Receiver bridge thread acquires the recv pad lock, MPI-recvs [u64
     length][blob], deserializes with ``torch.load``, wraps the host torch
     tensor as a ttnn host tensor via ``ttnn.from_torch(...)`` (no device=),
     issues ``ttnn.copy_host_to_device_tensor(host, recv_pad, cq_id=1)``
     (H->D on CQ1), records an event on CQ1. Sets ``has_data=True`` and
     notifies.
  2. Main thread calls ``acquire_recv_pad()``. Under the lock the bridge
     ``ttnn.wait_for_event(0, event)`` (makes CQ0 wait for CQ1's write to
     retire) then returns the recv pad WITH THE LOCK STILL HELD. The caller
     samples on CQ0 (e.g. ``ttnn.to_torch(recv_pad, cq_id=0)``) and then
     calls ``release_recv_pad()`` which clears ``has_data`` and drops the
     lock.

Framing: fixed ``[u64 length][blob]``. ``length == 0`` is the shutdown
sentinel sent by the sender's ``close()``.

Concurrency
===========
The pad lock is held END-TO-END for a message on both sides -- including the
MPI transfer. Cross-CQ ordering uses ``ttnn.record_event`` / ``wait_for_event``
per message (no ``synchronize_device``). Relies on the GIL-release wrap on
``DistributedContext`` methods in
``tt-train/sources/ttml/nanobind/nb_core.cpp``; otherwise the bridge thread's
blocking MPI call would hold both the GIL AND the pad lock.
"""

from __future__ import annotations

import io
import struct
import threading
from typing import Any, Optional

import torch

import ttnn

# `ttnn.distributed_context_duplicate()` returns a
# `tt::tt_metal::distributed::multihost::DistributedContext`. That C++ type's
# nanobind class binding lives in the ttml module (see
# `tt-train/sources/ttml/nanobind/nb_core.cpp`); ttnn only re-exports the
# function. Importing `ttml` here forces nanobind to register the class in
# its type registry, otherwise the `duplicate` call raises
# `TypeError: Unable to convert function return value to a Python type!`.
import ttml  # noqa: F401


_TAG_LEN: int = 0
_TAG_BODY: int = 1
_SHUTDOWN_JOIN_TIMEOUT_S: float = 10.0


class ThreadedWeightBridge:
    """One-slot threaded MPI weight bridge for a single fixed-shape tensor.

    See the module docstring for the full contract. Instances are constructed
    via ``ThreadedWeightBridge.sender(...)`` or ``.receiver(...)``.
    """

    _ROLE_SENDER = "sender"
    _ROLE_RECEIVER = "receiver"

    def __init__(
        self,
        *,
        role: str,
        peer_rank: int,
        mesh_device: "ttnn.MeshDevice",
        shape,
        dtype: "ttnn.DataType",
        layout: "ttnn.Layout",
        memory_config: "ttnn.MemoryConfig",
    ) -> None:
        if role not in (self._ROLE_SENDER, self._ROLE_RECEIVER):
            raise ValueError(f"role must be sender or receiver, got {role!r}")
        self._role: str = role
        self._peer_rank: int = int(peer_rank)
        self._mesh: "ttnn.MeshDevice" = mesh_device
        self._shape = shape
        self._dtype = dtype
        self._layout = layout
        self._memory_config = memory_config

        # Populated by connect().
        self._ctx: Optional[Any] = None
        self._pad: Optional["ttnn.Tensor"] = None
        self._thread: Optional[threading.Thread] = None

        self._lock: threading.Lock = threading.Lock()
        self._cv: threading.Condition = threading.Condition(self._lock)

        # Pad state. Never re-assign the pad after connect(); flip has_data
        # instead. pending_event carries the writer's ttnn.MeshEvent that the
        # reader must wait for before touching the pad's data on its CQ.
        self._has_data: bool = False
        self._pending_event: Any = None  # ttnn.MeshEvent

        self._shutdown: threading.Event = threading.Event()
        # Receiver-only: set to True once the receiver bridge has consumed the
        # peer's length-0 sentinel, so main knows to stop draining.
        self._shutdown_seen: bool = False

    # ---- construction --------------------------------------------------------

    @classmethod
    def sender(
        cls,
        *,
        peer_rank: int,
        mesh_device: "ttnn.MeshDevice",
        shape,
        dtype: "ttnn.DataType" = ttnn.bfloat16,
        layout: "ttnn.Layout" = ttnn.TILE_LAYOUT,
        memory_config: "ttnn.MemoryConfig" = ttnn.DRAM_MEMORY_CONFIG,
    ) -> "ThreadedWeightBridge":
        return cls(
            role=cls._ROLE_SENDER,
            peer_rank=peer_rank,
            mesh_device=mesh_device,
            shape=shape,
            dtype=dtype,
            layout=layout,
            memory_config=memory_config,
        )

    @classmethod
    def receiver(
        cls,
        *,
        peer_rank: int,
        mesh_device: "ttnn.MeshDevice",
        shape,
        dtype: "ttnn.DataType" = ttnn.bfloat16,
        layout: "ttnn.Layout" = ttnn.TILE_LAYOUT,
        memory_config: "ttnn.MemoryConfig" = ttnn.DRAM_MEMORY_CONFIG,
    ) -> "ThreadedWeightBridge":
        return cls(
            role=cls._ROLE_RECEIVER,
            peer_rank=peer_rank,
            mesh_device=mesh_device,
            shape=shape,
            dtype=dtype,
            layout=layout,
            memory_config=memory_config,
        )

    # ---- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Duplicate the world MPI context, pre-allocate the pad on device,
        and start the single background thread.

        The ``ttnn.distributed_context_duplicate`` call is collective across
        both ranks and acts as the connect-time barrier -- both ranks must
        reach here in the same order for the private contexts to match.
        """
        self._ctx = ttnn.distributed_context_duplicate()
        print(
            f"[bridge {self._role}] duplicated MPI context " f"(rank={self._ctx.rank()}, size={self._ctx.size()})",
            flush=True,
        )

        shape = self._shape if isinstance(self._shape, ttnn.Shape) else ttnn.Shape(list(self._shape))
        self._pad = ttnn.allocate_tensor_on_device(
            shape,
            self._dtype,
            self._layout,
            self._mesh,
            self._memory_config,
        )
        print(f"[bridge {self._role}] pre-allocated pad shape={tuple(self._pad.shape)}", flush=True)

        if self._role == self._ROLE_SENDER:
            self._thread = threading.Thread(
                target=self._sender_loop,
                name="threaded-weight-bridge-sender",
                daemon=True,
            )
        else:
            self._thread = threading.Thread(
                target=self._receiver_loop,
                name="threaded-weight-bridge-receiver",
                daemon=True,
            )
        self._thread.start()

    def close(self) -> None:
        """Set shutdown; on the sender the bridge thread then emits the
        length-0 sentinel and exits. On the receiver we can only wake the
        thread from a WAIT on the condvar -- if it's blocked in MPI_Recv
        we rely on the peer's close() having already sent the sentinel."""
        with self._cv:
            if self._shutdown.is_set():
                return
            self._shutdown.set()
            self._cv.notify_all()

        if self._role == self._ROLE_SENDER and self._thread is not None:
            self._thread.join(timeout=_SHUTDOWN_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                print(
                    "[bridge sender] WARNING: sender thread did not exit within "
                    f"{_SHUTDOWN_JOIN_TIMEOUT_S}s; leaving as daemon",
                    flush=True,
                )

    # ---- sender API ----------------------------------------------------------

    def push_tensor(self, x: "ttnn.Tensor") -> None:
        """Sender-only, main-thread call.

        Under the send pad lock:
          1. Wait until the pad is empty (bridge finished sending the previous
             message).
          2. ``ttnn.copy(x, send_pad)`` (D->D on CQ0 -- ttnn.copy runs on the
             default queue).
          3. ``ttnn.record_event(mesh, 0)`` so the bridge can order its
             CQ1 read against this CQ0 write.
          4. Set ``has_data=True``, notify, release the lock.

        Blocks until the pad is free. Not thread-safe against multiple
        callers; the test uses a single main thread.
        """
        if self._role != self._ROLE_SENDER:
            raise RuntimeError("push_tensor() called on a non-sender bridge")
        assert self._pad is not None, "connect() must be called before push_tensor"

        with self._cv:
            while self._has_data and not self._shutdown.is_set():
                self._cv.wait()
            if self._shutdown.is_set():
                return

            # D->D copy on CQ0 (default queue for ttnn.copy).
            ttnn.copy(x, self._pad)
            # Record an event on CQ0 AFTER the copy is enqueued. The bridge
            # thread's CQ1 will wait_for_event on this before its to_torch.
            self._pending_event = ttnn.record_event(self._mesh, 0)

            self._has_data = True
            self._cv.notify_all()

    def _sender_loop(self) -> None:
        """Sender bridge thread body. Holds the pad lock end-to-end for a
        message: wait_for_event + to_torch + serialize + MPI_Send."""
        assert self._ctx is not None and self._pad is not None
        while True:
            with self._cv:
                while not self._has_data and not self._shutdown.is_set():
                    self._cv.wait()

                if not self._has_data and self._shutdown.is_set():
                    # No pending message; send the length-0 shutdown sentinel
                    # under the lock and exit.
                    try:
                        self._ctx.send(struct.pack("<Q", 0), self._peer_rank, _TAG_LEN)
                    except Exception as e:
                        print(
                            f"[bridge sender] shutdown-sentinel send failed: " f"{type(e).__name__}: {e}",
                            flush=True,
                        )
                    return

                # Cross-CQ ordering: CQ1 waits for CQ0's copy to retire.
                ttnn.wait_for_event(1, self._pending_event)

                # Blocking D->H on CQ1.
                host_torch = ttnn.to_torch(self._pad, cq_id=1)

                # Serialize + MPI send: still under lock so the pad stays
                # "occupied" for the full send lifecycle.
                buf = io.BytesIO()
                torch.save(host_torch, buf)
                blob = buf.getvalue()
                try:
                    self._ctx.send(struct.pack("<Q", len(blob)), self._peer_rank, _TAG_LEN)
                    self._ctx.send(blob, self._peer_rank, _TAG_BODY)
                except Exception as e:
                    print(
                        f"[bridge sender] MPI send failed: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    return

                self._has_data = False
                self._pending_event = None
                self._cv.notify_all()

    # ---- receiver API --------------------------------------------------------

    def acquire_recv_pad(self) -> Optional["ttnn.Tensor"]:
        """Receiver-only, main-thread call.

        Blocks until the recv pad has a fresh message OR the peer's shutdown
        sentinel has been consumed. On sentinel returns ``None`` (and releases
        the lock). Otherwise returns the pre-allocated recv pad WITH THE LOCK
        STILL HELD; the caller must do its CQ0 read and then call
        ``release_recv_pad()``.

        Before returning, waits on the pending write event so the caller's
        subsequent ``ttnn.to_torch(pad, cq_id=0)`` is coherent with the
        bridge's CQ1 write.
        """
        if self._role != self._ROLE_RECEIVER:
            raise RuntimeError("acquire_recv_pad() called on a non-receiver bridge")
        assert self._pad is not None

        # Manually acquire (release is done by release_recv_pad or on
        # shutdown path below).
        self._lock.acquire()
        try:
            while not self._has_data and not self._shutdown_seen:
                self._cv.wait()
            if self._shutdown_seen and not self._has_data:
                self._lock.release()
                return None
        except BaseException:
            # Ensure the lock is released on unexpected exceptions during the
            # wait (e.g. KeyboardInterrupt).
            self._lock.release()
            raise

        # Cross-CQ ordering: CQ0 waits for the bridge's CQ1 write to retire.
        ttnn.wait_for_event(0, self._pending_event)
        return self._pad

    def release_recv_pad(self) -> None:
        """Receiver-only, main-thread call. Paired with a prior
        ``acquire_recv_pad()`` return of a real pad."""
        if self._role != self._ROLE_RECEIVER:
            raise RuntimeError("release_recv_pad() called on a non-receiver bridge")
        # Precondition: we hold self._lock.
        self._has_data = False
        self._pending_event = None
        self._cv.notify_all()
        self._lock.release()

    def _receiver_loop(self) -> None:
        """Receiver bridge thread body. Holds the pad lock end-to-end for a
        message: MPI_Recv + deserialize + copy_host_to_device_tensor + record
        event."""
        assert self._ctx is not None and self._pad is not None
        while True:
            with self._cv:
                while self._has_data and not self._shutdown.is_set():
                    self._cv.wait()
                if self._has_data and self._shutdown.is_set():
                    # unusual: main called close() while a message was queued.
                    return
                if not self._has_data and self._shutdown.is_set():
                    return

                # Blocking MPI_Recv of the length header. If the peer sent
                # the shutdown sentinel we get length == 0.
                try:
                    raw_len = self._ctx.recv(8, self._peer_rank, _TAG_LEN)
                except Exception as e:
                    print(
                        f"[bridge receiver] length recv failed: " f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    return
                (blob_len,) = struct.unpack("<Q", raw_len)

                if blob_len == 0:
                    self._shutdown_seen = True
                    self._cv.notify_all()
                    return

                # Full-size body recv, still under lock.
                try:
                    blob = self._ctx.recv(int(blob_len), self._peer_rank, _TAG_BODY)
                except Exception as e:
                    print(
                        f"[bridge receiver] body recv failed: " f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    return

                host_torch = torch.load(io.BytesIO(blob), weights_only=True)

                # Wrap the host torch tensor as a ttnn host tensor (no
                # device= arg -> stays on host). Cheap.
                ttnn_host = ttnn.from_torch(
                    host_torch,
                    dtype=self._dtype,
                    layout=self._layout,
                )

                # H->D on CQ1 into the pre-allocated pad, then record an
                # event on CQ1 so the main thread's CQ0 read can wait on it.
                ttnn.copy_host_to_device_tensor(ttnn_host, self._pad, cq_id=1)
                self._pending_event = ttnn.record_event(self._mesh, 1)

                self._has_data = True
                self._cv.notify_all()
