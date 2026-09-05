# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Threaded MPI weight bridge with single-slot sending / receiving pads.

Standalone class (NOT a :class:`WeightBridge` subclass): the async
``push`` / ``poll`` API doesn't fit the ABC's synchronous
``send_weights`` / ``receive_weights`` contract.

Threading contract
==================

The bridge OWNS exactly one backing thread -- a sender thread on sender
bridges, a receiver thread on receiver bridges. Started in :meth:`connect`,
joined in :meth:`close`. Nothing outside the class starts or stops them.

Sender side:
  * :meth:`push` runs on the CALLER's main thread. It validates the input
    ``hf_dict``, does the D->H copy (:func:`_device0_to_host`) + serialization
    (:func:`_torch_save_bytes`), stores the resulting bytes into a single-slot
    pad, and returns. The sender thread does the actual (blocking) MPI send
    with those bytes.
  * Only bytes cross the pad, never live ttnn.Tensor references. That keeps
    the sender thread from touching ttnn / device state, and prevents any
    subsequent ``optimizer.step`` in the caller from mutating a tensor we're
    still reading.
  * Backpressure: :meth:`push` blocks on a condvar if the previous blob is
    still sitting in the pad (sender thread hasn't started sending it yet).

Receiver side:
  * The receiver thread does a blocking recv on the private context in a loop,
    dropping each received blob into a single-slot receiving pad. Latest wins
    -- if the main thread hasn't consumed the previous blob by the time the
    next one arrives, the older blob is overwritten.
  * :meth:`poll` runs on the CALLER's main thread. It is non-blocking: acquire
    the pad lock, read + null the slot, release, return the bytes (or ``None``).

Fire-and-forget
===============

There is no ack round-trip. Backpressure comes from (a) the sender pad
condvar and (b) MPI's own flow control on the weight tag (a second send blocks
until the peer has consumed the first). Both are per-bridge; the caller does
not need to synchronize with the peer via events.

Private MPI communicator
========================

The bridge duplicates the current world context (via
:func:`ttnn.distributed_context_duplicate`) inside :meth:`connect` and runs ALL
of its subsequent MPI on that private context. Consequences:

  * Matching space is fully isolated from the world context, ``HostWeightBridge``,
    ``AsyncTrainingEventChannel``, ``MPIRolloutClient``, and any other duplicated
    context in the process. Tag values chosen here (0..3) can never cross-match
    with anything on the world context.
  * Under ``MPI_THREAD_MULTIPLE`` on OpenMPI 5.x builds with per-comm locking,
    MPI calls on the private context proceed independently of MPI calls on the
    world context -- the event channel and the bridge can genuinely run in
    parallel across threads.
  * ``MPI_Comm_dup`` is a collective; both ranks must call
    :meth:`connect` in a matching order for the resulting contexts to line up.

Wire format
===========

Fixed-size payload known ahead of time via the ``expected_bytes_size``
constructor arg. Each transfer is two MPI messages on the private context:
an 8-byte length on :data:`_THREADED_WEIGHT_LEN_TAG` (tag 2) followed by
the blob on :data:`_THREADED_WEIGHT_BLOB_TAG` (tag 3). The length prefix is
redundant with ``expected_bytes_size`` but matches :class:`HostWeightBridge`'s
framing so the two bridges are structurally comparable.
"""

from __future__ import annotations

import struct
import threading
from typing import Any, List, Optional

import ttnn

from .weight_bridge import (
    RECEIVER_RANK,
    SENDER_RANK,
    _ROLE_RECEIVER,
    _ROLE_SENDER,
    _check_role_rank,
    _device0_to_host,
    _require_distributed_context,
    _torch_save_bytes,
    _validate_source_tensor,
)


# Tag values are scoped to the bridge's PRIVATE duplicated MPI communicator
# (see :meth:`ThreadedHostWeightBridge.connect`), so we can use small, dense
# tag numbers without worrying about collisions with any other subsystem on
# the world context.
_HANDSHAKE_TAG_FROM_SENDER: int = 0
_HANDSHAKE_TAG_FROM_RECEIVER: int = 1
_THREADED_WEIGHT_LEN_TAG: int = 2
_THREADED_WEIGHT_BLOB_TAG: int = 3

_HANDSHAKE_PAYLOAD: bytes = b"ready"

# How long :meth:`close` waits for the sender thread to finish draining before
# giving up. The receiver thread is a daemon and simply dies with the process
# if it's stuck in a blocking recv at shutdown.
_CLOSE_JOIN_TIMEOUT_S: float = 30.0


class ThreadedHostWeightBridge:
    """Threaded weight bridge with a single-slot sending or receiving pad.

    See the module docstring for the full threading + lifecycle contract.
    """

    def __init__(
        self,
        *,
        role: str,
        peer_rank: int,
        expected_bytes_size: int,
        mesh: Optional["ttnn.MeshDevice"] = None,
        submeshes: Optional[List["ttnn.MeshDevice"]] = None,
    ) -> None:
        local_rank = _require_distributed_context("ThreadedHostWeightBridge")
        _check_role_rank("ThreadedHostWeightBridge", role, local_rank)

        self._role: str = role
        self._peer_rank: int = int(peer_rank)
        self._expected_bytes_size: int = int(expected_bytes_size)
        self._mesh: Optional["ttnn.MeshDevice"] = mesh

        # Sender-only receiver targets go unused here (no ttnn reconstruction on
        # the receiver side yet); keep the field for signature parity with
        # HostWeightBridge in case a future subclass wants to plumb them in.
        if role == _ROLE_SENDER:
            if mesh is None:
                raise ValueError("ThreadedHostWeightBridge.init_sender requires a mesh (any [1, N]).")
            if submeshes is not None:
                raise ValueError("ThreadedHostWeightBridge.init_sender: 'submeshes' must be None on the sender.")
            self._targets: Optional[List["ttnn.MeshDevice"]] = None
        else:
            if not submeshes:
                raise ValueError("ThreadedHostWeightBridge.init_receiver requires a non-empty submeshes list.")
            self._targets = list(submeshes)

        # Private MPI context for this bridge. Populated by :meth:`connect`
        # via ``ttnn.distributed_context_duplicate`` (collective MPI_Comm_dup).
        # The runtime object is an instance of the ttml-registered class
        # ``_ttml.core.distributed.DistributedContext``; the shared_ptr comes
        # from nanobind's cross-module type registry. All MPI calls on this
        # bridge -- handshake, blob length, blob body -- run on this context,
        # NOT on the world context.
        self._ctx: Optional[Any] = None

        # Sender-side state (bytes handed off from main -> sender thread).
        self._send_pad: Optional[bytes] = None
        self._send_pad_lock: threading.Lock = threading.Lock()
        self._send_pad_cv: threading.Condition = threading.Condition(self._send_pad_lock)
        self._sender_thread: Optional[threading.Thread] = None

        # Receiver-side state (bytes handed off from receiver thread -> main).
        self._recv_pad: Optional[bytes] = None
        self._recv_pad_lock: threading.Lock = threading.Lock()
        self._recv_pad_version: int = 0
        self._receiver_thread: Optional[threading.Thread] = None

        self._shutdown: threading.Event = threading.Event()

    # ---- construction ----------------------------------------------------

    @classmethod
    def init_sender(
        cls,
        *,
        mesh: "ttnn.MeshDevice",
        peer_rank: int,
        expected_bytes_size: int,
    ) -> "ThreadedHostWeightBridge":
        if int(peer_rank) != RECEIVER_RANK:
            raise ValueError(
                f"ThreadedHostWeightBridge.init_sender: peer_rank must be RECEIVER_RANK="
                f"{RECEIVER_RANK} (got {peer_rank})."
            )
        return cls(
            role=_ROLE_SENDER,
            peer_rank=peer_rank,
            expected_bytes_size=expected_bytes_size,
            mesh=mesh,
        )

    @classmethod
    def init_receiver(
        cls,
        *,
        mesh: "ttnn.MeshDevice",
        peer_rank: int,
        submeshes: List["ttnn.MeshDevice"],
        expected_bytes_size: int,
    ) -> "ThreadedHostWeightBridge":
        if int(peer_rank) != SENDER_RANK:
            raise ValueError(
                f"ThreadedHostWeightBridge.init_receiver: peer_rank must be SENDER_RANK="
                f"{SENDER_RANK} (got {peer_rank})."
            )
        return cls(
            role=_ROLE_RECEIVER,
            peer_rank=peer_rank,
            expected_bytes_size=expected_bytes_size,
            mesh=mesh,
            submeshes=submeshes,
        )

    # ---- lifecycle -------------------------------------------------------

    def connect(self) -> None:
        """Duplicate the current world MPI context into a private one, do the
        two-rank handshake on it, then start the backing thread.

        Threading contract at this point:
          * ``ttnn.distributed_context_duplicate`` is a collective on the
            world context; both ranks must reach ``connect()`` in the same
            order for the resulting private contexts to match. The duplicate
            call itself acts as a synchronization barrier.
          * The duplicate runs on the CALLER thread with no other MPI activity
            in flight -- safe under any MPI thread-level.
          * The handshake also runs on the CALLER thread over the private
            context. Confirms both sides ended up with matching duplicated
            contexts.
          * The backing thread (sender or receiver) is spawned only AFTER
            the handshake completes. From that point on the backing thread is
            the sole user of the private context on this rank; the caller
            thread never touches ``self._ctx`` again.
        """
        self._ctx = ttnn.distributed_context_duplicate()
        print(
            f"[threaded-bridge {self._role}] duplicated MPI context (rank={self._ctx.rank()}, "
            f"size={self._ctx.size()})",
            flush=True,
        )
        self._handshake_on_ctx()
        print(f"[threaded-bridge {self._role}] handshake complete on private context", flush=True)

        if self._role == _ROLE_SENDER:
            self._sender_thread = threading.Thread(
                target=self._sender_loop,
                name="threaded-host-weight-bridge-sender",
                daemon=True,
            )
            self._sender_thread.start()
        else:
            self._receiver_thread = threading.Thread(
                target=self._receiver_loop,
                name="threaded-host-weight-bridge-receiver",
                daemon=True,
            )
            self._receiver_thread.start()

    def _handshake_on_ctx(self) -> None:
        """Two-rank barrier on the bridge's private context. Uses tags
        0 / 1 -- safe because the context is disjoint from every other
        subsystem's matching space.

        Uses ttml's ``send`` / ``recv`` method names (defined on the
        ``DistributedContext`` class in
        ``tt-train/sources/ttml/nanobind/nb_core.cpp``); the ``bytes`` /
        ``nbytes`` naming mismatch with ``ttnn.distributed_context_recv_bytes``
        is a ttml-side convention, not something we normalize here.
        """
        assert self._ctx is not None, "connect() must have duplicated the context first"
        if self._role == _ROLE_SENDER:
            self._ctx.send(_HANDSHAKE_PAYLOAD, self._peer_rank, _HANDSHAKE_TAG_FROM_SENDER)
            self._ctx.recv(len(_HANDSHAKE_PAYLOAD), self._peer_rank, _HANDSHAKE_TAG_FROM_RECEIVER)
        else:
            self._ctx.recv(len(_HANDSHAKE_PAYLOAD), self._peer_rank, _HANDSHAKE_TAG_FROM_SENDER)
            self._ctx.send(_HANDSHAKE_PAYLOAD, self._peer_rank, _HANDSHAKE_TAG_FROM_RECEIVER)

    def close(self) -> None:
        """Signal shutdown and join the sender thread. Safe to call twice.

        The receiver thread is a daemon and may be blocked in ``recv_bytes``
        at close() time; it dies with the process on interpreter exit rather
        than being join()ed here.
        """
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        if self._role == _ROLE_SENDER:
            with self._send_pad_cv:
                # Wake the sender thread if it's waiting for a blob.
                self._send_pad_cv.notify_all()
            if self._sender_thread is not None:
                self._sender_thread.join(timeout=_CLOSE_JOIN_TIMEOUT_S)
                if self._sender_thread.is_alive():
                    print(
                        "[threaded-bridge] WARNING: sender thread did not exit within "
                        f"{_CLOSE_JOIN_TIMEOUT_S}s; leaving it as a daemon",
                        flush=True,
                    )

    # ---- sender-side API -------------------------------------------------

    def push(self, hf_dict: dict) -> None:
        """Sender-only. Copy weights into the sending pad on the CALLER's
        main thread, then hand off to the sender thread.

        Steps (all on the caller thread):

          1. Validate the input dict (mock supports exactly one tensor).
          2. Validate the tensor's dtype/layout/memory/replication.
          3. D->H via ``_device0_to_host`` + serialize via ``_torch_save_bytes``.
          4. Acquire the pad lock; wait on the condvar if the previous blob
             hasn't been snapshotted out by the sender thread yet.
          5. Store the blob in the pad; notify the sender thread; release.

        Returns as soon as the pad is populated. The actual MPI send is
        performed by the sender thread.

        Raises:
            RuntimeError: if the bridge is not a sender or is shutting down.
            ValueError: if the input dict is malformed for this mock.
        """
        if self._role != _ROLE_SENDER:
            raise RuntimeError("ThreadedHostWeightBridge.push called on a non-sender bridge.")
        if self._shutdown.is_set():
            raise RuntimeError("ThreadedHostWeightBridge.push called after close().")
        if len(hf_dict) != 1:
            raise ValueError(
                f"ThreadedHostWeightBridge.push: mock only supports exactly one weight per push, "
                f"got {len(hf_dict)}."
            )

        ((key, tensor),) = tuple(hf_dict.items())
        _validate_source_tensor(key, tensor)

        # D->H + torch.save happens on the caller thread. Sender thread never
        # touches ttnn.
        blob = _torch_save_bytes(_device0_to_host(tensor))

        with self._send_pad_cv:
            while self._send_pad is not None and not self._shutdown.is_set():
                self._send_pad_cv.wait()
            if self._shutdown.is_set():
                return
            self._send_pad = blob
            self._send_pad_cv.notify_all()

    def _sender_loop(self) -> None:
        """Sender thread body. Owns the MPI send on the bridge's private
        context only -- no ttnn / no device work, no touching the world
        context. Runs until ``self._shutdown`` is set AND the pad is drained
        (so a push that arrived just before close() still ships)."""
        assert self._ctx is not None, "connect() must have populated the context before the sender starts"
        while True:
            with self._send_pad_cv:
                while self._send_pad is None and not self._shutdown.is_set():
                    self._send_pad_cv.wait()
                if self._shutdown.is_set() and self._send_pad is None:
                    return
                snapshot = self._send_pad
                self._send_pad = None
                self._send_pad_cv.notify_all()

            # Outside the lock: blocking MPI send on the PRIVATE context.
            # `snapshot` is a local ref; main is free to write a new blob
            # into the pad concurrently.
            print(
                f"[threaded-bridge sender] sending blob ({len(snapshot)} B) to rank {self._peer_rank}...",
                flush=True,
            )
            self._ctx.send(struct.pack("<Q", len(snapshot)), self._peer_rank, _THREADED_WEIGHT_LEN_TAG)
            self._ctx.send(snapshot, self._peer_rank, _THREADED_WEIGHT_BLOB_TAG)
            print(
                f"[threaded-bridge sender] send complete ({len(snapshot)} B)",
                flush=True,
            )

    # ---- receiver-side API -----------------------------------------------

    def poll(self) -> Optional[bytes]:
        """Receiver-only. Non-blocking peek + consume of the receiving pad.

        Returns the latest received blob (as raw bytes -- caller can
        ``torch.load(io.BytesIO(blob))`` to get a torch tensor back), or
        ``None`` if the receiver thread hasn't landed anything new since the
        previous ``poll``.

        Latest-wins semantics: if the receiver thread wrote multiple times
        between two ``poll`` calls, only the newest blob is returned.
        """
        if self._role != _ROLE_RECEIVER:
            raise RuntimeError("ThreadedHostWeightBridge.poll called on a non-receiver bridge.")
        with self._recv_pad_lock:
            data = self._recv_pad
            self._recv_pad = None
            return data

    def latest_version(self) -> int:
        """Receiver-only. Monotonic counter incremented every time the
        receiver thread lands a blob in the pad; useful for logging dropped
        intermediate versions.
        """
        if self._role != _ROLE_RECEIVER:
            raise RuntimeError("ThreadedHostWeightBridge.latest_version called on a non-receiver bridge.")
        with self._recv_pad_lock:
            return self._recv_pad_version

    def _receiver_loop(self) -> None:
        """Receiver thread body. Owns the MPI recv on the bridge's private
        context only -- no ttnn / no device work, no touching the world
        context. Runs until ``self._shutdown`` is set; if the thread is
        blocked in ``recv_bytes`` when shutdown fires, it dies with the
        process as a daemon.
        """
        assert self._ctx is not None, "connect() must have populated the context before the receiver starts"
        while not self._shutdown.is_set():
            try:
                (blob_len,) = struct.unpack("<Q", self._ctx.recv(8, self._peer_rank, _THREADED_WEIGHT_LEN_TAG))
                blob = self._ctx.recv(int(blob_len), self._peer_rank, _THREADED_WEIGHT_BLOB_TAG)
            except Exception as e:
                if self._shutdown.is_set():
                    return
                print(
                    f"[threaded-bridge receiver] recv error: {type(e).__name__}: {e}",
                    flush=True,
                )
                return

            with self._recv_pad_lock:
                self._recv_pad = blob
                self._recv_pad_version += 1
                v = self._recv_pad_version
            print(
                f"[threaded-bridge receiver] wrote pad v={v} ({len(blob)} B)",
                flush=True,
            )
