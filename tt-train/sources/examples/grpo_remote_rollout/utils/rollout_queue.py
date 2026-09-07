# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""SPSC cross-process rollout queue for the fully-async GRPO trainer.

Exercised end-to-end by the tests under
``tt-train/tests/python/grpo_remote_rollout/fully_async/``.

Design
======
Two Python processes. Rank 1 is the producer (rollout worker in the real
system). Rank 0 is the consumer (trainer). Each process has:

  * A local `queue.Queue(maxsize=capacity)` that decouples the main thread
    from the transport thread inside that process.
  * A background transport thread that owns one side of the MPI transfer.

Data flows through THIS pipeline:

  rank 1 main -> queue1.put -> queue1.get -> torch.save + MPI_Send -.
                                                                    |
                                                                    v
    rank 0 main <- queue0.get <- queue0.put <- torch.load + MPI_Recv

The two `queue.Queue` instances are separate objects in two address
spaces. MPI is the only interprocess link. Because `queue.Queue.put` and
`queue.Queue.get` block on `maxsize` and emptiness respectively, and
`MPI_Send` blocks when the wire buffer is full, back-pressure travels
from the slow consumer all the way to the producer main thread with no
extra bookkeeping.

Policy A back-pressure: the producer main thread's `push()` blocks if
the pipeline is full. No batch is lost.

MPI communicator
================
The queue calls `ttnn.distributed_context_duplicate()` at connect time,
which does `MPI_Comm_dup` under the hood. All queue traffic runs on that
private communicator so it does not collide with the weight bridge or
any other subsystem on the world context.

Wire format
===========
Fixed framing: `[u64 length]` on tag 0, then `[body bytes]` on tag 1.
Body is `torch.save` of a Python dict with the RolloutBatch fields
(``batch_id``, ``weight_version``, ``prompts``, ``completions``,
``logprobs``, ``extra``). The ``extra`` field is a plain-Python-builtins
dict (typically ``{"answer": [...]}``) that carries dataset columns the
consumer's reward functions need. ``length == 0`` is the close message
sent by the producer's `close()` to tell the consumer transport thread
to stop.
"""

from __future__ import annotations

import io
import queue
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

import ttnn

# `ttnn.distributed_context_duplicate()` returns a
# `tt::tt_metal::distributed::multihost::DistributedContext`. That C++
# type's nanobind class binding lives in the ttml module, not ttnn.
# Importing ttml registers the class so the duplicate call can hand its
# return value back to Python.
import ttml  # noqa: F401


_TAG_LEN: int = 0
_TAG_BODY: int = 1
_CLOSE_JOIN_TIMEOUT_S: float = 10.0


# Private in-process marker. Producer main thread puts this into the
# local queue during `close()`. Producer transport thread pops it, sees
# it is not a real batch, sends the length-0 close message over MPI, and
# exits.
class _StopMarker:
    pass


_STOP_MARKER = _StopMarker()


@dataclass(frozen=True)
class RolloutBatch:
    """One rollout batch shipped from the producer to the consumer.

    Fields:
        batch_id: Monotonic counter picked by the producer.
        weight_version: Which theta version produced this batch. The
            trainer can use it to detect / down-weight stale samples.
        prompts: B ragged prompt token IDs.
        completions: B ragged completion token IDs (one per prompt in
            single-generation mode; per prompt-completion pair otherwise).
        logprobs: [B, max_completion_length] float32. Per-generated-token
            log pi_old(a_t | s_t). Padding positions are don't-care; the
            trainer masks them out.
    """

    batch_id: int
    weight_version: int
    prompts: List[List[int]]
    completions: List[List[int]]
    logprobs: torch.Tensor

    # Dataset columns needed by the consumer's reward functions (e.g.
    # ``{"answer": ["42", ...]}`` for gsm8k). Values must be plain Python
    # builtins so ``torch.load(..., weights_only=True)`` accepts them.
    extra: dict = field(default_factory=dict, hash=False, compare=False)


class RolloutQueue:
    """SPSC cross-process rollout queue.

    Construct via `RolloutQueue.producer(...)` or `.consumer(...)`.
    Call `connect()` on both ranks. Producer calls `push(batch)`. Consumer
    calls `pop()` until it returns `None` (peer closed).
    """

    _ROLE_PRODUCER = "producer"
    _ROLE_CONSUMER = "consumer"

    def __init__(self, *, role: str, peer_rank: int, capacity: int) -> None:
        if role not in (self._ROLE_PRODUCER, self._ROLE_CONSUMER):
            raise ValueError(f"role must be producer or consumer, got {role!r}")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self._role: str = role
        self._peer_rank: int = int(peer_rank)
        self._capacity: int = int(capacity)

        # Local queue between main thread and transport thread in THIS process.
        # For the producer: main puts, transport gets.
        # For the consumer: transport puts, main gets.
        self._local: "queue.Queue[Any]" = queue.Queue(maxsize=self._capacity)

        # Populated by connect().
        self._ctx: Optional[Any] = None  # ttml DistributedContext
        self._thread: Optional[threading.Thread] = None

        self._shutdown: threading.Event = threading.Event()

    # ---- construction --------------------------------------------------------

    @classmethod
    def producer(cls, *, peer_rank: int, capacity: int = 2) -> "RolloutQueue":
        return cls(role=cls._ROLE_PRODUCER, peer_rank=peer_rank, capacity=capacity)

    @classmethod
    def consumer(cls, *, peer_rank: int, capacity: int = 2) -> "RolloutQueue":
        return cls(role=cls._ROLE_CONSUMER, peer_rank=peer_rank, capacity=capacity)

    # ---- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Duplicate the current world MPI context and start the transport
        thread.

        `ttnn.distributed_context_duplicate` is a collective; both ranks
        must call `connect()` in the same order for the private contexts
        to line up. It also acts as the connect-time barrier.
        """
        self._ctx = ttnn.distributed_context_duplicate()
        print(
            f"[rollout-queue {self._role}] duplicated MPI context "
            f"(rank={self._ctx.rank()}, size={self._ctx.size()}, "
            f"capacity={self._capacity})",
            flush=True,
        )

        if self._role == self._ROLE_PRODUCER:
            self._thread = threading.Thread(
                target=self._producer_loop,
                name="rollout-queue-producer",
                daemon=True,
            )
        else:
            self._thread = threading.Thread(
                target=self._consumer_loop,
                name="rollout-queue-consumer",
                daemon=True,
            )
        self._thread.start()

    def close(self) -> None:
        """Producer: flush the local queue, send the length-0 close message,
        join the transport thread. Consumer: signal the transport thread to
        stop; join if not blocked in MPI_Recv (see caveat in module docstring).
        Idempotent."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()

        if self._role == self._ROLE_PRODUCER:
            # Put a stop marker at the end of the local queue. The transport
            # thread will drain everything before it, then see the marker and
            # send the length-0 close message.
            self._local.put(_STOP_MARKER)
            if self._thread is not None:
                self._thread.join(timeout=_CLOSE_JOIN_TIMEOUT_S)
                if self._thread.is_alive():
                    print(
                        "[rollout-queue producer] WARNING: transport thread did "
                        f"not exit within {_CLOSE_JOIN_TIMEOUT_S}s",
                        flush=True,
                    )
        else:
            # Consumer transport thread might be blocked in MPI_Recv. It only
            # unblocks when the peer's close() sends the length-0 message.
            # The thread is a background thread and dies with the process
            # if the peer never closes cleanly.
            if self._thread is not None:
                self._thread.join(timeout=_CLOSE_JOIN_TIMEOUT_S)

    # ---- producer API --------------------------------------------------------

    def push(self, batch: RolloutBatch) -> None:
        """Producer-only, main-thread call. Blocks if the local queue is
        full (Policy A back-pressure)."""
        if self._role != self._ROLE_PRODUCER:
            raise RuntimeError("push() called on a non-producer queue")
        if self._shutdown.is_set():
            raise RuntimeError("push() called after close()")
        self._local.put(batch)

    def _producer_loop(self) -> None:
        """Producer transport thread body: get from local queue -> serialize
        -> MPI_Send on the private context. Exits on the stop marker after
        sending the length-0 close message."""
        assert self._ctx is not None
        while True:
            item = self._local.get()

            if item is _STOP_MARKER:
                # Send the length-0 close message and exit.
                try:
                    self._ctx.send(struct.pack("<Q", 0), self._peer_rank, _TAG_LEN)
                except Exception as e:
                    print(
                        f"[rollout-queue producer] close-message send failed: " f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                return

            blob = _serialize_batch(item)
            try:
                self._ctx.send(struct.pack("<Q", len(blob)), self._peer_rank, _TAG_LEN)
                self._ctx.send(blob, self._peer_rank, _TAG_BODY)
            except Exception as e:
                print(
                    f"[rollout-queue producer] MPI send failed: " f"{type(e).__name__}: {e}",
                    flush=True,
                )
                return

    # ---- consumer API --------------------------------------------------------

    def pop(self) -> Optional[RolloutBatch]:
        """Consumer-only, main-thread call. Blocks until a batch is
        available. Returns `None` after the peer has closed and the local
        queue is drained."""
        if self._role != self._ROLE_CONSUMER:
            raise RuntimeError("pop() called on a non-consumer queue")
        item = self._local.get()
        if item is None:
            # Transport thread put None to signal shutdown drained.
            return None
        return item

    def qsize(self) -> int:
        """Approximate number of batches currently sitting in the local
        queue on this rank."""
        return self._local.qsize()

    def _consumer_loop(self) -> None:
        """Consumer transport thread body: MPI_Recv -> deserialize ->
        put in local queue. Exits when the peer sends the length-0
        close message."""
        assert self._ctx is not None
        while True:
            try:
                raw_len = self._ctx.recv(8, self._peer_rank, _TAG_LEN)
            except Exception as e:
                print(
                    f"[rollout-queue consumer] length recv failed: " f"{type(e).__name__}: {e}",
                    flush=True,
                )
                self._local.put(None)
                return
            (blob_len,) = struct.unpack("<Q", raw_len)

            if blob_len == 0:
                # Peer sent the close message. Signal main thread's
                # blocking pop() and exit.
                self._local.put(None)
                return

            try:
                blob = self._ctx.recv(int(blob_len), self._peer_rank, _TAG_BODY)
            except Exception as e:
                print(
                    f"[rollout-queue consumer] body recv failed: " f"{type(e).__name__}: {e}",
                    flush=True,
                )
                self._local.put(None)
                return

            batch = _deserialize_batch(blob)
            # Blocks if the consumer's local queue is full. That is the
            # back-pressure signal to the wire and then to the producer.
            self._local.put(batch)


# ---- serialization ---------------------------------------------------------


def _serialize_batch(batch: RolloutBatch) -> bytes:
    """Pack a RolloutBatch into bytes via torch.save on a plain dict."""
    payload = {
        "batch_id": batch.batch_id,
        "weight_version": batch.weight_version,
        "prompts": batch.prompts,
        "completions": batch.completions,
        "logprobs": batch.logprobs,
        "extra": dict(batch.extra),
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _deserialize_batch(blob: bytes) -> RolloutBatch:
    payload = torch.load(io.BytesIO(blob), weights_only=True)
    return RolloutBatch(
        batch_id=int(payload["batch_id"]),
        weight_version=int(payload["weight_version"]),
        prompts=list(payload["prompts"]),
        completions=list(payload["completions"]),
        logprobs=payload["logprobs"],
        extra=dict(payload.get("extra", {})),
    )
