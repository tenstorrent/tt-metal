# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Typed MPI event channel between the training and inference ranks of a
fully-async GRPO example (``gsm8k_fully_async``).

Every event is a fixed 8-byte message: ``<event_id:u32><payload:u32>``. Because
the size is fixed, ``poll()`` only needs to probe for message *presence* via
:func:`ttnn.distributed_context_iprobe_bytes` -- there is no manifest, no
persistent posted irecv, no request bookkeeping. ``send()`` uses the existing
blocking :func:`ttnn.distributed_context_send_bytes`; 8-byte messages sit
comfortably inside MPI's eager threshold and return effectively instantly.

MPI tag ``22200`` is disjoint from :mod:`utils.weight_bridge` (22101-22106) and
:mod:`utils.mpi_rollout` (22001-22013), so all three protocols coexist in the
same distributed context. Direction is disambiguated by the peer rank supplied
to ``recv_bytes`` / ``iprobe_bytes``.
"""

from __future__ import annotations

import struct
from enum import IntEnum
from typing import Optional, Tuple

import ttnn

# 8 bytes: event_id (u32) + payload (u32), little-endian.
_MSG_FMT: str = "<II"
_MSG_LEN: int = struct.calcsize(_MSG_FMT)

# Single tag pair; the source-rank filter on ``recv_bytes`` /
# ``iprobe_bytes`` keeps the two directions disjoint.
_MSG_TAG: int = 22200


class AsyncTrainingEvent(IntEnum):
    """Typed events exchanged between the training rank and the inference rank."""

    # Training -> inference: initial payload is completions-per-batch (u32).
    TRAINING_BATCH_SIZE = 1
    # Training -> inference: about to push a fresh weight version.
    TRAINING_ABOUT_TO_SEND_WEIGHTS = 2
    # Training -> inference: the receiving pad now holds the new weights.
    TRAINING_SENT_WEIGHTS = 3
    # Inference -> training: the pad contents have been installed into the model.
    INFERENCE_RECEIVED_WEIGHTS = 4
    # Training -> inference: no more steps; drop out of the generation loop.
    TRAINING_STOPPED = 5


class AsyncTrainingEventChannel:
    """Bidirectional 8-byte event channel for one peer rank.

    Both sides construct one with the peer's rank; sends and receives filter
    on that rank so the two directions do not collide on the shared tag.

    Lifecycle: construct, then use ``send`` / ``poll`` / ``wait_for_next_event``
    freely. There is no ``connect()`` handshake (each event is standalone) and
    no ``close()`` (no persistent state to unwind).
    """

    def __init__(self, *, peer_rank: int, tag: int = _MSG_TAG) -> None:
        self._peer: int = int(peer_rank)
        self._tag: int = int(tag)

    # ---- send --------------------------------------------------------------

    def send(self, event: AsyncTrainingEvent, payload: int = 0) -> None:
        """Blocking MPI send of a single 8-byte event to the peer.

        The 8-byte body fits within MPI's eager threshold on every reasonable
        transport, so this returns effectively instantly without engaging
        the receiver.
        """
        body = struct.pack(_MSG_FMT, int(event), int(payload))
        ttnn.distributed_context_send_bytes(body, self._peer, self._tag)

    # ---- receive -----------------------------------------------------------

    def poll(self) -> Optional[Tuple[AsyncTrainingEvent, int]]:
        """Non-blocking check for a pending event.

        Returns ``None`` if nothing is currently waiting, otherwise
        consumes the pending message and returns ``(event, payload)``.
        """
        size = ttnn.distributed_context_iprobe_bytes(self._peer, self._tag)
        if size is None:
            return None
        if size != _MSG_LEN:
            raise RuntimeError(
                f"AsyncTrainingEventChannel: expected {_MSG_LEN}-byte event on tag {self._tag} "
                f"from rank {self._peer}, got {size} bytes"
            )
        return self._consume()

    def wait_for_next_event(self) -> Tuple[AsyncTrainingEvent, int]:
        """Blocking receive of the next event from the peer."""
        return self._consume()

    def _consume(self) -> Tuple[AsyncTrainingEvent, int]:
        body = ttnn.distributed_context_recv_bytes(_MSG_LEN, self._peer, self._tag)
        event_id, payload = struct.unpack(_MSG_FMT, body)
        return AsyncTrainingEvent(event_id), int(payload)
