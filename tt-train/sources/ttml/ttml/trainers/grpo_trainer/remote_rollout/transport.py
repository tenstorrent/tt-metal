# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Length-prefixed frames between two ranks, and the private communicators that carry them."""

from __future__ import annotations

import io
import struct
from typing import Any, Callable, Iterator, List, Optional, Tuple

import torch
import ttml


def duplicate_distributed_context():
    """Collective: both ranks must call it in the same order."""
    return ttml.autograd.AutoContext.get_instance().get_distributed_context().duplicate()


def send_frame(send: Callable, data: bytes, peer_rank: int, len_tag: int, body_tag: int) -> None:
    """Sends the length of ``data``, then ``data`` if it is not empty."""
    send(struct.pack("<Q", len(data)), peer_rank, len_tag)
    if data:
        send(data, peer_rank, body_tag)


def recv_frame(recv: Callable, peer_rank: int, len_tag: int, body_tag: int) -> Optional[bytes]:
    """Returns None for an empty frame."""
    (n,) = struct.unpack("<Q", recv(8, peer_rank, len_tag))
    return recv(int(n), peer_rank, body_tag) if n else None


def _torch_save_bytes(t) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _torch_load_bytes(blob: bytes):
    return torch.load(io.BytesIO(blob), weights_only=True)


def send_weight_dict(
    send: Callable, weights: dict, read: Callable, peer_rank: int, len_tag: int, body_tag: int
) -> None:
    """Sends ``read(weights[k])``, a host torch tensor, for each key in sorted order."""
    for k in sorted(weights):
        send_frame(send, _torch_save_bytes(read(weights[k])), peer_rank, len_tag, body_tag)


def recv_weight_dict(
    recv: Callable, entries: List[dict], peer_rank: int, len_tag: int, body_tag: int
) -> Iterator[Tuple[dict, Any]]:
    """Yields (entry, host torch tensor) for each manifest entry, in order."""
    for e in entries:
        yield e, _torch_load_bytes(recv_frame(recv, peer_rank, len_tag, body_tag))
