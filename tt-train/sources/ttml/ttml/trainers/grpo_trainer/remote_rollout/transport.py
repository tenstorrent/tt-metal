# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Length-prefixed frames between two ranks, and the private communicators that carry them."""

from __future__ import annotations

import struct
from typing import Callable, Optional

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
