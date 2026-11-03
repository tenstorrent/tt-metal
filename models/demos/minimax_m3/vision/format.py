# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stored value of one media item: a 64-byte header, then n_tokens bf16 embeddings of width hidden.

Written by the media module's encode and parsed by the prefill runtime. Little-endian header words:
grid_h, grid_w, n_tokens, hidden; the rest of the header is zero.
"""

import struct
from dataclasses import dataclass

import numpy as np

HEADER_BYTES = 64
_HEADER = struct.Struct("<IIII")
_BF16_BYTES = 2


@dataclass(frozen=True)
class Header:
    grid_h: int
    """Image-token rows: resized height / 28."""
    grid_w: int
    """Image-token columns: resized width / 28."""
    n_tokens: int
    """Embeddings stored, grid_h * grid_w; the start and end tokens have none."""
    hidden: int


def value_size(n_tokens: int, hidden: int) -> int:
    return HEADER_BYTES + n_tokens * hidden * _BF16_BYTES


def write_header(buf: memoryview, h: Header) -> None:
    assert h.n_tokens == h.grid_h * h.grid_w, h
    assert len(buf) == value_size(h.n_tokens, h.hidden), f"buffer of {len(buf)} B for {h}"
    buf[:HEADER_BYTES] = _HEADER.pack(h.grid_h, h.grid_w, h.n_tokens, h.hidden).ljust(HEADER_BYTES, b"\0")


def read_header(buf: memoryview) -> Header:
    if len(buf) < HEADER_BYTES:
        raise ValueError(f"stored value of {len(buf)} B is shorter than its header")
    h = Header(*_HEADER.unpack_from(buf))
    if h.n_tokens != h.grid_h * h.grid_w:
        raise ValueError(f"{h}: n_tokens is not grid_h * grid_w")
    if len(buf) != value_size(h.n_tokens, h.hidden):
        raise ValueError(f"stored value of {len(buf)} B, {h} needs {value_size(h.n_tokens, h.hidden)} B")
    return h


def embeddings(buf: memoryview, h: Header) -> np.ndarray:
    """[n_tokens, hidden] view of the payload as bf16 bit patterns (uint16). Row i is the i-th image token's."""
    return np.frombuffer(buf, dtype=np.uint16, count=h.n_tokens * h.hidden, offset=HEADER_BYTES).reshape(
        h.n_tokens, h.hidden
    )
