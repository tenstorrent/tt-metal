# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stored value of one media item: a Header, then n_tokens bf16 embeddings of width hidden.

Written by the media module's encode and read by the prefill runtime, which knows n_tokens from the media ref and
takes the media kind from the header. Header and embeddings take a C-contiguous view of any item format, count it in
bytes, and keep no view of it when they raise.
"""

import struct
from dataclasses import dataclass

import numpy as np

KINDS = ("image",)
_HEADER = struct.Struct("<B15x")  # kind, padded so the rows stay 16-byte aligned
_BF16_BYTES = 2


@dataclass(frozen=True)
class Header:
    kind: str

    def write(self, buf: memoryview) -> None:
        _HEADER.pack_into(buf, 0, KINDS.index(self.kind))

    @classmethod
    def read(cls, buf: memoryview) -> "Header":
        nbytes = memoryview(buf).nbytes
        if nbytes < _HEADER.size:
            raise ValueError(f"stored value of {nbytes} B is shorter than its {_HEADER.size} B header")
        (kind,) = _HEADER.unpack_from(buf)
        if kind >= len(KINDS):
            raise ValueError(f"unknown media kind {kind}")
        return cls(KINDS[kind])


def stored_nbytes(n_tokens: int, hidden: int) -> int:
    return _HEADER.size + n_tokens * hidden * _BF16_BYTES


def embeddings(buf: memoryview, hidden: int) -> np.ndarray:
    """[n_tokens, hidden] view of the rows after the header as bf16 bit patterns (uint16). Row i is the i-th media
    token's."""
    nbytes = memoryview(buf).nbytes
    if nbytes <= _HEADER.size or (nbytes - _HEADER.size) % (hidden * _BF16_BYTES):
        raise ValueError(f"stored value of {nbytes} B is not a header and whole bf16 rows of width {hidden}")
    return np.frombuffer(memoryview(buf).cast("B"), dtype=np.uint16, offset=_HEADER.size).reshape(-1, hidden)
