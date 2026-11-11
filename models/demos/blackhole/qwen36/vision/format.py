# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stored value of one Qwen3.5 media item: a 64-byte header, then n_tokens bf16 embeddings of width hidden (4096 for
the 9B). Written by the media module's encode, read by the prefill runner.

The header carries what prefill needs besides the rows (the encoder-service design doc: "per-image metadata travels
inside the stored value, so nothing extra passes through the engine"). For Qwen3.5 that is the image grid: M-RoPE
gives each image token a (t, row, col) position and the text after an image resumes at max(grid_h, grid_w) / 2 past
it, so a 32 x 16 and a 16 x 32 grid (same n_tokens) need different positions; decode's rope delta follows from the same
grids. Row i is the i-th image token's embedding, in the order the <|image_pad|> run appears in the prompt (the HF
processor's 2x2 merge-block order).

Header (little-endian, 64 bytes; the rows start 64-byte aligned):
  0  u32 magic            b"Q35M"
  4  u16 version          1
  6  u16 media_type       1 = image (2 = video, 3 = audio: reserved)
  8  u32 n_tokens         rows that follow (= grid_t * grid_h * grid_w / merge^2)
  12 u32 hidden           row width
  16 u16 dtype            1 = bf16
  18 u16 merge            spatial merge size (2)
  20 u32 grid_t           HF image_grid_thw
  24 u32 grid_h
  28 u32 grid_w
  32 u32 n_refs           images in the request (picked the pixel cap)
  36 ..64                 zero
"""

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"Q35M"
VERSION = 1
HEADER_BYTES = 64
MEDIA_IMAGE = 1
DTYPE_BF16 = 1
_BF16_BYTES = 2
_LAYOUT = struct.Struct("<4sHHIIHHIIII")  # 36 bytes; the rest of the header is zero


@dataclass(frozen=True)
class Header:
    n_tokens: int
    hidden: int
    grid_thw: tuple[int, int, int]
    n_refs: int = 1
    merge: int = 2
    media_type: int = MEDIA_IMAGE

    def __post_init__(self):
        t, h, w = self.grid_thw
        if min(t, h, w) <= 0 or h % self.merge or w % self.merge:
            raise ValueError(f"grid {self.grid_thw} is not a positive grid of whole {self.merge}x{self.merge} blocks")
        if t * h * w != self.n_tokens * self.merge * self.merge:
            raise ValueError(f"grid {self.grid_thw} holds {t * h * w // self.merge ** 2} tokens, not {self.n_tokens}")
        if self.hidden <= 0:
            raise ValueError(f"hidden {self.hidden}")


def stored_nbytes(n_tokens: int, hidden: int) -> int:
    """value_size: header + rows."""
    return HEADER_BYTES + n_tokens * hidden * _BF16_BYTES


def write_header(buf: memoryview, header: Header) -> None:
    if len(buf) != stored_nbytes(header.n_tokens, header.hidden):
        raise ValueError(f"buffer of {len(buf)} B for a value of {stored_nbytes(header.n_tokens, header.hidden)} B")
    raw = _LAYOUT.pack(
        MAGIC, VERSION, header.media_type, header.n_tokens, header.hidden, DTYPE_BF16, header.merge, *header.grid_thw,
        header.n_refs,
    )
    buf[:HEADER_BYTES] = raw + bytes(HEADER_BYTES - len(raw))


def read_header(buf) -> Header:
    """The header of a stored value; checks magic, version, dtype and that the value is exactly header + rows."""
    if len(buf) < HEADER_BYTES:
        raise ValueError(f"stored value of {len(buf)} B is shorter than the {HEADER_BYTES} B header")
    magic, version, media_type, n_tokens, hidden, dtype, merge, t, h, w, n_refs = _LAYOUT.unpack_from(buf, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"not a Qwen3.5 v{VERSION} media value (magic {magic!r}, version {version})")
    if media_type != MEDIA_IMAGE or dtype != DTYPE_BF16:
        raise ValueError(f"media type {media_type} / dtype {dtype}: only bf16 images (1 / 1)")
    header = Header(n_tokens, hidden, (t, h, w), n_refs, merge, media_type)
    if len(buf) != stored_nbytes(n_tokens, hidden):
        raise ValueError(f"stored value of {len(buf)} B, its header says {stored_nbytes(n_tokens, hidden)} B")
    return header


def embeddings(buf, hidden: int) -> np.ndarray:
    """[n_tokens, hidden] view of the value's rows as bf16 bit patterns (uint16), in place (no copy). Takes the whole
    value (header included); the row count comes from the buffer size, so the encoder can fill the rows before or after
    the header."""
    rows = len(buf) - HEADER_BYTES
    if rows <= 0 or rows % (hidden * _BF16_BYTES):
        raise ValueError(f"stored value of {len(buf)} B is not a {HEADER_BYTES} B header + whole bf16 rows of {hidden}")
    return np.frombuffer(buf, dtype=np.uint16, offset=HEADER_BYTES).reshape(-1, hidden)


def parse(buf) -> tuple[Header, np.ndarray]:
    """The prefill runner's read: (header, [n_tokens, hidden] uint16 rows view)."""
    header = read_header(buf)
    return header, embeddings(buf, header.hidden)
