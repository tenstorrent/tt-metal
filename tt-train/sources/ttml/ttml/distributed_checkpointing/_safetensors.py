# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A minimal safetensors container: one file per device, every chunk a tensor in it.

The format is the published one, so any device file opens with the ``safetensors`` package or the C++ reader:
an 8-byte little-endian header length, a JSON header mapping tensor names to ``dtype``, ``shape`` and
``data_offsets`` (relative to the data section), and the data section. The header is padded with spaces so the
data section starts on a 4 KiB boundary, which the spec allows. Tensors are laid out back to back in the order given.

The ``safetensors`` package itself is not used: the writer here needs to know every offset before a byte moves, so
that per-device threads can write their chunks at fixed offsets.
"""

from __future__ import annotations

import json
import os
import struct
from math import prod
from typing import Sequence

ALIGNMENT = 4096
ITEMSIZE = {"BF16": 2, "F32": 4, "U32": 4, "I32": 4, "U16": 2, "U8": 1, "I8": 1}


def build_header(
    tensors: Sequence[tuple[str, str, Sequence[int], int]], metadata: dict[str, str] | None = None
) -> tuple[bytes, dict[str, tuple[int, int]], int]:
    """The file header for ``tensors`` given as ``(name, dtype, shape, nbytes)`` in file order.

    Returns the header bytes including the length prefix, each tensor's ``(begin, end)`` within the data section,
    and the total size of the data section. ``metadata`` becomes the file's ``__metadata__``, whose values safetensors
    requires to be strings.

    Each tensor's entry is formatted as one string, the JSON ``json.dumps`` would give its dict: a save on 32
    devices writes headers of about 16k tensors each, where building a dict per tensor for the encoder to walk took
    twice as long.
    """
    offsets: dict[str, tuple[int, int]] = {}
    parts: list[str] = []
    if metadata is not None:
        for key, value in metadata.items():
            if not isinstance(value, str):
                raise ValueError(f"safetensors: metadata {key!r} must be a string, not {type(value).__name__}")
        parts.append('"__metadata__":' + json.dumps(metadata, separators=(",", ":")))
    cursor = 0
    for name, dtype, shape, nbytes in tensors:
        if name in offsets or (name == "__metadata__" and metadata is not None):
            raise ValueError(f"safetensors: tensor {name!r} listed twice in one file")
        itemsize = ITEMSIZE.get(dtype)
        if itemsize is None:
            raise ValueError(f"safetensors: unsupported dtype {dtype!r}")
        dims = [int(s) for s in shape]
        nbytes = int(nbytes)
        if nbytes != itemsize * prod(dims):
            raise ValueError(
                f"safetensors: {name!r} has {nbytes} bytes but shape {dims} of {dtype} needs {itemsize * prod(dims)}"
            )
        end = cursor + nbytes
        offsets[name] = (cursor, end)
        shape_json = ",".join(map(str, dims))
        parts.append(f'{json.dumps(name)}:{{"dtype":"{dtype}","shape":[{shape_json}],"data_offsets":[{cursor},{end}]}}')
        cursor = end
    body = ("{" + ",".join(parts) + "}").encode("utf-8")
    padding = (-(8 + len(body))) % ALIGNMENT
    body += b" " * padding
    return struct.pack("<Q", len(body)) + body, offsets, cursor


def create_file(path: str, header: bytes, data_bytes: int) -> int:
    """Write the header and size the file for its data section; returns the data section's offset in the file."""
    with open(path, "wb") as f:
        f.write(header)
        f.truncate(len(header) + data_bytes)
    return len(header)


def read_file_header(path: str) -> tuple[dict, int]:
    """The parsed header of a safetensors file and the offset of its data section."""
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"safetensors: {path} is too short for a header")
        (length,) = struct.unpack("<Q", prefix)
        body = f.read(length)
    if len(body) != length:
        raise ValueError(f"safetensors: {path} is truncated inside its header")
    return json.loads(body), 8 + length


def pread_all(fd: int, nbytes: int, offset: int) -> bytes:
    """Read exactly ``nbytes`` at ``offset`` (``os.pread`` may return short reads on some filesystems)."""
    chunks = []
    remaining, cursor = nbytes, offset
    while remaining:
        piece = os.pread(fd, remaining, cursor)
        if not piece:
            raise EOFError(f"safetensors: read past the end of the file at offset {cursor}")
        chunks.append(piece)
        remaining -= len(piece)
        cursor += len(piece)
    return b"".join(chunks)


def pwrite_all(fd: int, data, offset: int) -> None:
    """Write all of ``data`` (any buffer) at ``offset``."""
    view = memoryview(data).cast("B")
    cursor = 0
    while cursor < len(view):
        written = os.pwrite(fd, view[cursor:], offset + cursor)
        if written <= 0:
            raise OSError(f"safetensors: short write at offset {offset + cursor}")
        cursor += written
