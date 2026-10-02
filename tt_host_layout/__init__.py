# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Decode host copies of tiled and row-major block-float tensors.

Pure NumPy. Importing this module does not load ttnn or open a device.
"""

import numpy as np

CHUNK_ROWS = 32
BFP8_TILE_BYTES = 1088
BF16_TILE_BYTES = 2048


def _tile_faces_to_rows(values):
    """(n_tiles, 4, 16, 16) face-major values -> row-major (32, n_tiles * 32)."""
    n_tiles = values.shape[0]
    by_face = values.reshape(n_tiles, 2, 2, 16, 16).transpose(0, 1, 3, 2, 4)
    rows = by_face.reshape(n_tiles, CHUNK_ROWS, CHUNK_ROWS).transpose(1, 0, 2)
    return np.ascontiguousarray(rows.reshape(CHUNK_ROWS, -1))


def decode_chunk(buf, *, dtype="bfp8", width=576, storage="tile"):
    """One 32-row chunk -> float32 array of shape [32, width].

    ``storage="tile"`` accepts ``dtype="bfp8"`` (1088 B per 32x32 tile: 64 shared
    exponents, then 1024 mantissa bytes) and ``dtype="bf16"`` (2048 B per tile,
    four 16x16 faces). BFP8 values are ``sign * (mantissa & 0x7f) * 2**(exponent - 133)``,
    the same reconstruction as ``unpack_bfp8_tiles_into_float_vec`` for normalized mantissas.
    ``storage="row-major"`` accepts contiguous little-endian BF16 with no row padding.
    """
    raw = memoryview(buf)
    if storage == "row-major":
        if dtype != "bf16" or len(raw) != CHUNK_ROWS * width * 2:
            raise ValueError("unsupported row-major geometry")
        decoded = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        return decoded.reshape(CHUNK_ROWS, width).copy()
    if storage != "tile":
        raise ValueError(f"unsupported storage layout {storage!r}")
    if dtype not in ("bfp8", "bf16") or width % CHUNK_ROWS:
        raise ValueError("unsupported or inconsistent tile geometry")
    n_tiles = width // CHUNK_ROWS
    tile_bytes = BFP8_TILE_BYTES if dtype == "bfp8" else BF16_TILE_BYTES
    if len(raw) != n_tiles * tile_bytes:
        raise ValueError(f"chunk is {len(raw)} B, expected {n_tiles * tile_bytes}")
    if dtype == "bf16":
        values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        return _tile_faces_to_rows(values.reshape(n_tiles, 4, 16, 16))
    tiles = np.frombuffer(raw, dtype=np.uint8).reshape(n_tiles, BFP8_TILE_BYTES)
    exponents = tiles[:, :64].astype(np.int32).reshape(n_tiles, 4, 16)
    mantissas = tiles[:, 64:].reshape(n_tiles, 4, 16, 16)
    magnitude = (mantissas & 0x7F).astype(np.float32)
    scale = np.exp2((exponents - 133).astype(np.float32))[..., None]
    values = np.where(mantissas & 0x80, -(magnitude * scale), magnitude * scale)
    return _tile_faces_to_rows(values)


def row_bytes(buf, start, end, *, dtype="bfp8", width=576, storage="tile"):
    """Raw bytes backing rows ``[start, end)`` of one 32-row chunk, tiling included."""
    if not 0 <= start < end <= CHUNK_ROWS:
        raise ValueError(f"row range [{start}, {end}) is outside a {CHUNK_ROWS}-row chunk")
    raw = bytes(buf)
    if storage == "row-major":
        if dtype != "bf16":
            raise ValueError("row-major byte comparison requires BF16")
        return raw[start * width * 2 : end * width * 2]
    if storage != "tile" or dtype not in ("bfp8", "bf16") or width % CHUNK_ROWS:
        raise ValueError("unsupported or inconsistent tile geometry")
    tile_bytes = BFP8_TILE_BYTES if dtype == "bfp8" else BF16_TILE_BYTES
    if len(raw) != width // CHUNK_ROWS * tile_bytes:
        raise ValueError(f"chunk is {len(raw)} B, expected {width // CHUNK_ROWS * tile_bytes}")
    out = bytearray()
    for tile in range(width // CHUNK_ROWS):
        base = tile * tile_bytes
        for row in range(start, end):
            for col_face in range(2):
                face, row_in_face = row // 16 * 2 + col_face, row % 16
                if dtype == "bfp8":
                    out.append(raw[base + face * 16 + row_in_face])
                    offset = base + 64 + face * 256 + row_in_face * 16
                    out.extend(raw[offset : offset + 16])
                else:
                    offset = base + face * 512 + row_in_face * 32
                    out.extend(raw[offset : offset + 32])
    return bytes(out)
