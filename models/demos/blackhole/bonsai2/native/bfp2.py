# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packing ternary weights into BFP2 tiles for the Tensix matmul.

A BFP2 32x32 tile is 320 B: 64 exponent bytes (one shared 8-bit exponent per
16-element face row, in face order) followed by 256 bytes of 2-bit data
(4 faces x 16 rows x 4 bytes, row-major inside each face).

The 2-bit datum is ``[sign, magnitude]``:

    code    value
    0b00    0
    0b01    +1
    0b10    -Inf      <-- never emit
    0b11    -1

So a ternary weight maps straight through with no arithmetic::

    trit  -1  ->  0b11
    trit   0  ->  0b00
    trit  +1  ->  0b01

and since every magnitude is exactly 1, every face row that holds any nonzero
weight gets exponent 127.  That is what makes the "forced shared exponent"
variant possible: drop the 64 exponent bytes entirely and pass a constant
shared exponent to the unpacker, taking the tile from 320 B to 256 B.

``pack_tiles`` is vectorized numpy.  ``check_against_ttnn`` cross-checks it
against ttnn's C++ packer on random ternary tiles, so the fast path in this
module and the hardware see the same bytes.
"""

from __future__ import annotations

import numpy as np

TILE = 32
FACE = 16
FACE_BYTES = FACE * FACE // 4  # 64 B of 2-bit data per face
EXP_BYTES = 64  # 4 faces x 16 rows, one exponent byte each
BFP2_TILE_BYTES = EXP_BYTES + 4 * FACE_BYTES  # 320
BFP2_SHARED_EXP_TILE_BYTES = 4 * FACE_BYTES  # 256, no exponent section

#: Exponent byte for |value| == 1.0.
EXP_ONE = 127

#: Trit (-1, 0, +1) -> 2-bit datum.  Indexed by ``trit + 1``.
TRIT_TO_CODE = np.array([0b11, 0b00, 0b01], dtype=np.uint8)


def trits_to_codes(trits: np.ndarray) -> np.ndarray:
    """Map int8 trits in {-1, 0, +1} to 2-bit BFP2 data.  Shape preserved."""
    t = np.asarray(trits)
    if t.size and (t.min() < -1 or t.max() > 1):
        raise ValueError("trits must be in {-1, 0, +1}")
    return TRIT_TO_CODE[t.astype(np.int32) + 1]


def as_codes(x: np.ndarray) -> np.ndarray:
    """Coerce either trits or 2-bit codes to a uint8 code array."""
    c = np.asarray(x)
    if c.dtype == np.int8 and c.size and (c < 0).any():
        return trits_to_codes(c)
    if c.size and (c.min() < 0 or c.max() > 3):
        raise ValueError(f"expected trits or 0..3 codes, got [{c.min()}, {c.max()}]")
    return c.astype(np.uint8)


def pack_face_codes(codes: np.ndarray) -> np.ndarray:
    """Pack a 16x16 face of 2-bit codes to 64 bytes (row-major inside the face).

    Four codes per byte, element ``4k + j`` of a row landing at bits
    ``2j .. 2j+1`` of byte ``k``.
    """
    c = np.asarray(codes).reshape(FACE, FACE).astype(np.uint8)
    b = (c.reshape(FACE, 4, 4).astype(np.uint16) << np.array([0, 2, 4, 6], dtype=np.uint16)).sum(axis=2)
    return b.astype(np.uint8).reshape(FACE * 4)


def pack_tile_codes_only(codes: np.ndarray) -> np.ndarray:
    """Pack one 32x32 code tile to its 256 data bytes (no exponent section).

    Face order is ttnn's: (0,0), (0,1), (1,0), (1,1) -- top-left, top-right,
    bottom-left, bottom-right of the 32x32 tile.
    """
    c = as_codes(codes).reshape(TILE, TILE)
    parts = [
        pack_face_codes(c[0:FACE, 0:FACE]),
        pack_face_codes(c[0:FACE, FACE:TILE]),
        pack_face_codes(c[FACE:TILE, 0:FACE]),
        pack_face_codes(c[FACE:TILE, FACE:TILE]),
    ]
    return np.concatenate(parts)


def pack_tile_exponents(codes: np.ndarray) -> np.ndarray:
    """The 64 exponent bytes for one 32x32 ternary code tile.

    Every face row is one exponent byte: 127 (|1.0|) if that row holds any
    nonzero weight, else 0.
    """
    c = as_codes(codes).reshape(TILE, TILE)
    parts = []
    for r0, r1 in ((0, FACE), (FACE, TILE)):
        for c0, c1 in ((0, FACE), (FACE, TILE)):
            face = c[r0:r1, c0:c1]
            parts.append(np.where(np.any(face != 0, axis=1), EXP_ONE, 0).astype(np.uint8))
    return np.concatenate(parts)


def pack_tile(codes: np.ndarray, shared_exp: bool = False) -> np.ndarray:
    """One 32x32 ternary tile -> packed BFP2 bytes.

    ``shared_exp=True`` emits the 256-byte form (no exponent section) for the
    forced-shared-exponent path; otherwise the standard 320-byte Bfp2_b tile.
    """
    data = pack_tile_codes_only(codes)
    if shared_exp:
        return data
    return np.concatenate([pack_tile_exponents(codes), data])


def pack_tiles(codes: np.ndarray, shared_exp: bool = False) -> np.ndarray:
    """Pack ``[num_tiles, 32, 32]`` code tiles -> ``[num_tiles, bytes]`` uint8.

    ``codes`` are 2-bit data values (0..3).  A signed int8 array of *trits* is
    also accepted and mapped through :func:`trits_to_codes`, since callers
    naturally have +-1 in hand and the two representations differ only by the
    index shift.
    """
    c = as_codes(codes)
    if c.ndim != 3 or c.shape[1:] != (TILE, TILE):
        raise ValueError(f"expected [n, 32, 32] code tiles, got {c.shape}")
    n = c.shape[0]
    # Data: [n, 4 faces, 16 rows, 4 bytes]
    f = c.reshape(n, 2, FACE, 2, FACE)  # n, tr, r, tc, col
    f = f.transpose(0, 1, 3, 2, 4)  # n, tr, tc, row, col  -> face order (0,0),(0,1),(1,0),(1,1)
    fb = (f.reshape(n, 4, FACE, 4, 4).astype(np.uint16) << np.array([0, 2, 4, 6], dtype=np.uint16)).sum(axis=4)
    data = fb.astype(np.uint8).reshape(n, BFP2_SHARED_EXP_TILE_BYTES)
    if shared_exp:
        return data
    # One exponent byte per face row: a 32x32 tile is 2x2 faces of 16x16, so 64
    # face rows.  Order is face-major (tr, tc) then row, matching the data section.
    face_rows = c.reshape(n, 2, FACE, 2, FACE)  # n, tr, row, tc, col
    exps = np.where(np.any(face_rows != 0, axis=4), EXP_ONE, 0).astype(np.uint8)  # n, tr, row, tc
    e = exps.transpose(0, 1, 3, 2).reshape(n, EXP_BYTES)
    return np.concatenate([e, data], axis=1)


def unpack_tile_codes(packed: np.ndarray, shared_exp: bool = False) -> np.ndarray:
    """Inverse of :func:`pack_tile`: packed bytes -> ``[32, 32]`` 2-bit codes."""
    p = np.asarray(packed).ravel()
    data = p if shared_exp else p[EXP_BYTES:]
    if data.size != BFP2_SHARED_EXP_TILE_BYTES:
        raise ValueError(f"expected {BFP2_SHARED_EXP_TILE_BYTES} data bytes, got {data.size}")
    faces = data.reshape(4, FACE, 4)  # face, row, byte-group
    out = np.empty((4, FACE, FACE), dtype=np.uint8)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    for k in range(4):  # byte k of a row holds columns 4k .. 4k+3
        out[:, :, 4 * k : 4 * k + 4] = (faces[:, :, k][:, :, None] >> shifts) & 0x3
    c = out.reshape(2, 2, FACE, FACE)  # tr, tc, row, col
    return c.transpose(0, 2, 1, 3).reshape(TILE, TILE)


def codes_to_trits(codes: np.ndarray) -> np.ndarray:
    """Inverse of :func:`trits_to_codes`; raises on the invalid ``0b10`` datum."""
    c = np.asarray(codes)
    if np.any(c == 0b10):
        raise ValueError("BFP2 datum 0b10 decodes as -Inf and must not appear in a ternary tile")
    return np.where(c == 0, 0, np.where(c == 1, 1, -1)).astype(np.int8)


def check_against_ttnn(seed: int = 0) -> None:
    """Cross-check :func:`pack_tiles` against ttnn's C++ BFP2 packer.

    Packs random ternary tiles both ways and compares bytes.  The C++ packer
    is the definition of what the hardware unpacks, so this pins the layout
    (face order, exponent placement, bit order) rather than trusting the
    derivation above.
    """
    import ttnn

    rng = np.random.default_rng(seed)
    trits = rng.integers(-1, 2, size=(8, TILE, TILE)).astype(np.int8)
    codes = trits_to_codes(trits)

    mine = pack_tiles(codes, shared_exp=False)
    # ttnn's packer wants float32 in row-major tile order and returns uint32.
    vals = trits.astype(np.float32).reshape(-1)
    ref_u32 = np.asarray(ttnn._ttnn.bfp_utils.pack_bfp2(vals, True, False))
    ref = ref_u32.view(np.uint8).reshape(8, BFP2_TILE_BYTES)
    if not np.array_equal(mine, ref):
        bad = np.argwhere(np.any(mine != ref, axis=1)).ravel()
        raise AssertionError(f"BFP2 pack mismatch on tiles {bad.tolist()}; first: mine={mine[bad[0]][:8]} ref={ref[bad[0]][:8]}")
