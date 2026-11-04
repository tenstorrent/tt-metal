# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Block scales for the 2D exact prefill linear (linear2.py), straight from the decode records.

The SFPU multiplies a product tile P (fp32 DEST) by its columns' fp16 scales. SFPU geometry of an fp32 DEST tile
(tests/probe4.py): face f is SFPU rows 8f..8f+7; SFPU row 8f+k holds face rows 4*(k/2)..+3 (one per 8-lane subvector) at
the even (k even) or odd (k odd) columns of the face. So for one output tile column and one kb, P's rows need 4 lane
vectors (fc, par): face column half fc (faces 0, 2 or 1, 3) and column parity par, lane (s, i) = scale of column
16 fc + 2 i + par for every face row s. A broadcast-scale tile holds those 4 vectors for 8 output tile columns (combo z,
vector v = 2 fc + par at SFPU row 4 z + v), so the SFPU loads a scale vector with one DEST read."""

import numpy as np

from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import KBLOCK, PAGE_BYTES, TILE

GROUP_BYTES = 2048 + 32 * 4 * 256  # a record group: its fp16 scale tile, then 32 kbs of 4 ternary tiles


def scales_from_records(rec: np.ndarray, ppr: int, K: int, N: int) -> np.ndarray:
    """decode_linear3.pack_records_inline records -> s [N, KB] float16."""
    KB, Nt = K // KBLOCK, N // TILE
    G = -(-KB // 32)
    r = rec.reshape(Nt, ppr * PAGE_BYTES // 4).view(np.uint8)  # bytes of each record
    s = np.zeros((N, KB), dtype=np.float16)
    for g in range(G):
        tile = r[:, g * GROUP_BYTES : g * GROUP_BYTES + 2048].copy().view(np.float16).reshape(Nt, 2, 2, 16, 16)
        t = tile.transpose(0, 1, 3, 2, 4).reshape(Nt, TILE, TILE)  # [Nt, row j (kb 32g + j), col]
        nj = min(32, KB - 32 * g)
        s[:, 32 * g : 32 * g + nj] = t[:, :nj, :].transpose(0, 2, 1).reshape(N, nj)
    return s


def pack_scale_bcast(s_nb: np.ndarray) -> np.ndarray:
    """s [N, KB] float16 -> [Nt / 8 * KB, 1024] uint16: per unit of 8 output tile columns and kb, one fp16 32x32 tile
    (face order) of broadcast scale vectors (module docstring). Nt must be a multiple of 8."""
    N, KB = s_nb.shape
    Nt = N // TILE
    assert Nt % 8 == 0, Nt
    s = s_nb.reshape(Nt // 8, 8, TILE, KB)  # [unit, z, col, kb]
    tiles = np.zeros((Nt // 8, KB, 4, 2, 16, 16), dtype=np.float16)  # [unit, kb, face, row, col pair-split]
    out = np.zeros((Nt // 8, KB, 4, 16, 16), dtype=np.float16)  # [unit, kb, face f, face row, face col]
    for z in range(8):
        for v in range(4):
            fc, par = v >> 1, v & 1
            q = 4 * z + v  # SFPU row
            f, k = q // 8, q % 8
            cols = 16 * fc + 2 * np.arange(8) + par  # the 8 lanes' columns
            vals = s[:, z, cols, :]  # [unit, 8 lanes, kb]
            for sub in range(4):
                row = 4 * (k >> 1) + sub
                out[:, :, f, row, 2 * np.arange(8) + (k & 1)] = vals.transpose(0, 2, 1)
    del tiles
    return np.ascontiguousarray(out.reshape(Nt // 8 * KB, 1024)).view(np.uint16)
