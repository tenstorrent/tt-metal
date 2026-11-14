# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Block scales for the 2D exact prefill linear (linear2.py), straight from the decode records.

The SFPU multiplies a product tile P (fp32 DEST) by its columns' fp16 scales. SFPU geometry of an fp32 DEST tile
(tests/probe4.py): face f is SFPU rows 8f..8f+7; SFPU row 8f+k holds face rows 4*(k/2)..+3 (one per 8-lane subvector) at
the even (k even) or odd (k odd) columns of the face. So for one output tile column and one kb, P's rows need 4 lane
vectors (fc, par): face column half fc (faces 0, 2 or 1, 3) and column parity par, lane (s, i) = scale of column
16 fc + 2 i + par for every face row s, so the SFPU loads a vector with one DEST read. linear2.py chains J kbs per DEST
pass and multiplies by scale ratios (ratio_chain); one tile per column and kb group holds them (pack_ratio_group)."""

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


def ratio_chain(s_nb: np.ndarray, J: int) -> np.ndarray:
    """s [N, KB] -> r [N, KB] float64 for a chain of J kbs per DEST pass (kb groups J g .. J g + J - 1): after kb j's
    product P_j the SFPU multiplies the running sum by r_j = s_j / s_(j+1) (s_j for a group's last kb), so the next
    product adds onto (sum_(i <= j) s_i P_i) / s_(j+1) and the group ends as sum_j s_j P_j. Every scale must be
    nonzero (Bonsai's are all positive)."""
    N, KB = s_nb.shape
    assert KB % J == 0, (KB, J)
    s = s_nb.astype(np.float64)
    assert (s != 0).all()
    r = s.copy()
    last = (np.arange(KB) % J) == J - 1
    r[:, ~last] = s[:, ~last] / s[:, np.roll(~last, 1)]
    return r


def pack_ratio_group(r_nb: np.ndarray, J: int) -> np.ndarray:
    """r [N, KB] (float64) -> [Nt * KB / J, 1024] uint16: per output tile column n and kb group g (J <= 4 kbs), one fp16
    32x32 tile (face order) whose SFPU rows 8 j + 4 h + v hold kb J g + j's lane vectors, so one copy into DEST serves the
    column's whole group. Vector v = 2 fc + par (module docstring: face column half fc, column parity par) has lane
    (s, i) = the ratio of column 16 fc + 2 i + par for every face row s; h = 0: fp16(r), h = 1: fp16(r - fp16(r)), so the
    SFPU's fp32 sum of the two is r to ~2^-22. Tile index n * KB / J + g."""
    N, KB = r_nb.shape
    Nt, G = N // TILE, KB // J
    assert J <= 4 and KB % J == 0, (J, KB)
    hi = r_nb.astype(np.float16)
    lo = (r_nb - hi.astype(np.float64)).astype(np.float16)
    parts = [hi.reshape(Nt, TILE, G, J), lo.reshape(Nt, TILE, G, J)]  # [n, col, g, j]
    out = np.zeros((Nt, G, 4, 16, 16), dtype=np.float16)  # [n, g, face f, face row, face col]
    for j in range(J):
        for h in range(2):
            for v in range(4):
                fc, par = v >> 1, v & 1
                q = 8 * j + 4 * h + v  # SFPU row
                f, k = q // 8, q % 8
                cols = 16 * fc + 2 * np.arange(8) + par
                vals = parts[h][:, :, :, j][:, cols, :]  # [n, 8 lanes, g] (one index array: no axis move)
                for sub in range(4):
                    out[:, :, f, 4 * (k >> 1) + sub, 2 * np.arange(8) + (k & 1)] = vals.transpose(0, 2, 1)
    return np.ascontiguousarray(out.reshape(Nt * G, 1024)).view(np.uint16)
