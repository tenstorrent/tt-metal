# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Plain expert tensors from a PF-layout cache, on the device: what the chunked prefill reads in a decode process.

A PF-layout cache (convert/pf_layout.py) holds each expert tensor [1, E, K, N] as the same tiles, byte for byte, in
two tensors: main [1, E, K, 70 * 32] (ND-sharded, its tile columns in ring order) and an extra tensor with the two
columns left over. The chunked prefill's expert op reads a plain interleaved [1, E, K, N] tensor. These functions
rebuild that tensor from main + extra with whole-tile moves only (slice, concat and transpose of tile-aligned
pieces, page views): no value is requantized, so the result is the TILE cache's tensor.

    gate_up(main, extra) -> [1, E, K, N]     down(main, extra) -> [1, E, K, N]     (the caller deallocates them)
"""

from __future__ import annotations

import os

import ttnn
from models.demos.inkling2_mini.convert.stick import N_BANKS

X_LO = 2  # first bank of gate_up's extra pair (convert/bank_layout.extra_bank)
# How tile-column pieces are joined. "pages": as [E * Kt, columns, 32, 32] page views, concat over dim 1 (whole
# pages by construction). "pairs": width concats of two tensors at a time. "width": one width concat of all (on
# a p100a this one returned other tiles for six inputs; two inputs were exact).
JOIN = os.environ.get("INKLING_PF_TILES_JOIN") or "pages"


def _view(t, shape):
    """The same pages under another shape (page order kept: a reinterpretation, not a value reshape)."""
    return ttnn.experimental.view(t, shape)


def _slice(t, begins, ends):
    """A copy of whole tiles, interleaved in DRAM whatever the placement of t."""
    return ttnn.slice(t, begins, ends, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _cols(t, a, b):
    """Tile columns [a, b) of a [1, E, K, N] tensor."""
    return _slice(t, [0, 0, 0, a * 32], [1, t.shape[1], t.shape[2], b * 32])


def _join(parts):
    """[1, E, K, w_i * 32] tensors -> [1, E, K, sum w_i * 32]; frees the parts."""
    dram = ttnn.DRAM_MEMORY_CONFIG
    if JOIN == "width" or len(parts) == 1:
        out = ttnn.concat(parts, dim=3, memory_config=dram) if len(parts) > 1 else parts[0]
    elif JOIN == "pairs":
        while len(parts) > 1:
            nxt = []
            for i in range(0, len(parts) - 1, 2):
                nxt.append(ttnn.concat([parts[i], parts[i + 1]], dim=3, memory_config=dram))
                ttnn.deallocate(parts[i])
                ttnn.deallocate(parts[i + 1])
            parts = nxt + parts[len(parts) - len(parts) % 2 :]
        return parts[0]
    else:
        _, E, K, _ = parts[0].shape
        pages = [_view(p, [E * (K // 32), p.shape[3] // 32, 32, 32]) for p in parts]  # pages ((e, k), column)
        out = ttnn.concat(pages, dim=1, memory_config=dram)
        out = _view(out, [1, E, K, out.shape[1] * 32])
    for t in parts:
        if t is not out:
            ttnn.deallocate(t)
    return out


def down(main, extra, nb=N_BANKS):
    """main: columns 0 .. 69 in order. extra [1, 1, P*32, 32]: page nb * (e * Kt + i) + b is tile i of expert e's
    column 70 + b for b < 2 (zero pages for b >= 2)."""
    _, E, K, n_main = main.shape
    kt = K // 32
    x = _view(extra, [E * kt, nb, 32, 32])
    x = _slice(x, [0, 0, 0, 0], [E * kt, 2, 32, 32])  # pages ((e, i), column)
    x = _view(x, [1, E, K, 64])
    m = _cols(main, 0, n_main // 32)  # concat takes no ND-sharded input beside an interleaved one
    return _join([m, x])


def _gate_up_extra(extra, E, kt, nb):
    """extra [1, 1, P*32, 32]: page nb * slot + bank; expert e = 5 g + b keeps its extra pair in bank X_LO + b at
    slots g * 2 Kt + i (gate column tile i) and g * 2 Kt + Kt + i (up column tile i). Returns the two columns
    [1, E, K, 32]."""
    per = nb - X_LO
    groups = -(-E // per)
    slots = groups * 2 * kt
    x = _view(extra, [slots, nb, 32, 32])
    x = _slice(x, [0, X_LO, 0, 0], [slots, nb, 32, 32])  # pages ((g, t, i), b)
    y = ttnn.transpose(x, 0, 1)  # pages (b, (g, t, i))
    ttnn.deallocate(x)
    y = _view(y, [per, groups, 2 * kt * 32, 32])  # pages (b, g, (t, i))
    z = ttnn.transpose(y, 0, 1)  # pages (g, b, (t, i))
    ttnn.deallocate(y)
    z = _view(z, [groups * per, 2, kt * 32, 32])  # pages (e, t, i)
    cols = [_view(_slice(z, [0, t, 0, 0], [E, t + 1, kt * 32, 32]), [1, E, kt * 32, 32]) for t in range(2)]
    ttnn.deallocate(z)
    return cols


def gate_up(main, extra, nb=N_BANKS):
    """main: tile column m is the gate (m // nb even) or up column of pair m % nb + nb * (m // (2 nb)). Returns
    the plain tensor: the gate columns 0 .. P-1, then the up columns 0 .. P-1."""
    _, E, K, n_main = main.shape
    kt, blocks = K // 32, n_main // 32 // nb
    gx, ux = _gate_up_extra(extra, E, kt, nb)
    cols = [_cols(main, nb * i, nb * (i + 1)) for i in range(blocks)]  # even blocks of nb columns: gate, odd: up
    return _join(cols[0::2] + [gx] + cols[1::2] + [ux])
