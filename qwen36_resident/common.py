# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model dimensions and the streamer layout helpers shared by the resident decode, the prefill and the benches."""
import os
import struct

import torch

import ttnn

TILE = 32
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
HIDDEN = 5120
INTER = 17408  # full MLP width; each chip holds INTER / num_chips
EPS = 1e-6
PER_BANK = int(os.environ.get("RESIDENT_PER_BANK", "4"))  # streamer cores per DRAM bank
L1_WEIGHT_BUDGET = int(os.environ.get("RESIDENT_RING_BYTES", "1000000"))  # weight ring bytes per streamer


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


MAX_PAGE = int(os.environ.get("RESIDENT_MAX_PAGE", "8192"))  # DRAM read packet size cap


def block_geometry(Kt, dtype, max_block=20_000, max_page=None):
    max_page = max_page or MAX_PAGE
    tb = TILE_BYTES[dtype]
    sb = max(d for d in range(2, Kt + 1, 2) if Kt % d == 0 and d * tb <= max_block)
    block = sb * tb
    page = (max_page // tb) * tb
    while block % page:
        page -= tb
    return sb, block // page, page, block


def bank_core_cols(total_tiles, banks, offset=0, first_extra=0):
    """core_cols for column tiles [0, total_tiles) split bank-major, then contiguously over PER_BANK cores."""
    per_bank = total_tiles // banks
    return [
        [
            list(range(offset + b * per_bank + f, offset + b * per_bank + f + c))
            for c, f in split(per_bank, PER_BANK, first_extra)
        ]
        for b in range(banks)
    ]


def gate_up_core_cols(it_total, banks, first_extra=0):
    """core_cols of cat(G, U): each core streams its gate tiles, then its up tiles."""
    g = bank_core_cols(it_total, banks, first_extra=first_extra)
    u = bank_core_cols(it_total, banks, offset=it_total, first_extra=first_extra)
    return [[gj + uj for gj, uj in zip(gb, ub)] for gb, ub in zip(g, u)]


def split(n, parts, first_extra=0):
    """(count, first) of `parts` contiguous pieces of n; the n % parts larger pieces start at part first_extra
    (cyclically), so entries with a remainder can put it on different cores."""
    q, r = divmod(n, parts)
    out, first = [], 0
    for j in range(parts):
        c = q + ((j - first_extra) % parts < r)
        out.append((c, first))
        first += c
    return out


def streamer_cores(device):
    """PER_BANK cores per DRAM bank. The banks' adjacent cores sit in two columns; each column's banks get
    one compact block of two columns (the bank column and the next) by PER_BANK / 2 rows, the banks in row
    order, so the hub's slot multicast covers all streamers with one rectangle per block."""
    primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    grid = device.compute_with_storage_grid_size()
    assert PER_BANK % 2 == 0
    rows = PER_BANK // 2
    groups = [None] * len(primary)
    used = set()
    for x in sorted({c.x for c in primary}):
        banks = sorted((c.y, i) for i, c in enumerate(primary) if c.x == x)
        height = rows * len(banks)
        top = min(max(round(sum(y for y, _ in banks) / len(banks) - height / 2), 0), grid.y - height)
        assert x + 1 < grid.x and top >= 0
        for k, (_, i) in enumerate(banks):
            g = [ttnn.CoreCoord(x + dx, top + k * rows + dy) for dy in range(rows) for dx in range(2)]
            used |= {(c.x, c.y) for c in g}
            groups[i] = g
    assert len(used) == PER_BANK * len(primary)
    return groups, used


def rect_cover(mesh, cores):
    """rectangles covering exactly `cores` (logical), as hub multicast args: count, then per rectangle
    noc x0, y0, x1, y1 (physical) and its number of cores"""
    left = {(c.x, c.y) for c in cores}
    out = []
    for y, x in sorted((c.y, c.x) for c in cores):
        if (x, y) not in left:
            continue
        x1 = x
        while (x1 + 1, y) in left:
            x1 += 1
        y1 = y
        while all((xx, y1 + 1) in left for xx in range(x, x1 + 1)):
            y1 += 1
        for yy in range(y, y1 + 1):
            for xx in range(x, x1 + 1):
                left.discard((xx, yy))
        p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x1, y1))
        out += [p0.x, p0.y, p1.x, p1.y, (x1 - x + 1) * (y1 - y + 1)]
    return [len(out) // 5] + out


def quantize(w, dtype):
    return ttnn.to_torch(ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()
