# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dimensions and streamer layout of the Gemma-4-31B resident decode prototype (TP4 on a 1 x 4 Blackhole mesh).

Adapted from changh95/qwen3.8-27b-p300x2 code/qwen36_resident/common.py and model.py (Apache-2.0): block_geometry,
bank_core_cols, split, gate_up_core_cols, streamer_cores, rect_cover and BankLayout keep that design (each streamer
core owns K-complete column slices of every weight, laid out in its DRAM bank so the bank's cores read one stream).
"""
import os

import torch

TILE = 32
BF4_TILE, BF8_TILE = 576, 1088
HIDDEN = 5376
HT = HIDDEN // TILE  # 168 hidden tiles (1 x 32 activation tiles)
INTER = 21504
N_CHIPS = 4
IC = INTER // N_CHIPS  # 5376 MLP columns per chip
IT = IC // TILE  # 168
VOCAB = 262144
VC = VOCAB // N_CHIPS  # 65536 vocab columns per chip
LAYERS = 60
FULL_EVERY = 6  # layer l is full attention when l % 6 == 5
Q_S, Q_F = 8 * 256, 8 * 512  # q columns per chip (sliding, full): the o_proj input width
KV_S, KV_F = 4 * 256 * 2, 512  # k|v columns per chip (sliding k and v; full: one shared K=V head)
VIEWS = 6  # 32x32 bf16 views of one hidden vector on the hub (5.25 real, zero padded)
VEC_BYTES = HIDDEN * 2
SLOT_BYTES = VIEWS * 2048
EPS = 1e-6
PER_BANK = int(os.environ.get("GRES_PER_BANK", "4"))  # streamer cores per DRAM bank
RING_BYTES = int(os.environ.get("GRES_RING_BYTES", "1200000"))  # weight ring per streamer (before rounding)
MAX_PAGE = int(os.environ.get("GRES_MAX_PAGE", "8192"))  # DRAM read packet cap

# weight entries: (name, Kt, bf4 or bf8, N tiles)
ENTRIES = [
    ("qkv_s", HT, "bf4", (Q_S + KV_S) // TILE),  # 128
    ("qkv_f", HT, "bf4", (Q_F + KV_F) // TILE),  # 144 (K=V stored once)
    ("o_s", Q_S // TILE, "bf4", HT),  # 64 x 168
    ("o_f", Q_F // TILE, "bf4", HT),  # 128 x 168
    ("gu", HT, "bf4", 2 * IT),  # gate | up
    ("down", IT, "bf4", HT),
    ("head", HT, "bf8", VC // TILE),  # 2048
]
E_QKV_S, E_QKV_F, E_O_S, E_O_F, E_GU, E_DOWN, E_HEAD = range(7)
TILE_BYTES = {"bf4": BF4_TILE, "bf8": BF8_TILE}


def is_full(l):
    return l % FULL_EVERY == FULL_EVERY - 1


def block_geometry(Kt, tile_bytes, max_block=20_000, max_page=None):
    """(sb, pages, page, block): a block is sb K-tiles of one column (the largest even divisor of Kt within
    max_block bytes), read as `pages` packets of `page` bytes"""
    max_page = max_page or MAX_PAGE
    sb = max(d for d in range(2, Kt + 1, 2) if Kt % d == 0 and d * tile_bytes <= max_block)
    block = sb * tile_bytes
    page = (max_page // tile_bytes) * tile_bytes
    while block % page:
        page -= tile_bytes
    return sb, block // page, page, block


def split(n, parts, first_extra=0):
    """(count, first) of `parts` contiguous pieces of n; the n % parts larger pieces start at part first_extra"""
    q, r = divmod(n, parts)
    out, first = [], 0
    for j in range(parts):
        c = q + ((j - first_extra) % parts < r)
        out.append((c, first))
        first += c
    return out


def bank_core_cols(total_tiles, banks, offset=0, first_extra=0):
    """per bank, per core: the column tiles it streams ([0, total) split bank-major, then over PER_BANK cores)"""
    assert total_tiles % banks == 0, (total_tiles, banks)
    per_bank = total_tiles // banks
    return [
        [
            list(range(offset + b * per_bank + f, offset + b * per_bank + f + c))
            for c, f in split(per_bank, PER_BANK, first_extra)
        ]
        for b in range(banks)
    ]


def gate_up_core_cols(it_total, banks, first_extra=0):
    """core_cols of cat(G, U): each core streams its gate tiles, then its up tiles"""
    g = bank_core_cols(it_total, banks, first_extra=first_extra)
    u = bank_core_cols(it_total, banks, offset=it_total, first_extra=first_extra)
    return [[gj + uj for gj, uj in zip(gb, ub)] for gb, ub in zip(g, u)]


GU_EXTRA = 1  # the gate|up remainder on core 1 of a bank (o / down remainders are on core 0)


def entry_core_cols(e, banks):
    name, Kt, _, N = ENTRIES[e]
    if e == E_GU:
        return gate_up_core_cols(IT, banks, GU_EXTRA)
    return bank_core_cols(N, banks)


class BankLayout:
    """Tile order of one weight in its DRAM shards (one gather, applied to every weight of the entry): bank b's
    shard holds the blocks of its PER_BANK cores interleaved, slot (t * nkb + kb) * PER_BANK + j = block kb of
    column t of the bank's core j. Columns a core lacks are zero tiles that are never read."""

    def __init__(self, Kt, core_cols, sb):
        self.Kt, self.sb = Kt, sb
        nkb = Kt // sb
        self.width = PER_BANK * max(len(c) for cols in core_cols for c in cols)
        self.banks = len(core_cols)
        rows, cols_ = [], []
        for cols in core_cols:
            src_k = torch.full((Kt * self.width,), -1, dtype=torch.long)
            src_c = torch.full((Kt * self.width,), -1, dtype=torch.long)
            for j, cj in enumerate(cols):
                for t, col in enumerate(cj):
                    for kb in range(nkb):
                        slot = (t * nkb + kb) * PER_BANK + j
                        lin = slot * sb + torch.arange(sb)
                        src_k[lin] = kb * sb + torch.arange(sb)
                        src_c[lin] = col
            rows.append(src_k)
            cols_.append(src_c)
        self.src_k, self.src_c = torch.stack(rows), torch.stack(cols_)  # [banks, Kt * width]

    def __call__(self, w):
        """w [K, N] -> [K, banks * width * 32]"""
        K, N = w.shape
        Kt, Nt = K // TILE, N // TILE
        tiles = w.reshape(Kt, TILE, Nt, TILE).permute(0, 2, 1, 3)
        tiles = torch.cat([tiles, torch.zeros(Kt, 1, TILE, TILE, dtype=w.dtype)], dim=1)  # column Nt = zeros
        k = self.src_k.clamp(min=0)
        c = torch.where(self.src_c < 0, torch.full_like(self.src_c, Nt), self.src_c)
        g = tiles[k, c]  # [banks, Kt * width, 32, 32]
        g = g.reshape(self.banks, Kt, self.width, TILE, TILE).permute(1, 3, 0, 2, 4)
        return g.reshape(K, self.banks * self.width * TILE)

    def inverse(self, laid, N):
        """[K, banks * width * 32] -> w [K, N]"""
        K = laid.shape[0]
        Kt, Nt = K // TILE, N // TILE
        g = laid.reshape(Kt, TILE, self.banks, self.width, TILE).permute(2, 0, 3, 1, 4)  # [banks, Kt, width, 32, 32]
        g = g.reshape(self.banks, Kt * self.width, TILE, TILE)
        out = torch.zeros(Kt, Nt, TILE, TILE, dtype=laid.dtype)
        valid = self.src_c >= 0
        out[self.src_k[valid], self.src_c[valid]] = g[valid]
        return out.permute(0, 2, 1, 3).reshape(K, N)


def layout_of(e, banks=8):
    _, Kt, dt, _ = ENTRIES[e]
    sb = block_geometry(Kt, TILE_BYTES[dt])[0]
    return BankLayout(Kt, entry_core_cols(e, banks), sb)


def streamer_cores(device):
    """PER_BANK cores per DRAM bank: each column of the banks' optimal reader cores gets one compact block of two
    columns by PER_BANK / 2 rows per bank, the banks in row order (so a multicast covers them in few rectangles)"""
    import ttnn

    primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    grid = device.compute_with_storage_grid_size()
    assert PER_BANK % 2 == 0
    # (PER_BANK 4: two columns by two rows per bank, as the original; 6 / 8: three / four columns by two rows)
    rows, width = 2, PER_BANK // 2
    groups = [None] * len(primary)
    used = set()
    for x in sorted({c.x for c in primary}):
        banks = sorted((c.y, i) for i, c in enumerate(primary) if c.x == x)
        height = rows * len(banks)
        top = min(max(round(sum(y for y, _ in banks) / len(banks) - height / 2), 0), grid.y - height)
        assert x + width - 1 < grid.x and top >= 0
        for k, (_, i) in enumerate(banks):
            g = [ttnn.CoreCoord(x + dx, top + k * rows + dy) for dy in range(rows) for dx in range(width)]
            assert not any((c.x, c.y) in used for c in g), "streamer blocks overlap"
            used |= {(c.x, c.y) for c in g}
            groups[i] = g
    assert len(used) == PER_BANK * len(primary)
    return groups, used


def seam_split(rows):
    """F9 (M2b): whether rect_cover splits rectangles at Blackhole's worker-column gap for an R-row program.
    Measured on M1L (bench/resident/m2b/ra/f9_*.json): not splitting saves R = 8 0.16-0.17 ms at ctx 128-31K but
    costs R = 1 0.05-0.21 ms, so the default splits at R = 1 only. GRES_RECT_SEAM=1 / 0 forces split / no split."""
    mode = os.environ.get("GRES_RECT_SEAM", "auto")
    return rows == 1 if mode == "auto" else mode == "1"


def rect_cover(mesh, cores, seam=True):
    """rectangles covering exactly `cores` (logical) as multicast args: count, then per rectangle physical
    x0, y0, x1, y1 and its number of cores. seam: also split where the virtual coordinates jump (seam_split)"""
    import ttnn

    left = {(c.x, c.y) for c in cores}
    out = []
    # rectangles may span Blackhole's worker-column gap (virtual x 8, 9 between logical 6 and 7), as tt-metal's own
    # multicasts do (matmul_multicore_reuse_mcast_dram_sharded: bounding-box top-left to bottom-right) and as step 2
    # / M1 did; with seam, M1L's split at the gap (more, smaller multicasts: faster at R = 1, slower at R = 8)
    vx = lambda x: mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x  # noqa: E731
    vy = lambda y: mesh.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y  # noqa: E731
    if not seam:
        vx = vy = lambda v: v  # noqa: E731
    for y, x in sorted((c.y, c.x) for c in cores):
        if (x, y) not in left:
            continue
        x1 = x
        while (x1 + 1, y) in left and vx(x1 + 1) == vx(x1) + 1:
            x1 += 1
        y1 = y
        while all((xx, y1 + 1) in left for xx in range(x, x1 + 1)) and vy(y1 + 1) == vy(y1) + 1:
            y1 += 1
        for yy in range(y, y1 + 1):
            for xx in range(x, x1 + 1):
                left.discard((xx, yy))
        p0 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        p1 = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x1, y1))
        out += [p0.x, p0.y, p1.x, p1.y, (x1 - x + 1) * (y1 - y + 1)]
    return [len(out) // 5] + out


def f32_bits(v):
    import struct

    return struct.unpack("<I", struct.pack("<f", v))[0]
