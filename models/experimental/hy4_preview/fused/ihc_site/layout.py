# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host <-> device layouts of the fused iHC site (fused/ihc_site/op.py, kernels/ihc_site.cpp).

Every tensor of the site is a plain [1, 1, 32, 32 * pages] TILE tensor used as a container of 4 KB (fp32) or 2 KB
(BF16) pages; a page is 4 faces of 16 x 16 values in memory order. The kernels read a page three ways:

  * as one 32 x 32 tile, unpacked straight into DEST for the SFPU (fp32, bit exact);
  * as two [16, 32] tiles (faces 0 | 1, faces 2 | 3): the streams as the left operand of the fn projection;
  * as two [32, 16] tiles (faces 0 / 1, faces 2 / 3): fn^T as its right operand.

Token rows. The site carries ROWS = 8 token rows (users); fewer are zero padded. A face holds one [8 rows, 32
columns] "row tile" as two halves: face row 8 h + r, column c is token row r, column 16 h + c.

  streams   [T, n = 4, C]   page t (C / 32 of them), face j: stream j, columns [32 t, 32 t + 32) as a row tile.
                            786 KB for C = 6144, fp32.
  rows      [T, C]          page p (C / 128), face k: columns [32 (4 p + k), ... + 32) as a row tile. The collapsed
                            sublayer input y and the sublayer output F. The bytes are those of a [8, C] tensor of
                            [8, 32] tiles (2 faces of [8, 16]) laid back to back.
  fn^T      [2 n, n C]      page t, face j, row c, column col(h, o) = fn[o, j C + 32 t + 16 h + c]: stream j's 32
                            columns of chunk t against the 2 n outputs, the two column halves in disjoint columns.
                            With the streams face as left operand one face-by-face matmul leaves the output o of
                            token row r in DEST rows r (h = 0) and 8 + r (h = 1) at columns col(h, o); col(h, o) is the
                            column lane o of SFPU lane set h reads, so the two are added lane by lane.

SFPU addressing (measured by tests/test_ihc_site.py::test_probe_lanes, see SFPU_LANES). One SFPU register holds 4
DEST rows x 8 lanes. "Unit" u of a page is DEST address 2 u relative to the page: face u // 8, rows
4 * ((u % 8) // 2) .. + 3, and the column set u % 2. A "cell" below is one unit: [4 rows, 8 lanes].
"""

from __future__ import annotations

import torch

ROWS = 8  # token rows of one pass
TILE = 32
FACE = 16
PAGE_VALUES = TILE * TILE
UNITS_PER_PAGE = 32

# Which 8 of a face's 16 columns one SFPU load of column set p takes: "half" = columns 8 p + lane,
# "evenodd" = columns 2 lane + p. Set from the probe (test_probe_lanes); only the host builders of the constant page
# and of fn^T depend on it, the kernels do not.
SFPU_LANES = "evenodd"

# Units of the constant page (kernels/ihc_sfpu.hpp CONST_*).
CONST_SCALE, CONST_BASE, CONST_GAIN, CONST_EPS, CONST_NORM_EPS, CONST_INV_N, CONST_INV_C = range(7)
CONST_LANE = 8  # .. 15: lane o is one, the others zero


def _pad_rows(t: torch.Tensor) -> torch.Tensor:
    assert t.shape[0] <= ROWS, f"{t.shape[0]} token rows, the site carries {ROWS}"
    if t.shape[0] == ROWS:
        return t
    pad = torch.zeros(ROWS - t.shape[0], *t.shape[1:], dtype=t.dtype)
    return torch.cat([t, pad], dim=0)


def streams_to_pages(x: torch.Tensor) -> torch.Tensor:
    """[T, 4, C] -> the [32, C] matrix whose TILE-layout bytes are the streams pages."""
    T, n, C = x.shape
    assert n == 4 and C % TILE == 0, (n, C)
    v = _pad_rows(x.float()).reshape(ROWS, 2, 2, C // TILE, 2, FACE)  # r, ja, jb, t, h, c
    return v.permute(1, 4, 0, 3, 2, 5).reshape(TILE, C).contiguous()  # (ja, h, r), (t, jb, c)


def pages_to_streams(m: torch.Tensor, T: int = ROWS) -> torch.Tensor:
    """Inverse of streams_to_pages: [32, C] -> [T, 4, C]."""
    C = m.shape[-1]
    v = m.reshape(2, 2, ROWS, C // TILE, 2, FACE)  # ja, h, r, t, jb, c
    return v.permute(2, 0, 4, 3, 1, 5).reshape(ROWS, 4, C)[:T].contiguous()


def rows_to_pages(v: torch.Tensor) -> torch.Tensor:
    """[T, C] -> the [32, C / 4] matrix whose TILE-layout bytes are the rows pages."""
    T, C = v.shape
    assert C % (4 * TILE) == 0, C
    w = _pad_rows(v.float()).reshape(ROWS, C // (4 * TILE), 2, 2, 2, FACE)  # r, p, ka, kb, h, c
    return w.permute(2, 4, 0, 1, 3, 5).reshape(TILE, C // 4).contiguous()  # (ka, h, r), (p, kb, c)


def pages_to_rows(m: torch.Tensor, T: int = ROWS) -> torch.Tensor:
    """Inverse of rows_to_pages: [32, C / 4] -> [T, C]."""
    P = m.shape[-1] // TILE
    w = m.reshape(2, 2, ROWS, P, 2, FACE)  # ka, h, r, p, kb, c
    return w.permute(2, 3, 0, 4, 1, 5).reshape(ROWS, P * 4 * TILE)[:T].contiguous()


def fn_to_pages(fn: torch.Tensor, n: int = 4, lanes: str = None, mm32: bool = False) -> torch.Tensor:
    """fn [2 n = 8, n C] -> the [32, C] matrix whose TILE-layout bytes are the fn^T pages.

    Output o of column half h is the face column that lane o of SFPU lane set h reads (8 h + o or 2 o + h), so the
    two halves of the projection are lane aligned in DEST whichever way the lanes run.

    mm32 (the fallback projection, one full [32, 32] x [32, 32] matmul per page): stream j's face sits at face
    row j % 2, face column j // 2 instead of the other way round, so that the product's faces 0 and 3 are
    streams 0, 1 and streams 2, 3 against their own fn."""
    lanes = lanes or SFPU_LANES
    outs, nC = fn.shape
    C = nC // n
    assert outs == 8 and n == 4 and C % TILE == 0, (outs, n, C)
    v = fn.float().reshape(outs, 2, 2, C // TILE, 2, FACE)  # o, ja, jb, t, h, c
    row, col = (2, 1) if mm32 else (1, 2)  # which half of the stream index picks the face row / the face column
    if lanes == "half":
        v = v.permute(row, 5, 3, col, 4, 0)  # (face row, c), (t, face column, h, o)
    else:
        v = v.permute(row, 5, 3, col, 0, 4)  # (face row, c), (t, face column, o, h)
    return v.reshape(TILE, C).contiguous()


def fn_to_row_pages(fn: torch.Tensor, n: int = 4, lanes: str = None) -> torch.Tensor:
    """fn [2 n = 8, n C] -> the [32, 4 C] matrix whose TILE-layout bytes are the fn pages of the SFPU projection
    (IhcSite(sfpu_proj=True)): fp32, four pages per streams page.

    Page 4 t + j belongs to streams page t, stream j. Its unit 8 (2 h + p) + o is fn[o] at the 8 columns that
    lane set p of column half h holds in the streams page, the same on all 4 rows: the SFPU multiplies a streams
    cell (4 token rows x 8 columns) by it lane by lane and sums per output."""
    lanes = lanes or SFPU_LANES
    outs, nC = fn.shape
    C = nC // n
    assert outs == 8 and n == 4 and C % TILE == 0, (outs, n, C)
    P = C // TILE
    v = fn.float().reshape(outs, n, P, 2, FACE)  # o, j, t, h, c
    out = torch.zeros(TILE, P, n, TILE)  # row, t, j, column
    lane = torch.arange(8)
    for h in range(2):
        for p in range(2):
            src = 8 * p + lane if lanes == "half" else 2 * lane + p  # the columns of lane set p
            for o in range(outs):
                unit = 8 * (2 * h + p) + o
                rows, cols = unit_slices(unit, lanes)
                block = v[o, :, :, h][:, :, src]  # j, t, lane
                out[rows][:, :, :, cols] = block.permute(1, 0, 2).unsqueeze(0).expand(4, -1, -1, -1)
    return out.reshape(TILE, P * n * TILE).contiguous()


def unit_slices(unit: int, lanes: str = None):
    """(row slice, column index tensor) of a unit inside a [32, 32] page matrix: [4 rows, 8 lanes]."""
    lanes = lanes or SFPU_LANES
    face, rg, p = unit // 8, (unit % 8) // 2, unit % 2
    r0 = FACE * (face // 2) + 4 * rg
    c0 = FACE * (face % 2)
    lane = torch.arange(8)
    cols = c0 + (8 * p + lane if lanes == "half" else 2 * lane + p)
    return slice(r0, r0 + 4), cols


def unit_get(page: torch.Tensor, unit: int, lanes: str = None) -> torch.Tensor:
    rows, cols = unit_slices(unit, lanes)
    return page[rows][:, cols]


def unit_set(page: torch.Tensor, unit: int, value: torch.Tensor, lanes: str = None) -> None:
    rows, cols = unit_slices(unit, lanes)
    page[rows, cols] = value.float().expand(4, 8)


def const_page(base: torch.Tensor, scale: torch.Tensor, cfg, lanes: str = None) -> torch.Tensor:
    """The [32, 32] page of one site's constants as SFPU lane vectors (lane o = output o of fn: pre 0..3, post 4..7).

    base [2 n], scale [2] = (a_pre, a_post). cfg: reference.ihc.IHCConfig."""
    n = cfg.n
    assert n == 4 and tuple(base.shape) == (2 * n,) and tuple(scale.shape) == (2,)
    page = torch.zeros(TILE, TILE)
    ones = torch.ones(n)
    scale = scale.float()
    unit_set(page, CONST_SCALE, torch.cat([ones * scale[0], ones * scale[1]]), lanes)
    unit_set(page, CONST_BASE, base.float(), lanes)
    unit_set(page, CONST_GAIN, torch.cat([ones, ones * float(cfg.magnitude)]), lanes)
    unit_set(page, CONST_EPS, torch.full((8,), float(cfg.eps)), lanes)
    unit_set(page, CONST_NORM_EPS, torch.full((8,), float(cfg.norm_eps)), lanes)
    unit_set(page, CONST_INV_N, torch.full((8,), 1.0 / (n * cfg.dim)), lanes)
    unit_set(page, CONST_INV_C, torch.full((8,), 1.0 / cfg.dim), lanes)
    for o in range(8):
        unit_set(page, CONST_LANE + o, torch.eye(8)[o], lanes)
    return page


def coef_from_page(page: torch.Tensor, T: int = ROWS, lanes: str = None):
    """The coefficient state page -> (pre [T, 4], post [T, 4]) as the kernel broadcast them (lane 0 of each cell)."""
    g = torch.zeros(ROWS, 8)
    for o in range(8):
        for grp in range(2):
            g[4 * grp : 4 * grp + 4, o] = unit_get(page, 2 * o + grp, lanes)[:, 0]
    return g[:T, :4].contiguous(), g[:T, 4:].contiguous()


def coef_to_page(pre: torch.Tensor, post: torch.Tensor, lanes: str = None) -> torch.Tensor:
    """(pre [T, 4], post [T, 4]) -> the coefficient state page."""
    g = _pad_rows(torch.cat([pre.float(), post.float()], dim=-1))
    page = torch.zeros(TILE, TILE)
    for o in range(8):
        for grp in range(2):
            rows, cols = unit_slices(2 * o + grp, lanes)
            page[rows, cols] = g[4 * grp : 4 * grp + 4, o : o + 1].expand(4, 8)
    return page


def stat_from_page(page: torch.Tensor, T: int = ROWS) -> torch.Tensor:
    """The statistic page -> 1 / RMS of the collapsed sublayer input per token row [T]: face 0, row r, column 0."""
    return page[:ROWS, 0][:T].contiguous()


def _cells(page: torch.Tensor, unit0: int, step: int, lanes: str) -> torch.Tensor:
    """Two cells (token rows 0..3 at unit0, 4..7 at unit0 + step) as [8 token rows, 8 lanes]."""
    return torch.cat([unit_get(page, unit0, lanes), unit_get(page, unit0 + step, lanes)], dim=0)


def debug_from_page(page: torch.Tensor, lanes: str = None) -> dict:
    """A core's scratch page (kernels/ihc_sfpu.hpp U_SCR) -> its intermediates, each [8 token rows, 8 lanes]:
    proj      this core's projection, lane o = output o: DEST [r][.] + [8 + r][.] of the projection face
    part_mix  the same as the core sent it, part_ss the lane sums of squares of the streams as it sent them
    ss, y2    the running lane sums of squares of the streams / of y
    mix       the normalised mixes (all cores), g = pre | post (all cores)."""
    cell = lambda g, h, p: 2 * (2 * h + g) + p
    proj = torch.cat(
        [unit_get(page, cell(g, 0, 0), lanes) + unit_get(page, cell(g, 1, 1), lanes) for g in range(2)], dim=0
    )
    return {
        "proj": proj,
        "part_mix": _cells(page, 8 + 0, 2, lanes),
        "part_ss": _cells(page, 8 + 1, 2, lanes),
        "ss": _cells(page, 16 + 0, 1, lanes),
        "y2": _cells(page, 16 + 2, 1, lanes),
        "mix": _cells(page, 24 + 0, 1, lanes),
        "g": _cells(page, 24 + 2, 1, lanes),
    }
