# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side plan for the fused kernel: DRAM layouts and per-core work.

Decomposition: core (x, y) of the GX x GY grid owns mesh-column band x and depth-level group y.
Everything is column-major with a common pitch H (rows per column, H % 4 == 0), so
  - each band is one contiguous range of every array,
  - neighbour (dR, dC) of an edge is the constant offset dC*H + dR//2 into cell plane dR % 2,
  - output term (dr, dc) is the constant offset dc*H + dr into the group's F array.
A 128-item "chunk" is the unit of data movement; a 4 KB tile holds 8 chunks.

Per (group, band, 128-edge block): 24 static chunks (21 used) in DRAM, one 12 KB read:
  0..9  sign*advCoefs*advCoefs3rd   (weights of P)
  10..19 sign*advCoefs3rd           (weights of Q)
  20    sign*0.5*dvEdge on slanted edges, 0 on vertical edges
Per (level, group, band, block): f and mask chunks, one 1 KB read.
Per (level, band): both cell planes (band columns + 4 halo columns).
The kernel computes, per edge and level (all chunks lane-aligned):
  F = fm*(P + copysign(0.25, f)*Q) + (f - fm)*S20*(g_lowA + g_lowB),  fm = f*mask
and per output cell  out = inv_area * sum of 6 F terms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from models.experimental.tensorocean.tt.formulation import build_tables, group_plan

CH = 128  # items per chunk
GX, GY = 11, 10  # Blackhole P150 core grid as harvested on this chip (11 x 10)


def ru(a, b):
    return (a + b - 1) // b * b


@dataclass
class Plan:
    n: int
    L: int
    H: int
    nbx: int  # bands actually used (<= GX)
    lc: int  # levels per level-group (<= ceil(L/GY))
    nby: int  # level groups actually used
    bands: list  # [(oc0, oc1)]
    groups: list  # formulation.Group
    taps: list  # per group: 10 x (plane, offset)
    lowidx: list  # per group: (tap index A, tap index B) or None
    terms: dict  # part -> 6 x (group, dr, dc)
    cell_len: int  # per band plane buffer length (items), incl. halo/pad
    f_len: int  # per band per group F length (items) = nblk*CH
    nblk: int
    out_len: int  # per band out length (items) = noblk*CH
    noblk: int
    shifts: list  # [(plane, rho)] shifted copies the taps need (rho > 0)
    fshift_groups: list  # groups whose F needs a 1-item shifted copy (out terms with dr == 1)
    arrays: dict = field(default_factory=dict)


def make_plan(n: int, L: int, gx: int = GX, gy: int = GY) -> Plan:
    groups = group_plan(n)
    M = n + 4
    max_r = max(g.rows - 1 + max(dR // 2 for dR, _ in g.taps) for g in groups)
    H = ru(max(M // 2, max_r + 1, max(g.rows for g in groups) + 1), 4)
    nbx = min(gx, n)
    edges = np.linspace(0, n, nbx + 1).round().astype(int)
    bands = [(int(edges[i]), int(edges[i + 1])) for i in range(nbx)]
    lc = math.ceil(L / gy)
    nby = math.ceil(L / lc)
    wmax = max(b - a for a, b in bands)
    nblk = math.ceil((wmax + 1) * H / CH)
    f_len = nblk * CH
    noblk = math.ceil(wmax * H / CH)
    out_len = noblk * CH
    taps, lowidx, shifts = [], [], set()
    for g in groups:
        tl = [(dR % 2, dC * H + dR // 2) for dR, dC in g.taps]
        taps.append(tl)
        for (p, off), (dR, _) in zip(tl, g.taps):
            if (dR // 2) % 4:
                shifts.add((p, (dR // 2) % 4))
        lowidx.append(None if g.low is None else tuple(g.taps.index(tuple(x)) for x in g.low))
    # max item read from a plane: last block end + max tap offset
    max_off = max(off for tl in taps for _, off in tl)
    cell_len = ru(max(f_len + max_off + 4, (wmax + 4) * H), CH)
    plan = Plan(
        n=n,
        L=L,
        H=H,
        nbx=nbx,
        lc=lc,
        nby=nby,
        bands=bands,
        groups=groups,
        taps=taps,
        lowidx=lowidx,
        terms={},
        cell_len=cell_len,
        f_len=f_len,
        nblk=nblk,
        out_len=out_len,
        noblk=noblk,
        shifts=sorted(shifts),
        fshift_groups=[],
    )
    plan.terms = _output_terms(n, groups)
    plan.fshift_groups = sorted({g for part in plan.terms.values() for g, dr, dc in part if dr})
    return plan


def _output_terms(n, groups):
    """Li's 6 accumulation terms per output part as (group, dr, dc), read off the ID-trick tables."""
    t = build_tables(n)
    where = {}
    for gi, g in enumerate(groups):
        for r in range(g.rows):
            for c in range(g.cols):
                where[(g.fam, g.src(r, c, n))] = (gi, r, c)
    half = n // 2
    terms = {}
    for part, rows in (("even", range(0, half)), ("odd", range(half, n))):
        lst = []
        for j in range(6):
            found = set()
            for ro_i, ro in enumerate(rows):
                for co in range(n):
                    e = t.out_edges[ro * n + co, j]
                    gi, r, c = where[(int(t.fam[e]), int(t.src[e]))]
                    found.add((gi, r - ro_i, c - co))
            assert len(found) == 1, (part, j, found)
            lst.append(found.pop())
        terms[part] = lst
    return terms


def host_arrays(plan: Plan, host: dict) -> dict:
    """Build the DRAM arrays (float32 numpy) from Li's input dict (torch tensors)."""
    n, L, H, CHn = plan.n, plan.L, plan.H, CH
    M = n + 4
    cell = host["cell"].numpy()
    out = {}
    # CELL[l][x][p][cell_len]
    CELL = np.zeros((L, plan.nbx, 2, plan.cell_len), np.float32)
    for xi, (oc0, oc1) in enumerate(plan.bands):
        for p in (0, 1):
            pl = cell[:, p::2, :]  # [L, M/2, M]
            c1 = min(oc1 + 4, M)
            blk = np.zeros((L, c1 - oc0, H), np.float32)
            blk[:, :, : pl.shape[1]] = pl[:, :, oc0:c1].transpose(0, 2, 1)
            CELL[:, xi, p, : blk.shape[1] * H] = blk.reshape(L, -1)
    out["CELL"] = CELL
    # per group, per band: f, mk (per level) and statics
    FMK = np.zeros((len(plan.groups), plan.nbx, plan.nblk, L, 2, CHn), np.float32)  # all levels of a block contiguous
    STAT = np.zeros((len(plan.groups), plan.nbx, plan.nblk, 24, CHn), np.float32)
    for gi, g in enumerate(plan.groups):
        sfx = str(g.fam)
        flux = host["normalThicknessFlux" + sfx].numpy().reshape(L, -1)
        mask = host["advMaskHighOrder" + sfx].numpy().reshape(L, -1)
        a = host["advCoefs" + sfx].numpy().reshape(-1, 10)
        b = host["advCoefs3rd" + sfx].numpy().reshape(-1, 10)
        dv = host["dvEdge" + sfx].numpy().reshape(-1)
        es = host["edgeSignOnCell" + sfx].numpy().reshape(-1)
        for xi, (oc0, oc1) in enumerate(plan.bands):
            fcols = min(oc1 + 1, g.cols) - oc0
            if fcols <= 0:
                continue
            rr, cc = np.meshgrid(np.arange(g.rows), np.arange(fcols), indexing="ij")
            item = (cc * H + rr).reshape(-1)  # band-local column-major
            src = np.array([g.src(r, c + oc0, n) for r, c in zip(rr.reshape(-1), cc.reshape(-1))])
            fl = np.zeros((L, plan.f_len), np.float32)
            fl[:, item] = flux[:, src]
            mkk = np.zeros((L, plan.f_len), np.float32)
            mkk[:, item] = mask[:, src]
            FMK[gi, xi, :, :, 0, :] = fl.reshape(L, plan.nblk, CHn).transpose(1, 0, 2)
            FMK[gi, xi, :, :, 1, :] = mkk.reshape(L, plan.nblk, CHn).transpose(1, 0, 2)
            st = np.zeros((24, plan.f_len), np.float32)
            for i in range(10):
                st[i, item] = es[src] * a[src, i] * b[src, i]
                st[10 + i, item] = es[src] * b[src, i]
            if g.fam == 1:
                st[20, item] = es[src] * 0.5 * dv[src]
            STAT[gi, xi] = st.reshape(24, plan.nblk, CHn).transpose(1, 0, 2)
    out["FMK"] = FMK
    out["STAT"] = STAT
    # inverse areas per out item (column-major per band, rows < n/2)
    INV = np.zeros((plan.nbx, plan.out_len), np.float32)
    area = host["areaCell"].numpy()
    for xi, (oc0, oc1) in enumerate(plan.bands):
        v = np.zeros((oc1 - oc0, H), np.float32)
        v[:, : n // 2] = (1.0 / area[oc0:oc1])[:, None]
        INV[xi, : v.size] = v.reshape(-1)
    out["INV"] = INV
    return out


def unpack_output(plan: Plan, OUT: np.ndarray):
    """OUT[l][part][x][out_len] -> (even, odd) torch [L, n/2, n]."""
    n, H, L = plan.n, plan.H, plan.L
    res = []
    for part in (0, 1):
        full = np.zeros((L, n // 2, n), np.float32)
        for xi, (oc0, oc1) in enumerate(plan.bands):
            v = OUT[:, part, xi, : (oc1 - oc0) * H].reshape(L, oc1 - oc0, H)
            full[:, :, oc0:oc1] = v[:, :, : n // 2].transpose(0, 2, 1)
        res.append(torch.from_numpy(full))
    return res
