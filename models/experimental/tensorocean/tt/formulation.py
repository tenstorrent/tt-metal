# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Li's optimized kernel rewritten as an explicit edge list, with index tables extracted
mechanically from Li's own selection code (the "ID trick"): run Li's slicing / unfold /
mask code on tensors that hold element IDs instead of values, and read off which element
lands where.  No index is derived by hand.

The kernel then reads, for every depth level l:

  fm[l,e]   = flux[l,e] * mask[l,e]                          (flux = normalThicknessFlux)
  w[l,e,i]  = fm[l,e] * (advCoefs[e,i] + 0.25*sign(flux[l,e])) * advCoefs3rd[e,i]
  low[l,e]  = 0.5*dvEdge[e] * (1 - mask[l,e]) * flux[l,e]    (slanted edges only; 0 otherwise)
  F[l,e]    = edgeSign[e] * ( sum_i w[l,e,i]*cell[l,nbr[e,i]] + low[l,e]*(cell[l,c1[e]] + cell[l,c2[e]]) )
  out[l,o]  = ( sum_j F[l, edge_of_out[o,j]] ) / area[col(o)]          j = 1..6

`e` runs over one unified list holding the slanted ("family 1") and vertical ("family 2")
edges that the outputs use; `o` runs over the N*N output cells (even rows then odd rows).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Tables:
    n: int
    E: int  # number of edges in the unified list
    fam: np.ndarray  # [E] 1 or 2
    src: np.ndarray  # [E] flat index into that family's (rows x cols) edge grid
    nbr: np.ndarray  # [E,10] flat index into the (N+4)^2 cell grid
    low: np.ndarray  # [E,2] cell indices for the 2nd-order average (-1: none)
    out_edges: np.ndarray  # [N*N, 6] index into the unified edge list
    out_col: np.ndarray  # [N*N] column -> areaCell index

    @property
    def shape1(self):
        return (self.n + 1, 2 * self.n + 1)

    @property
    def shape2(self):
        return (self.n, self.n + 1)


def _ids(*shape):
    return torch.arange(int(np.prod(shape)), dtype=torch.float64).reshape(shape)


def build_tables(n: int) -> Tables:
    m = n + 4
    cell = _ids(1, m, m)
    e1 = _ids(1, n + 1, 2 * n + 1)
    e2 = _ids(1, n, n + 1)

    # --- Li's neighbour selection (horizontal_flux_torch), applied to IDs ---
    masks = {
        "even2": [False, True, True, True, True, True, True, True, False, True, True, True],
        "odd2": [True, True, True, False, True, True, True, True, True, True, True, False],
        "ee": [False, True, True, False, True, True, True, False, False, True, True, True, False, True, True, False],
        "eo": [False, False, True, True, False, True, True, True, False, True, True, True, False, True, True, False],
        "oe": [False, True, True, False, False, True, True, True, True, True, True, False, False, True, True, False],
        "oo": [False, True, True, False, False, True, True, True, False, True, True, True, False, False, True, True],
    }
    masks = {k: torch.tensor(v) for k, v in masks.items()}
    a = cell[:, 1:-1, :].unfold(2, 4, 1)
    b = torch.cat((a[:, :-2, :], a[:, 1:-1, :], a[:, 2:, :]), 3)
    nbrs = {"even2": b[:, ::2, :, masks["even2"]], "odd2": b[:, 1::2, :, masks["odd2"]]}
    a = cell.unfold(2, 4, 1)
    b = torch.cat((a[:, :-3, :], a[:, 1:-2, :], a[:, 2:-1, :], a[:, 3:, :]), 3)
    er, orw = b[:, ::2], b[:, 1::2]
    nbrs.update(
        ee=er[:, :, :, masks["ee"]],
        eo=er[:, :, :-1, masks["eo"]],
        oe=orw[:, :, :, masks["oe"]],
        oo=orw[:, :, :-1, masks["oo"]],
    )
    # Which edge (weights, flux, sign, mask) each group position uses: Li's tracer_wgt slices.
    srcs = {
        "even2": (2, e2[:, ::2]),
        "odd2": (2, e2[:, 1::2]),
        "ee": (1, e1[:, ::2, ::2]),
        "eo": (1, e1[:, ::2, 1::2]),
        "oe": (1, e1[:, 1::2, ::2]),
        "oo": (1, e1[:, 1::2, 1::2]),
    }
    # Li's 2nd-order cell pairs (only on family-1 groups).
    ev = cell[:, 1:-1:2, 1:-1]
    od = cell[:, 2:-1:2, 1:-1]
    shp = {g: nbrs[g].shape[1:3] for g in nbrs}
    lows = {
        "ee": (ev[:, : shp["ee"][0], :-1], od[:, : shp["ee"][0], 1:]),
        "eo": (ev[:, : shp["eo"][0], 1:-1], od[:, : shp["eo"][0], 1:-1]),
        "oe": (ev[:, 1 : 1 + shp["oe"][0], :-1], od[:, : shp["oe"][0], 1:]),
        "oo": (ev[:, 1 : 1 + shp["oo"][0], 1:-1], od[:, : shp["oo"][0], 1:-1]),
    }

    # --- one global position id per (group, r, c): Li's accumulation applied to IDs ---
    order = ["ee", "eo", "oe", "oo", "even2", "odd2"]
    base, gid = 0, {}
    for g in order:
        r, c = shp[g]
        fam, s = srcs[g]
        assert tuple(s.shape[1:3]) == (r, c), (g, s.shape, r, c)
        gid[g] = base + _ids(1, r, c)
        base += r * c
    G = gid
    even_terms = [
        G["ee"][:, : shp["even2"][0], : shp["eo"][1]],
        G["eo"][:, : shp["even2"][0]],
        G["oe"][:, :, : shp["eo"][1]],
        G["oo"],
        G["even2"][:, :, :-1],
        G["even2"][:, :, 1:],
    ]
    odd_terms = [
        G["ee"][:, 1:, : shp["eo"][1]],
        G["eo"][:, 1:],
        G["oe"][:, :, : shp["eo"][1]],
        G["oo"],
        G["odd2"][:, :, :-1],
        G["odd2"][:, :, 1:],
    ]
    half = n // 2
    for t in even_terms + odd_terms:
        assert tuple(t.shape[1:]) == (half, n), t.shape
    out_pos = torch.stack([torch.cat([te[0], to[0]], 0) for te, to in zip(even_terms, odd_terms)], -1)
    out_pos = out_pos.reshape(n * n, 6).long().numpy()  # global position ids

    # Per global position: family, edge source, neighbours, low pair.
    P = base
    fam_p = np.zeros(P, np.int64)
    src_p = np.zeros(P, np.int64)
    nbr_p = np.zeros((P, 10), np.int64)
    low_p = -np.ones((P, 2), np.int64)
    for g in order:
        ids = gid[g][0].reshape(-1).long().numpy()
        fam, s = srcs[g]
        fam_p[ids] = fam
        src_p[ids] = s[0].reshape(-1).long().numpy()
        nbr_p[ids] = nbrs[g][0].reshape(-1, 10).long().numpy()
        if g in lows:
            low_p[ids, 0] = lows[g][0][0].reshape(-1).long().numpy()
            low_p[ids, 1] = lows[g][1][0].reshape(-1).long().numpy()

    # Keep only positions some output uses; renumber.
    used = np.unique(out_pos)
    remap = -np.ones(P, np.int64)
    remap[used] = np.arange(len(used))
    out_col = np.tile(np.arange(n), n)  # areaCell index = output column
    return Tables(
        n=n,
        E=len(used),
        fam=fam_p[used],
        src=src_p[used],
        nbr=nbr_p[used],
        low=low_p[used],
        out_edges=remap[out_pos],
        out_col=out_col,
    )


def reference_edge_list(host: dict, t: Tables) -> tuple[torch.Tensor, torch.Tensor]:
    """The edge-list formula in torch (float64), for checking the tables against Li's code."""
    n = t.n
    L = host["cell"].shape[0]
    cell = host["cell"].double().reshape(L, -1)

    def per_edge(name1, name2, levels=True):
        a1 = host[name1].double().reshape((L if levels else 1), -1, *host[name1].shape[3:])
        a2 = host[name2].double().reshape((L if levels else 1), -1, *host[name2].shape[3:])
        fam = torch.from_numpy(t.fam)
        src = torch.from_numpy(t.src)
        out = torch.where(
            (fam == 1).view(1, -1, *([1] * (a1.dim() - 2))),
            a1[:, src.clamp(max=a1.shape[1] - 1)],
            a2[:, src.clamp(max=a2.shape[1] - 1)],
        )
        return out

    f = per_edge("normalThicknessFlux1", "normalThicknessFlux2")[..., 0]  # [L,E]
    mk = per_edge("advMaskHighOrder1", "advMaskHighOrder2")[..., 0]  # [L,E]
    a = per_edge("advCoefs1", "advCoefs2", levels=False)[0]  # [E,10]
    b = per_edge("advCoefs3rd1", "advCoefs3rd2", levels=False)[0]  # [E,10]
    dv = per_edge("dvEdge1", "dvEdge2", levels=False)[0, :, 0]  # [E]
    es1 = host["edgeSignOnCell1"].double().reshape(1, -1, 1)
    es2 = host["edgeSignOnCell2"].double().reshape(1, -1, 1)
    sgn_src = {"edgeSignOnCell1": es1, "edgeSignOnCell2": es2}
    host2 = dict(host, edgeSignOnCell1=es1.reshape(1, *t.shape1, 1), edgeSignOnCell2=es2.reshape(1, *t.shape2, 1))
    del sgn_src
    es = per_edge.__wrapped__ if hasattr(per_edge, "__wrapped__") else None
    fam = torch.from_numpy(t.fam)
    src = torch.from_numpy(t.src)
    sign = torch.where(fam == 1, es1[0, src.clamp(max=es1.shape[1] - 1), 0], es2[0, src.clamp(max=es2.shape[1] - 1), 0])
    nbr = torch.from_numpy(t.nbr)
    w = (f * mk)[..., None] * (a[None] + 0.25 * torch.sign(f)[..., None]) * b[None]  # [L,E,10]
    F = (w * cell[:, nbr]).sum(-1)
    has_low = torch.from_numpy(t.low[:, 0] >= 0)
    lo = torch.from_numpy(t.low.clip(min=0))
    lowterm = 0.5 * dv[None] * (1 - mk) * f * (cell[:, lo[:, 0]] + cell[:, lo[:, 1]])
    F = sign[None] * (F + torch.where(has_low[None], lowterm, torch.zeros_like(lowterm)))
    out = F[:, torch.from_numpy(t.out_edges)].sum(-1) / host["areaCell"].double()[torch.from_numpy(t.out_col)][None]
    out = out.reshape(L, n, n)
    return out[:, : n // 2], out[:, n // 2 :]


@dataclass
class Group:
    name: str
    fam: int  # 1 slanted, 2 vertical
    pr: int  # edge row parity within the family grid
    pc: int | None  # edge column parity (family 1 only)
    rows: int  # group grid rows  (edge row = 2*r + pr)
    cols: int  # group grid cols  (edge col = 2*c + pc, or c)
    taps: list  # 10 x (dR, dC): neighbour cell = (2r + dR, c + dC)
    low: list | None  # 2 x (dR, dC) for the 2nd-order pair (family 1 only)

    def src(self, r, c, n):
        if self.fam == 1:
            return (2 * r + self.pr) * (2 * n + 1) + (2 * c + self.pc)
        return (2 * r + self.pr) * (n + 1) + c


def group_plan(n: int) -> list[Group]:
    """The 6 edge groups as regular grids with constant tap offsets, read off the ID-trick tables
    and checked: every edge in a group has the same 10 (+2) offsets."""
    t = build_tables(n)
    m = n + 4
    out = []
    for fam, C in ((1, 2 * n + 1), (2, n + 1)):
        sel = np.where(t.fam == fam)[0]
        er, ec = np.divmod(t.src[sel], C)
        for pr in (0, 1):
            for pc in (0, 1) if fam == 1 else (None,):
                g = sel[(er % 2 == pr) & ((ec % 2 == pc) if pc is not None else True)]
                r_, c_ = np.divmod(t.src[g], C)
                gr, gc = r_ // 2, (c_ // 2 if fam == 1 else c_)
                nr, nc = np.divmod(t.nbr[g], m)
                dR, dC = nr - 2 * gr[:, None], nc - gc[:, None]
                assert (dR == dR[0]).all() and (dC == dC[0]).all()
                low = None
                if fam == 1:
                    lr, lc = np.divmod(t.low[g], m)
                    lR, lC = lr - 2 * gr[:, None], lc - gc[:, None]
                    assert (lR == lR[0]).all() and (lC == lC[0]).all()
                    low = list(zip(lR[0].tolist(), lC[0].tolist()))
                assert gr.min() == 0 and gc.min() == 0
                out.append(
                    Group(
                        name=f"f{fam}r{pr}c{pc}",
                        fam=fam,
                        pr=pr,
                        pc=pc,
                        rows=int(gr.max()) + 1,
                        cols=int(gc.max()) + 1,
                        taps=list(zip(dR[0].tolist(), dC[0].tolist())),
                        low=low,
                    )
                )
    return out
