# SPDX-License-Identifier: Apache-2.0
"""tp32-prefill2: host plan builder for the TP32 grouped-GEMM prefill (K1 gate/up + K2 down + owner combine).

Input: ids [Tt, 8] (expert ids), scores [Tt, 8], Tt = 4 users x N tokens (all tokens of the chunk, identical on every chip).
Output tensors (torch, all chip-identical): see Plan.

Conventions
  * pair p = (token t, slot j); expert e = ids[t, j]; pairs are grouped by expert (stable, by (e, t, j)), every expert's pairs are
    padded to an EVEN number of M-tiles (32 rows each); M-tile mt of expert e starts at global M-tile start[e].
  * unit = (expert e, first M-tile mt0, M-tile count mtc (even)); each core runs a list of units (LPT balanced by cost).
  * gidx[mt // 2][64]: token of every padded row (pad rows -> token 0).
  * dest[mt // 2][64]: u32 per padded row for the K2 combine: 0xFFFFFFFF = pad; else owner << 16 | slot j << 8 | token % 32,
    owner = token // 32 (an owner core o < Tt // 32 collects the 8 partial rows of the 32 tokens t in [32 o, 32 o + 32)).
  * sc[mt]: bf16 tile whose column 0 holds the score of the 32 rows of M-tile mt (K1 scales the act rows by it; padded rows 0).
"""
from dataclasses import dataclass

import torch

PLAN_WORDS = 512  # u32 per core plan page: [nu, (e, mt0, mtc) x nu]
MAX_UNITS = 100


@dataclass
class Plan:
    plan: torch.Tensor  # [nc, 512] int32
    gidx: torch.Tensor  # [mt_cap // 2, 64] int32
    dest: torch.Tensor  # [mt_cap // 2, 64] int32 (bit pattern of u32)
    sc: torch.Tensor  # [mt_cap * 32, 32] bf16 (TILE layout when uploaded)
    mt_used: int
    mts: list
    start: list
    max_load: float


def mt_cap(tt, ne=256):
    """static bound on the padded M-tile count: ceil(pairs / 32) + one extra pair-of-tiles per expert, even."""
    c = (tt * 8 + 31) // 32 + 2 * ne
    return (c + 1) // 2 * 2


def build_plan(ids, scores, nc=110, ne=256, unit_cost=(10, 8), split=0):
    """ids [Tt, 8] int, scores [Tt, 8] float -> Plan. split>0: experts with more than `split` M-tiles are cut into units of <= split."""
    tt = ids.shape[0]
    assert tt % 32 == 0
    cap = mt_cap(tt, ne)
    flat = ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat * (tt * 8) + torch.arange(tt * 8), stable=True)  # by (e, p) with p = t * 8 + j
    cnt = torch.bincount(flat, minlength=ne)
    mts = [((int(c) + 31) // 32 + 1) // 2 * 2 for c in cnt]
    start, s = [], 0
    for m in mts:
        start.append(s)
        s += m
    assert s <= cap, (s, cap)
    gidx = torch.zeros(cap * 32, dtype=torch.int32)
    dest = torch.full((cap * 32,), -1, dtype=torch.int32)
    scv = torch.zeros(cap * 32, dtype=torch.float32)
    pos = 0
    for e in range(ne):
        c = int(cnt[e])
        ps = order[pos : pos + c]
        pos += c
        r0 = start[e] * 32
        t = ps // 8
        j = ps % 8
        gidx[r0 : r0 + c] = t.to(torch.int32)
        dest[r0 : r0 + c] = ((t // 32) << 16 | (j << 8) | (t % 32)).to(torch.int32)
        scv[r0 : r0 + c] = scores.reshape(-1)[ps].float()
    # score tiles: element (r, 0) of tile mt = score of row mt * 32 + r
    sc = torch.zeros(cap, 32, 32, dtype=torch.float32)
    sc[:, :, 0] = scv.reshape(cap, 32)
    # units
    units_all = []
    for e in range(ne):
        m = mts[e]
        if m == 0:
            continue
        if split and m > split:
            k = 0
            while k < m:
                c = min(split, m - k)
                units_all.append((e, start[e] + k, c))
                k += c
        else:
            units_all.append((e, start[e], m))
    units_all.sort(key=lambda u: -u[2])
    load = [0.0] * nc
    units = [[] for _ in range(nc)]
    for u in units_all:
        c = min(range(nc), key=lambda i: load[i])
        units[c].append(u)
        load[c] += u[2] * unit_cost[0] + unit_cost[1]
    plan = torch.zeros(nc, PLAN_WORDS, dtype=torch.int32)
    for c in range(nc):
        assert len(units[c]) <= MAX_UNITS, len(units[c])
        plan[c, 0] = len(units[c])
        for k, (e, s0, m) in enumerate(units[c]):
            plan[c, 1 + 3 * k : 4 + 3 * k] = torch.tensor([e, s0, m], dtype=torch.int32)
    return Plan(
        plan=plan,
        gidx=gidx.reshape(cap // 2, 64),
        dest=dest.reshape(cap // 2, 64),
        sc=sc.reshape(cap * 32, 32).to(torch.bfloat16),
        mt_used=s,
        mts=mts,
        start=start,
        max_load=max(load),
    )
