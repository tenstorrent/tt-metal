# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""layout.Window.cost compiled with numba: the same model, the same results, about 10 times faster.

The numpy model steps every candidate through the window together, one numpy call per instruction; here each candidate
runs the window on its own in compiled code. Candidates that a model part cannot tell apart run once for that part:
  - icache: a shift of whole lines common to all the window's code renames every line and every set alike, so the misses
    depend on the shifts only relative to the first touched segment's whole lines;
  - predictor: indexed by address bits 2 to 8 and its targets unshifted, so it depends on the shifts modulo 512.
layout.Window.cost uses it when numba imports, and the numpy model otherwise.
"""
import numpy as np
from numba import njit

from .layout import HAZ_COST, K_PRED, MIS_COST, MISS_COST


@njit(cache=True)
def _icache(pcs, seg, sh, counted, nsets):
    """misses per candidate, sh [candidates x segments]; layout.Window.cost's instruction cache"""
    misses = np.zeros(sh.shape[0], np.int64)
    w0 = np.empty(nsets, np.int32)
    w1 = np.empty(nsets, np.int32)
    fill = np.empty(nsets, np.int64)
    for c in range(sh.shape[0]):
        w0[:] = -1
        w1[:] = -1
        fill[:] = 0
        last = np.int32(-1)
        m = 0
        for k in range(pcs.shape[0]):
            ln = np.int32((np.int32(pcs[k]) + sh[c, seg[k]]) >> 4)
            if ln == last:
                continue
            last = ln
            st = ln % nsets
            if w0[st] == ln or w1[st] == ln:
                continue
            f = fill[st]
            if f & 1:
                w1[st] = ln
            else:
                w0[st] = ln
            fill[st] = f + 1
            if counted[k]:
                m += 1
        misses[c] = m
    return misses


@njit(cache=True)
def _predictor(ctl, pc, taken, nxt, first, seg, sh, counted):
    """(mispredicts per candidate, mispredicted [candidates x control records]); layout.Window.cost's predictor"""
    n, R = sh.shape[0], ctl.shape[0]
    mispred = np.zeros(n, np.int64)
    wrong = np.zeros((n, R), np.bool_)
    cnt = np.empty(16, np.int64)
    tgt = np.empty(16, np.int64)
    u_idx = np.empty(R, np.int64)
    u_cnt = np.empty(R, np.int64)
    for c in range(n):
        cnt[:] = 0
        tgt[:] = 0
        h = 0
        m_total = 0
        for r in range(R):
            k = ctl[r]
            # updates are seen K_PRED instructions later
            while h < r and ctl[h] + K_PRED <= k:
                cnt[u_idx[h]] = u_cnt[h]
                tgt[u_idx[h]] = nxt[h]
                h += 1
            a = pc[r] + np.int64(sh[c, seg[k]])
            idx = (
                ((1 - (((a >> 8) & 1) ^ ((a >> 6) & 1))) << 3)
                | ((((a >> 7) & 1) ^ ((a >> 5) & 1)) << 2)
                | ((1 - (((a >> 4) & 1) ^ ((a >> 3) & 1))) << 1)
                | ((a >> 2) & 1)
            )
            cv = cnt[idx]
            e = tgt[idx]
            t = taken[r]
            if first[r]:
                m = t
            else:
                m = (e != nxt[r]) or (((cv >= 0) and (e != pc[r])) != t)
            if counted[k] and m:
                m_total += 1
            wrong[c, r] = m
            right = e == nxt[r]
            if t:
                nc = 1 if not right else min(cv + 1, 1)
            else:
                nc = -2 if not right else max(cv - 1, -2)
            u_idx[r] = idx
            u_cnt[r] = nc
        mispred[c] = m_total
    return mispred, wrong


@njit(cache=True)
def _hazards(at, const, target, r0, r1, m0, m1, pcs, seg, sh, tsh, wrong, cls):
    """false hazards per candidate; tsh [candidates x sites] the branch target's shift, cls the predictor class"""
    haz = np.zeros(sh.shape[0], np.int64)
    for c in range(sh.shape[0]):
        b = cls[c]
        s = 0
        for j in range(at.shape[0]):
            i = at[j]
            if const[j]:
                h = True
            else:
                offs = (target[j] + tsh[c, j]) - (pcs[i] + np.int64(sh[c, seg[i]]))
                f = ((offs >> 1) & 0xF) << 1 | ((offs >> 11) & 1)
                h = f != 0 and (f == r0[j] or f == r1[j])
            if h and m0[j] >= 0 and wrong[b, m0[j]]:
                h = False
            if h and m1[j] >= 0 and wrong[b, m1[j]]:
                h = False
            if h:
                s += 1
        haz[c] = s
    return haz


def _classes(keys):
    """(representative rows, inverse) of the distinct rows of keys [candidates x k]"""
    _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return first, inv.ravel()


def _arrays(win):
    """the window's control records and hazard sites as arrays, once per window"""
    a = getattr(win, "_fast", None)
    if a is not None:
        return a
    rec = win.records
    pc = np.array([r[0] for r in rec], np.int64)
    taken = np.array([bool(r[1]) for r in rec], np.bool_)
    nxt = np.array([r[2] if r[1] else r[0] + 4 for r in rec], np.int64)
    # the first execution of a control instruction is not predicted
    first = np.zeros(len(rec), np.bool_)
    seen = set()
    for i, p in enumerate(pc.tolist()):
        if p not in seen:
            seen.add(p)
            first[i] = True
    row = {int(k): i for i, k in enumerate(win.ctl)}
    haz = win.haz
    a = win._fast = dict(
        pc=pc,
        taken=taken,
        nxt=nxt,
        first=first,
        ctl=np.asarray(win.ctl, np.int64).reshape(-1),
        at=np.array([h[0] for h in haz], np.int64).reshape(-1),
        const=np.array([bool(h[1]) for h in haz], np.bool_).reshape(-1),
        target=np.array([h[3] or 0 for h in haz], np.int64).reshape(-1),
        r0=np.array([h[4][0] for h in haz], np.int64).reshape(-1),
        r1=np.array([h[4][-1] for h in haz], np.int64).reshape(-1),
        m0=np.array([row.get(h[0], -1) for h in haz], np.int64).reshape(-1),
        m1=np.array([row.get(h[0] + 1, -1) for h in haz], np.int64).reshape(-1),
        counted=np.ascontiguousarray(win.counted, np.bool_),
        pcs=np.ascontiguousarray(win.pcs, np.int64),
    )
    return a


def cost(win, P, Z, nsets, actual=None):
    """layout.Window.cost"""
    a = _arrays(win)
    P = np.asarray(P, np.int64)
    Z = np.asarray(Z, np.int64)
    n = len(P)
    tsh = np.zeros((n, len(win.haz)), np.int64)
    if actual is None:
        segs = win.amap.segment_shifts(P, Z).astype(np.int32)
        sh = np.ascontiguousarray(segs.T)
        seg = np.ascontiguousarray(win.seg, np.int64)
        for j, (_, const, tseg, _, _) in enumerate(win.haz):
            if not const:
                tsh[:, j] = segs[tseg[0]]
    else:
        # one layout; every non-fill instruction is its own segment
        actual = np.asarray(actual, np.int64)
        sh = actual[None, :].astype(np.int32)
        seg = np.ascontiguousarray(win.idx, np.int64)
        for j, (_, const, tseg, _, _) in enumerate(win.haz):
            if not const:
                tsh[:, j] = int(actual[tseg[1]])
    pcs, counted = a["pcs"], a["counted"]
    pred = (a["ctl"], a["pc"], a["taken"], a["nxt"], a["first"], seg)
    if actual is None and n > 1:
        st = segs[np.unique(win.seg)].T.astype(np.int64)
        rep, inv = _classes(st - (st[:, :1] // 16) * 16)
        misses = _icache(pcs, seg, sh[rep], counted, nsets)[inv]
        rep, cls = _classes(st % 512)
        mp, wrong = _predictor(*pred, sh[rep], counted)
        mispred = mp[cls]
    else:
        misses = _icache(pcs, seg, sh, counted, nsets)
        mispred, wrong = _predictor(*pred, sh, counted)
        cls = np.zeros(n, np.int64)
    haz = _hazards(
        a["at"],
        a["const"],
        a["target"],
        a["r0"],
        a["r1"],
        a["m0"],
        a["m1"],
        pcs,
        seg,
        sh,
        tsh,
        wrong,
        cls,
    )
    total = MIS_COST * mispred + MISS_COST * misses + HAZ_COST * haz
    return total / win.n, mispred, misses, haz
