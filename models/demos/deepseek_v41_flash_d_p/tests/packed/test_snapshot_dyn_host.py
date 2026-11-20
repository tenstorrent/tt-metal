# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7C (host only, no device): the dynamic window-snapshot broadcast's role rule (tt/snapshot_send._TREE_CODE, run
by every chip from the block codes) gives every chip exactly the home kinds and port roles of SnapshotSend.plan (the
per-pattern programs it replaces), for every owner, on the four boxes, with and without torus wraps.

    PYTHONPATH=$PWD python -m pytest -q models/demos/deepseek_v41_flash_d_p/tests/packed/test_snapshot_dyn_host.py
"""

import itertools

from models.demos.deepseek_v41_flash_d_p.tt import snapshot_send as S

ZERO, LOCAL = S.CODE_ZERO, S.CODE_LOCAL


def _step(n, a, b, wrap):
    if wrap:
        f = (b + n - a) % n
        return 1 if f * 2 <= n else -1
    return 1 if b > a else -1


def _parent(r, c, r0, c0, R, C, wx, wy):  # (the kernels' snap_parent)
    if c != c0:
        return (r, (c + C + _step(C, c, c0, wx)) % C)
    return ((r + R + _step(R, r, r0, wy)) % R, c)


def _child(o, me, nb, R, C, wx, wy):  # (the kernels' snap_child)
    r0, c0 = divmod(o, C)
    return nb != (r0, c0) and _parent(nb[0], nb[1], r0, c0, R, C, wx, wy) == me


def _chips(R, C, wx, wy):
    out = {}
    for r in range(R):
        for c in range(C):
            d = {}
            if r + 1 < R or (wy and R > 2):
                d["Y+"] = {"peer": ((r + 1) % R, c)}
            if r > 0 or (wy and R > 2):
                d["Y-"] = {"peer": ((r - 1) % R, c)}
            if c + 1 < C or (wx and C > 2):
                d["X+"] = {"peer": (r, (c + 1) % C)}
            if c > 0 or (wx and C > 2):
                d["X-"] = {"peer": (r, (c - 1) % C)}
            out[(r, c)] = d
    return out


def test_snapshot_dyn_roles():
    bad, checked = 0, 0
    for R, C in ((8, 4), (4, 4), (2, 4), (2, 2)):
        for wx, wy in itertools.product((False, True), repeat=2):
            ss = S.SnapshotSend.__new__(S.SnapshotSend)
            ss.R, ss.C, ss.n, ss.wrap_x, ss.wrap_y, ss.trees = R, C, R * C, wx, wy, {}
            ss.chips = _chips(R, C, wx, wy)
            for o1, o2 in itertools.product(range(R * C), repeat=2):
                pattern = (o1, o1, o2, "local") if (o1 + o2) % 3 else ("zero", o1, o2, o2)
                plan = ss.plan(pattern)
                for (r, c), v in plan.items():
                    me = (r, c)
                    for b, k in enumerate(pattern):
                        code = ZERO if k == "zero" else LOCAL if k == "local" else k
                        kind = 0 if code == ZERO else 1 if code in (LOCAL, r * C + c) else 2
                        bad += kind != v["kinds"][b]
                        for d in sorted(S.DIRS.index(x) for x in ss.chips[me]):
                            nb = tuple(ss.chips[me][S.DIRS[d]]["peer"])
                            role = 0
                            if code not in (ZERO, LOCAL) and _child(code, me, nb, R, C, wx, wy):
                                role = 1 if code == r * C + c else 2
                            bad += role != v["ports"].get(d, [0] * S.BLOCKS)[b]
                            checked += 1
    print(f"snapshot dynamic roles: {checked} (chip, block, port) checked, {bad} differ from plan()")
    assert bad == 0
