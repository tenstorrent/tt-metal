# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os, sys, types, itertools, importlib.util
from dataclasses import dataclass


# Stub pipeline_builder.graph so the pure planner imports without ttnn.
@dataclass(frozen=True)
class CarveSlot:
    mesh_id: int
    shape: tuple
    offset: tuple


pkg = types.ModuleType("pipeline_builder")
pkg.__path__ = []
graph = types.ModuleType("pipeline_builder.graph")
graph.CarveSlot = CarveSlot
graph.validate_carve_plan = lambda *a, **k: None
sys.modules["pipeline_builder"] = pkg
sys.modules["pipeline_builder.graph"] = graph
TT_BLAZE = os.environ.get("TT_BLAZE")
if not TT_BLAZE:
    sys.exit("set TT_BLAZE to a tt-blaze checkout (needs pipeline_builder/carve_planner.py)")
spec = importlib.util.spec_from_file_location("cp", os.path.join(TT_BLAZE, "pipeline_builder", "carve_planner.py"))
cp = importlib.util.module_from_spec(spec)
sys.modules["cp"] = cp
spec.loader.exec_module(cp)


def grid(mesh, R, C):
    s = [cp.HostSlice(mesh, r, c, 1, 1) for r in range(R) for c in range(C)]
    adj = set()
    for r in range(R):
        for c in range(C):
            if c + 1 < C:
                adj.add(frozenset({(mesh, r, c), (mesh, r, c + 1)}))
            if r + 1 < R:
                adj.add(frozenset({(mesh, r, c), (mesh, r + 1, c)}))
    return s, adj


def stage_level_exists(R, C, shapes):
    """Ring of rectangles: disjoint, consecutive ones (and last->first) share an edge. Brute force."""

    def rects(h, w):
        return [(r, c, h, w) for r in range(R - h + 1) for c in range(C - w + 1)]

    def cells(x):
        r, c, h, w = x
        return {(i, j) for i in range(r, r + h) for j in range(c, c + w)}

    def touch(a, b):
        A, B = cells(a), cells(b)
        return any((i + di, j + dj) in B for i, j in A for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)))

    def go(k, used, placed):
        if k == len(shapes):
            return placed if touch(placed[-1], placed[0]) else None
        for x in rects(*shapes[k]):
            cx = cells(x)
            if cx & used or (placed and not touch(placed[-1], x)):
                continue
            got = go(k + 1, used | cx, placed + [x])
            if got:
                return got
        return None

    return go(0, set(), [])


def probe(R, C, shapes):
    s, adj = grid(0, R, C)
    try:
        plan = cp.plan_ring_carve(s, adj, shapes, {0: (R, C)})
        planner = "plan: " + ", ".join(f"{p.shape}@{p.offset}" for p in plan[: len(shapes)])
    except cp.CarvePlanError as e:
        planner = "CarvePlanError: " + str(e).splitlines()[0][:110]
    real = stage_level_exists(R, C, shapes)
    print(f"{R}x{C} slices, stages {shapes}\n  planner -> {planner}\n  stage-level ring exists -> {real}\n")


probe(4, 4, [(1, 1), (2, 2), (2, 2), (2, 2)])  # 13 slices: odd
probe(4, 4, [(1, 1), (2, 2), (2, 2), (2, 2), (1, 1)])  # 14 slices
probe(4, 4, [(1, 1), (2, 2), (2, 2), (1, 1)])  # 10 slices
probe(4, 6, [(1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (1, 1)])  # 18 slices
probe(4, 8, [(1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (1, 1)])  # 26 slices
probe(4, 8, [(1, 1), (2, 4), (2, 4), (2, 2), (1, 1)])  # mixed wide stages
