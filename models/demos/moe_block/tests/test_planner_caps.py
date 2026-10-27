# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only model of the planners' worst-case capacity (no device): whatever the routing, no planner instance gets more
entries or rows than its L1 holds, and no chip more rows than the plan tensors hold - so packed routing never drops a
row (a dropped row left a zero table entry the combine waited on forever).

routing_a2a's planner (the fused head): instance q takes the experts between the expert edges nearest q T / M and
(q + 1) T / M (the kernel's bound()), its entries at most inst_cap, its rows (each expert padded to 32) at most
inst_cap + 32 EPC. combine_plan's multi-core planner (the ops head): core p takes the 32-row tiles [p NT / M,
(p + 1) NT / M), at most _mc2_inst_rows."""

import random

import pytest
from ttnn.bringup.combine_rf.combine_plan import _mc2_inst_rows
from ttnn.bringup.combine_rf.routing_a2a import inst_cap, worst_rows


def _bound(tot, M, j):
    """The kernel's bound(j): the expert edge nearest j T / M."""
    EPC, T = len(tot), sum(tot)
    if j == 0:
        return 0
    if j >= M:
        return EPC
    c = 0
    for e in range(EPC):
        c1 = c + tot[e]
        if c1 * M >= j * T:
            return e if 2 * j * T <= (c + c1) * M else e + 1
        c = c1
    return EPC


def _routing(kind, N, S, K, EPC, rng):
    """Per chip's experts, the entries routed to them (tot[e]) under a routing kind, for the most loaded chip."""
    if kind == "uniform":  # (every token's K picks spread over N EPC experts: this chip's share)
        tot = [0] * EPC
        for _ in range(N * S):
            for g in rng.sample(range(N * EPC), K):
                if g < EPC:
                    tot[g] += 1
        return tot
    if kind == "packed":  # (every token's K picks on this chip: the worst case, T = N S min(K, EPC))
        tot = [0] * EPC
        for _ in range(N * S):
            for e in rng.sample(range(EPC), min(K, EPC)):
                tot[e] += 1
        return tot
    if kind == "hot_first":  # (K experts take every token: the first K, the rest empty)
        return [N * S if e < min(K, EPC) else 0 for e in range(EPC)]
    if kind == "hot_last":
        return [N * S if e >= EPC - min(K, EPC) else 0 for e in range(EPC)]
    if kind == "alternate":  # (every other expert hot)
        hot = list(range(0, EPC, 2))[: min(K, EPC)]
        return [N * S if e in hot else 0 for e in range(EPC)]
    raise ValueError(kind)


SHAPES = [  # (N, S, K, EPC, M): the LoudBox 2x4 / ring, the Galaxy 4x4 / 8x4 at K3 / GLM shapes, M3, DeepSeek V4.1
    (8, 160, 16, 28, 10),
    (8, 256, 16, 28, 14),
    (4, 640, 16, 28, 10),
    (16, 320, 16, 56, 14),
    (32, 160, 16, 28, 10),
    (32, 256, 16, 28, 14),
    (8, 160, 8, 32, 10),
    (8, 256, 8, 32, 14),
    (8, 128, 4, 16, 8),
    (8, 128, 6, 48, 12),
]


@pytest.mark.parametrize("shape", SHAPES, ids=[f"N{s[0]}-S{s[1]}-K{s[2]}-E{s[3]}-M{s[4]}" for s in SHAPES])
@pytest.mark.parametrize("kind", ["uniform", "packed", "hot_first", "hot_last", "alternate"])
def test_planner_caps(shape, kind):
    N, S, K, EPC, M = shape
    rng = random.Random(hash((shape, kind)) & 0xFFFF)
    for _ in range(3 if kind in ("uniform", "packed") else 1):
        tot = _routing(kind, N, S, K, EPC, rng)
        T = sum(tot)
        assert T <= N * S * min(K, EPC)
        rows = sum((t + 31) // 32 * 32 for t in tot)
        assert rows <= worst_rows(N, S, K, EPC), (rows, worst_rows(N, S, K, EPC))
        cap = inst_cap(N, S, K, EPC, M)
        for q in range(M):  # (the fused head's planner instances)
            e0, e1 = _bound(tot, M, q), _bound(tot, M, q + 1)
            ent = sum(tot[e0:e1])
            assert ent <= cap, (q, ent, cap)
            assert sum((t + 31) // 32 * 32 for t in tot[e0:e1]) <= cap + 32 * EPC
        NT, IR = rows // 32, _mc2_inst_rows(worst_rows(N, S, K, EPC), M)
        for p in range(M):  # (the ops head's multi-core planner cores)
            assert ((p + 1) * NT // M - p * NT // M) * 32 <= IR


def test_old_cap_overflowed_packed():
    """Not vacuous: the old fused-head cap (2 S K + 32 EPC: twice the average) is short of packed routing's rows."""
    N, S, K, EPC, M = SHAPES[0]
    tot = _routing("packed", N, S, K, EPC, random.Random(0))
    assert sum((t + 31) // 32 * 32 for t in tot) > 2 * S * K + 32 * EPC
