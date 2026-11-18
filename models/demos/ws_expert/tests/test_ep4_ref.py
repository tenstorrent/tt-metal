# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""EP-4 host reference (ep4_ref.py), no device: the layout's invariants at the full load, the readiness curve F(x)
against the x^3 model for the LoudBox and Galaxy stream orders, and the O / partial capacity statistics.
EP4_LEN: Galaxy-equivalent length (302K / 588K / 874K / 1M)."""

import os
import time

import pytest
import torch
from loguru import logger

from models.demos.ws_expert.ep4_ref import (
    GLX,
    LB,
    Geo,
    chip_layout,
    chip_rows,
    expert_cv,
    hidden_share,
    lane_orders,
    n_per_chip,
    partial_counts,
    positions,
    readiness,
    ready_lists,
    routing,
)

LEN = os.environ.get("EP4_LEN", "302K")
# section 2's speed of light (ms): expert phase S and the combine C, at 302K-eq (LoudBox @50 / @40 GB/s, Galaxy)
SOL = {"LB": (3.40, 2.81), "LB@40": (4.15, 3.51), "GLX": (5.81, 2.81)}


def check_chip(geo: Geo, ids, w, r, c):
    """The layout's invariants on chip (r, c)."""
    n = ids.shape[2]
    S = -(-n // 32) * 32
    ch = chip_layout(geo, ids, w, r, c, keep_pairs=True)
    gi = ids[r]
    sel = (gi // geo.EPC) == c
    assert ch.R_rows == int(sel.sum())
    # the O rows: a permutation of 0..R-1, every run's rows contiguous from its base
    rows = torch.cat(ch.lists_row)
    assert torch.equal(torch.sort(rows).values, torch.arange(ch.R_rows))
    assert int(ch.run_cnt.sum()) == ch.R_rows
    assert torch.equal(ch.run_base[1:], (ch.run_base + ch.run_cnt)[:-1])
    # every list entry: the token's pair on this expert, its O row
    for le in range(0, geo.EPC, 7):
        tok, row = ch.lists_tok[le], ch.lists_row[le]
        assert (tok[1:] > tok[:-1]).all(), "a list in (home, t) order"
        h, t = tok // S, tok % S
        k = (gi[h, t] == le + geo.EPC * c).long().argmax(-1)
        assert ((gi[h, t] == le + geo.EPC * c).sum(-1) == 1).all()
        assert torch.equal(ch.pair_row[h, t, k], row)
    # runs: experts / weights in k order, the j-th row of a run is its j-th expert's
    j = torch.arange(geo.K).view(1, -1)
    vm = j < ch.run_cnt.view(-1, 1)
    assert torch.equal(vm, ch.run_exp >= 0)
    assert (ch.run_w[~vm] == 0).all()
    # mask / counters: the home side of the same routing
    assert int(ch.cnt[c]) == int(((ch.mask >> c) & 1).sum())
    return ch


@pytest.mark.parametrize("kind", ["uniform", "zipf"])
def test_ep4_layout(kind):
    n = n_per_chip(LEN)
    t0 = time.time()
    ids, w = routing(LB, n, kind, seed=0)
    for r, c in ((0, 0), (1, 3)):
        check_chip(LB, ids, w, r, c)
    logger.info(f"EP4 ref {kind} LB {LEN} (n {n}): routing + 2 chips' layout and checks {time.time() - t0:.1f} s")


def orders_lb():
    return {
        "lane G2": lane_orders(2, 160, 2, "lane"),
        "own-first G2": lane_orders(2, 160, 2, "own_first"),
        "interl. G2": lane_orders(2, 160, 2, "interleaved"),
        "split G4": lane_orders(2, 160, 4, "split", lane_groups=2),
    }


def orders_glx():
    return {
        "lane G4": lane_orders(8, 160, 4, "lane"),
        "own-first G4": lane_orders(8, 160, 4, "own_first"),
        "interl. G4": lane_orders(8, 160, 4, "interleaved"),
    }


def test_ep4_readiness():
    """F(x) at x = .25 / .5 / .75 / .9 for chip (0, 0) and the busiest chip of mesh row 0, against x^c (each run's
    experts at random positions) and x^3; the hidden share of the combine in the fluid model (ep4_ref.hidden_share)."""
    for gname, geo, orders, sols in (("LB", LB, orders_lb(), ("LB", "LB@40")), ("GLX", GLX, orders_glx(), ("GLX",))):
        n = n_per_chip(LEN)
        if geo.R == 8:
            n = min(n, 4096)  # (readiness is a distribution: a smaller n keeps the Galaxy run short)
        for kind in ("uniform", "zipf"):
            ids, w = routing(geo, n, kind, seed=0)
            rows = chip_rows(geo, ids)[0]
            cb = int(rows.argmax())
            for c in sorted({0, cb}):
                ch = chip_layout(geo, ids, w, 0, c)
                rc = ch.run_cnt.double()
                logger.info(
                    f"{gname} {kind} chip (0,{c}): runs {rc.numel()} ({rc.numel() / (4 * n):.3f} of the row's "
                    f"tokens), experts a run {rc.mean():.2f}, O rows {ch.R_rows / n:.2f} n"
                )
                for name, per in orders.items():
                    pos, key = positions(per[0], geo.EPC)
                    grid, F, model = readiness(ch, pos, key)
                    at = lambda v: " ".join(f"{float(v[int(x * 100)]):.3f}" for x in (0.25, 0.5, 0.75, 0.9))
                    line = f"  {name:13s} F {at(F)} | x^c {at(model)} | x^3 {at(grid ** 3)}"
                    for s in sols:
                        S, C = (v * (n / n_per_chip("302K")) if geo.R == 2 else v for v in SOL[s])
                        h, tail = hidden_share(grid, F, S, C)
                        h3, t3 = hidden_share(grid, grid**3, S, C)
                        line += f" | {s}: hidden {h * 100:.0f}% tail {tail:.2f} ms (x^3 {h3 * 100:.0f}%, {t3:.2f})"
                    logger.info(line)
                rl, _ = ready_lists(ch, positions(orders[next(iter(orders))][0], geo.EPC)[1], geo.EPC)
                assert sum(len(v) for v in rl) == rc.numel()


def test_ep4_capacity():
    """Busiest chip's O rows and the (source, home) partial counts, both routings, 10 seeds; LoudBox and Galaxy."""
    for gname, geo in (("LB", LB), ("GLX", GLX)):
        n = n_per_chip(LEN) if geo.R == 2 else 4096
        for kind in ("uniform", "zipf"):
            mx, pm, cvs = [], [], []
            for seed in range(10):
                ids, _ = routing(geo, n, kind, seed=seed)
                rows = chip_rows(geo, ids).double()
                pc = partial_counts(geo, ids).double()
                mx.append(float(rows.max() / (12 * n)))
                pm.append(float(pc.max() / n))
                cvs.append(expert_cv(ids, geo.E))
            t = torch.tensor(mx)
            logger.info(
                f"{gname} {kind} n {n}: expert CV {sum(cvs) / len(cvs):.2f}; busiest chip's O rows / 12n: mean "
                f"{t.mean():.3f} max {t.max():.3f}; partials a (src, home) max / n: {max(pm):.3f} (mean p "
                f"{1 - 0.75 ** 12:.3f})"
            )
