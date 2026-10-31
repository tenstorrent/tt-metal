# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free tests of APC M0' (tt/apc_grid.py + the PrefixCache hooks): which boundaries get a snapshot, hits against the design simulator (tools/apc_paged_sim.py), budget / LRU / thinning,
text-only keys unchanged.

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_apc_grid.py
"""

import types

import numpy as np
import torch

from models.demos.blackhole.deepseek_v41_flash.tools import apc_paged_sim as SIM
from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as PC
from models.demos.blackhole.deepseek_v41_flash.tt.apc_grid import GridPolicy, pow2_wants

C = 512


def seg(seed, n):
    return SIM.seg(seed, n)


def test_plan_tip_tip2_and_divergence():
    g = GridPolicy(C, 128)
    t = seg(1, 10000)
    assert g.plan(t) == {9728: "tip"}  # 10000 % 512 = 272 >= 128: the repeat rule gives the same boundary
    t2 = seg(2, 9700)  # 9700 % 512 = 484: tip 9216 usable by a repeat too
    assert g.plan(t2) == {9216: "tip"}
    t3 = seg(3, 9300)  # 9300 - 9216 = 84 < 128: a repeat can only use 8704
    assert g.plan(t3) == {9216: "tip", 8704: "tip2"}
    sysp = seg(9, 6000)
    p1 = np.concatenate([sysp, seg(10, 3000)])
    p2 = np.concatenate([sysp, seg(11, 3000)])
    assert "div" not in g.plan(p1).values()
    plan2 = g.plan(p2)
    assert (
        plan2[5632] == "div" and 6000 // C * C == 5632 and len(plan2) == 2
    )  # the last fully shared block ends at 5632
    p3 = np.concatenate([sysp, seg(12, 3000)])
    assert g.plan(p3)[5632] == "div"  # still planned (the store dedups against the existing snapshot)


def test_plan_bounds_and_short_prompts():
    g = GridPolicy(C, 128)
    assert g.plan(seg(1, 500)) == {}  # shorter than one chunk
    assert g.plan(seg(1, 600)) == {512: "tip"} and g.plan(seg(1, 600), total=300) == {}
    # total larger than the tokens given: clamped, never an index past the prompt
    assert g.plan(seg(2, 1100), total=5000) == {1024: "tip", 512: "tip2"}
    cap = GridPolicy(C, 128, seen_cap=8)
    for i in range(20):
        cap.plan(seg(100 + i, 4000))
    assert len(cap.seen) <= 8


def test_division_of_labour_with_pow2():
    assert [k for k in range(1, 70) if pow2_wants(k * C)] == [1, 2, 4, 8, 16, 32, 64]


class FakeModel:
    def __init__(self):
        self.snaps = 0


def make_cache(monkeypatch, budget=10**9, grid=True, nbytes=1000):
    if grid:
        monkeypatch.setenv("DSV41_APC_GRID", "512")
    else:
        monkeypatch.delenv("DSV41_APC_GRID", raising=False)
    monkeypatch.delenv("DSV41_APC_FILE", raising=False)

    def begin(model, b, s, toks):
        model.snaps += 1
        return types.SimpleNamespace(s=s, toks=torch.as_tensor(toks)[:s].clone())

    def finish(model, p):
        return PC.PrefixSnapshot(
            s=p.s, toks=p.toks, pool={0: torch.zeros(nbytes // 2, dtype=torch.bfloat16)}, seconds=0.0
        )

    monkeypatch.setattr(PC, "snapshot_begin", begin)
    monkeypatch.setattr(PC, "snapshot_finish", finish)
    return PC.PrefixCache(FakeModel(), C, budget, "pow2", log=lambda *_: None)


def serve(cache, t, b=0):
    """One prompt through the cache the way ``prefill_interleaved`` drives it: lookup, new_prompt, a snapshot call at every chunk boundary the call computes. -> hit position."""
    t = torch.as_tensor(t)
    e = t.numel()
    cache.finish_pending()
    snap = cache.lookup(b, t, e)
    s0 = snap.s if snap else 0
    if cache.grid is not None:
        cache.new_prompt(b, t, e)
    for B in range(s0 + C, e // C * C + 1, C):
        cache.begin_snapshot(b, B, t, True)
    cache.finish_pending()
    return s0


def test_text_keys_unchanged(monkeypatch):
    t = torch.as_tensor(seg(5, 4000))
    a, b = make_cache(monkeypatch, grid=True), make_cache(monkeypatch, grid=False)
    assert a.grid is not None and b.grid is None
    assert a.store.key(t, 2048) == b.store.key(t, 2048)
    assert [B for B in range(C, 4001, C) if a.wants(B)] == [B for B in range(C, 4001, C) if b.wants(B)]


def test_grid_off_is_exactly_the_old_policy(monkeypatch):
    c = make_cache(monkeypatch, grid=False)
    assert serve(c, seg(1, 9000)) == 0
    assert sorted(k[2] for k in c.store.snaps) == [512, 1024, 2048, 4096, 8192]


def test_multi_turn_hits_the_previous_tip(monkeypatch):
    c = make_cache(monkeypatch)
    p = seg(1, 6000)
    assert serve(c, p) == 0
    p2 = np.concatenate([p, seg(2, 500), seg(3, 3900)])  # next turn, 10400
    assert serve(c, p2) == 5632  # the tip (6000 // 512 * 512) of turn 1, not pow2 4096
    p3 = np.concatenate([p2, seg(4, 500), seg(5, 3900)])
    assert serve(c, p3) == 10240
    # the tips of older turns are thinned, the pow2 points stay
    ss = sorted(k[2] for k in c.store.snaps)
    assert 5632 not in ss and 10240 not in ss and 14336 in ss and 4096 in ss and c.stats["thinned"] >= 2


def test_shared_system_prompt_divergence(monkeypatch):
    c = make_cache(monkeypatch)
    sysp = seg(7, 30000)
    hits = [serve(c, np.concatenate([sysp, seg(100 + i, 2000)])) for i in range(4)]
    assert hits[0] == 0 and hits[1] == 16384  # pow2 point; the divergence is learned on the 2nd request
    assert hits[2] == hits[3] == 29696  # boundary below the end of the shared 30000 tokens


def test_snapshot_not_repeated_and_cost_bounded(monkeypatch):
    c = make_cache(monkeypatch)
    p = seg(1, 9000)
    serve(c, p)
    n1 = c.m.snaps
    assert n1 == 5 + 1  # pow2 512..8192 are the 5 usual points (8192 <= 9000), the tip 8704 is the one extra
    serve(c, p)  # repeat: hit at the tip, nothing new is snapshotted
    assert c.m.snaps == n1


def test_budget_lru_still_bounds_the_store(monkeypatch):
    c = make_cache(monkeypatch, budget=12 * 1000)
    for i in range(10):
        serve(c, seg(50 + i, 7000 + 100 * i))
    assert c.store.bytes <= 12 * 1000 and c.store.evictions > 0
    assert c.store.bytes == sum(s.nbytes for s in c.store.snaps.values())


def test_grid_needs_defer(monkeypatch):
    monkeypatch.setenv("DSV41_APC_DEFER", "0")
    c = make_cache(monkeypatch)
    assert c.grid is None


def _policy_run(reqs):
    """Mean recompute fraction and mean simulated TTFT of the real PrefixCache policy on the simulator workloads (hit rule and TTFT model of tools/apc_paged_sim.py)."""
    mp = __import__("pytest").MonkeyPatch()
    try:
        c = make_cache(mp, nbytes=10)
        new = tot = 0
        ttft = []
        for t in reqs:
            e = len(t)
            s = serve(c, t)
            p0 = max(0, (e - SIM.ZR) // C * C)
            ch = sum(SIM.CH_AB if s + k * C >= p0 else SIM.CH_A for k in range(-(-(e - s) // C)))
            ttft.append(SIM.F + ch + (SIM.r_host(s) if s else 0))
            new += e - s
            tot += e
        return new / tot, float(np.mean(ttft))
    finally:
        mp.undo()


def test_against_the_simulator():
    cases = {
        "agents8": SIM.agents(8, 14),
        "agents8_small": SIM.agents(8, 20, sys_len=1500, grow=1600, cap=16400, first=(5000, 6500)),
        "sys6k": SIM.shared_sys(64, 6000),
        "sys30k": SIM.shared_sys(32, 30000),
    }
    for name, reqs in cases.items():
        pow2 = SIM.run(reqs, "host_pow2")
        div = SIM.run(reqs, "host_div")
        fr, tt = _policy_run(reqs)
        print(
            f"{name}: pow2 {pow2['ttft_mean']} div(sim) {div['ttft_mean']} policy {tt:.2f} recompute {fr:.3f} (sim {div['recompute_frac']})"
        )
        assert (
            tt <= div["ttft_mean"] * 1.03
        ), name  # the real policy is as good as the simulated tip + divergence host tier
        assert tt <= pow2["ttft_mean"] + 1e-6, name
        assert fr <= div["recompute_frac"] + 0.02, name
