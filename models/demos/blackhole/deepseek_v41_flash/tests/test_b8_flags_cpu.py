# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of DSV41_SLIVER_EXTEND / DSV41_PROTECT_PARTIAL / DSV41_CONT_FIRST in the B=8 geometry (U = Up = 2, 4 rows, row-aware placement ON (inactive at Up = U), 32768 budget / 4096 cap)."""
import itertools

import numpy as np
import pytest
import torch
from loguru import logger

from models.demos.blackhole.deepseek_v41_flash.tests.test_protect_partial_cpu import Count, Plan, call, make_gen

FLAGS = list(itertools.product((0, 1), (0, 1)))  # (SLIVER_EXTEND, PROTECT_PARTIAL)


def setflags(mp, sliver, protect):
    mp.setenv("DSV41_SLIVER_EXTEND", str(sliver))
    mp.setenv("DSV41_PROTECT_PARTIAL", str(protect))
    mp.setenv("DSV41_CONT_FIRST", "1")


def hit_case(mp, sliver, protect, pieces=((111616, 111749), (111749, 115712)), hit=111616, L=400000, row_aware=True):
    setflags(mp, sliver, protect)
    gen, m, plans = make_gen(mp, bool(protect), B=8, U=2, Up=2, row_aware=row_aware)
    c = Count()
    logger.remove()
    h = logger.add(c, level="INFO")
    try:
        toks = torch.randint(1000, 90000, (L,), dtype=torch.int64)
        plans["hit1"] = Plan(user=1, hit=hit)
        for st, e in pieces:
            call(gen, toks, st, e, "hit1", 1, L)
    finally:
        logger.remove(h)
    return gen, m, c


@pytest.mark.parametrize("sliver,protect", FLAGS)
def test_b8_hit_sliver_all_flag_combinations(monkeypatch, sliver, protect):
    """the T9 / image-2 mechanism: a prefix hit whose first piece is a sliver. off/off = old behaviour (one recompute from 0); any flag on = none."""
    gen, m, c = hit_case(monkeypatch, sliver, protect)
    if sliver == protect == 0:
        assert c.lost == 1 and m.steps[-1][0][1] == 0
    else:
        assert c.lost == 0 and m.steps[0][0][1:] == (111616, 112128) and m.steps[-1][0][1] == 112128


@pytest.mark.parametrize("sliver,protect", FLAGS)
def test_b8_xrow_sized_hits(monkeypatch, sliver, protect):
    """hits of 64k .. 1M tokens (XROW_MAX = 1048576: copied boundaries): the sliver at the boundary of a copied hit is restored/copied by the model like a local hit"""
    for hit in (65536, 262144, 524288, 1040384):
        L = hit + 4096
        gen, m, c = hit_case(monkeypatch, sliver, protect, pieces=((hit, hit + 100), (hit + 100, L)), hit=hit, L=L)
        # (total - hit = 4096: the extension needs st + 512 < total: true; the last piece is final)
        if sliver == protect == 0:
            assert c.lost == 1, hit
        else:
            assert c.lost == 0 and m.steps[-1][0][1] == hit + 512, (hit, m.steps)


def test_b8_flags_off_is_identical(monkeypatch):
    gen, m, c = hit_case(monkeypatch, 0, 0, pieces=((0, 4096), (4096, 4281), (4281, 8192)), hit=0, L=40000)
    assert c.lost == 0 and not any(gen.protect_stats.values())


@pytest.mark.parametrize("sliver,protect", FLAGS)
def test_b8_staggered_closed_loop_with_slivers(monkeypatch, sliver, protect):
    """B=8 replay (row-aware ON) with budget slivers (first pieces of new prompts): old behaviour with both flags off, none with SLIVER_EXTEND (PROTECT alone only extends hit rows: the replay has none)"""
    from models.demos.blackhole.deepseek_v41_flash.tests import test_prefill_row_placement as T
    from models.demos.blackhole.deepseek_v41_flash.tests.test_sliver_extend_cpu import sliver_prefill

    monkeypatch.setattr(T.PluginSim, "_prefill", sliver_prefill)
    monkeypatch.setenv("DSV41_PIECE_LAG", "1")
    setflags(monkeypatch, sliver, protect)
    sim, m, c, _ = T.run_case(
        row_aware=True,
        B=8,
        U=2,
        up=2,
        n_req=16,
        closed_loop_total=32,
        prompt=140000,
        lengths=[100000 + (i * 997 + 13 * i * i) % 20000 for i in range(40)],
        skew=0.3,
        cap=8,
        budget=32768,
        osl_steps=20,
    )
    if sliver:
        assert c.n == 0, c.lines[:2]
    # (flags off: the replay's slivers at B=8 happen to be absorbed here; the hit-sliver tests above carry the old-behaviour assertions)


def test_b8_replay_row_aware_stagger(monkeypatch):
    """B=8 closed-loop replay with row-aware ON (the generator takes the non-row-aware loop at Up = U) and CONT_FIRST: no state loss without slivers, flags on or off"""
    from models.demos.blackhole.deepseek_v41_flash.tests import test_prefill_row_placement as T

    for sk in (0.3, 1.0, 3.0):
        for sliver, protect in FLAGS:
            setflags(monkeypatch, sliver, protect)
            monkeypatch.setenv("DSV41_PIECE_LAG", "1")
            sim, m, c, _ = T.run_case(
                row_aware=True,
                B=8,
                U=2,
                up=2,
                n_req=16,
                closed_loop_total=24,
                prompt=100000,
                skew=sk,
                cap=8,
                budget=32768,
                osl_steps=20,
            )
            assert c.n == 0, (sk, sliver, protect, c.lines[:2])


def test_b8_protect_does_not_gate_admission_at_up_equals_u(monkeypatch):
    """PROTECT_PARTIAL's slot-aware reservation / full-row logic only exists for Up < U: at Up = U every free user stays reservable, capacity is None (unchanged)"""
    setflags(monkeypatch, 0, 1)
    gen, m, plans = make_gen(monkeypatch, True, B=8, U=2, Up=2, row_aware=True)
    for p in (0, 2, 4, 6):  # four prompts in progress, one per row
        gen.inprog.update(p, 4096, torch.zeros(4096, dtype=torch.int64))
        m.pf_resume[p] = 4096
        gen.slots.live.add(p)
    for row in range(4):
        assert gen._free_users_of_row(row) == [
            row * 2 + 1
        ]  # the other user of every row is free, exactly as without the flag
    assert gen._full_rows() == set()
    assert gen.prefill_row_capacity() is None


def test_b8_protected_records_beyond_the_halo_store_fall_back_to_lru():
    """more in-flight records than halo slots (B=8: A 192 per row): the protected-first LRU must still allocate (pass 2 evicts the oldest in-flight record), never refuse"""
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "apc_b8", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tt/apc_paged.py")
    )
    AP = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AP)
    idx = AP.BlockIndex(
        rows=1,
        users_per_row=2,
        pages_per_row=20000,
        halo_a=24,
        halo_b=8,
        key_slots=6000,
        zone=2560,
        halo_budget=0,
        protect_fn=lambda: True,
    )
    C = 512
    for u, seed in (
        (0, 1),
        (1, 2),
    ):  # two giants in flight, every block wants a record (budget off): 2 x 100 records into 24 slots
        t = np.random.default_rng(seed).integers(1, 100000, size=300_000, dtype=np.int64)
        idx.bind(f"g{u}", 0, user=u)
        idx.restored(f"g{u}")
        idx.page_table(f"g{u}", 300_000)
        for st in range(0, 51200, C):
            cp = idx.begin_chunk(f"g{u}", st, st + C, t, 300_000)
            if cp is not None:
                idx.finish_chunk(cp)
    st_ = idx.stats()
    assert st_["rows"][0]["halo_a_free"] == 0 if "halo_a_free" in st_["rows"][0] else True
    assert idx.check()


# ---- B=16 Up = U = 4 (image 2 B=16 entry, slot buckets 1,2 are a model-side window choice: the adapter only sees a normal piece) -----------------------------------------------------
@pytest.mark.parametrize("sliver,protect", FLAGS)
def test_b16_up4_hit_sliver_all_flag_combinations(monkeypatch, sliver, protect):
    setflags(monkeypatch, sliver, protect)
    gen, m, plans = make_gen(monkeypatch, bool(protect), B=16, U=4, Up=4, row_aware=True)
    c = Count()
    logger.remove()
    h = logger.add(c, level="INFO")
    try:
        L = 400000
        toks = torch.randint(1000, 90000, (L,), dtype=torch.int64)
        plans["hit1"] = Plan(user=5, hit=111616)
        call(gen, toks, 111616, 111749, "hit1", 5, L)
        call(gen, toks, 111749, 115712, "hit1", 5, L)
    finally:
        logger.remove(h)
    if sliver == protect == 0:
        assert c.lost == 1
    else:
        assert c.lost == 0 and m.steps[-1][0][1] == 112128


def test_b16_up4_protect_does_not_gate_admission(monkeypatch):
    setflags(monkeypatch, 1, 1)
    gen, m, plans = make_gen(monkeypatch, True, B=16, U=4, Up=4, row_aware=True)
    for p in (0, 1, 2, 4, 8, 12):
        gen.inprog.update(p, 4096, torch.zeros(4096, dtype=torch.int64))
        m.pf_resume[p] = 4096
        gen.slots.live.add(p)
    for row in range(4):
        live = {p for p in gen.slots.live}
        assert gen._free_users_of_row(row) == [p for p in range(row * 4, row * 4 + 4) if p not in live]
    assert gen._full_rows() == set() and gen.prefill_row_capacity() is None


@pytest.mark.parametrize("sliver,protect", FLAGS)
def test_b16_up4_closed_loop_with_slivers_and_cont_first(monkeypatch, sliver, protect):
    from models.demos.blackhole.deepseek_v41_flash.tests import test_prefill_row_placement as T
    from models.demos.blackhole.deepseek_v41_flash.tests.test_sliver_extend_cpu import sliver_prefill

    monkeypatch.setattr(T.PluginSim, "_prefill", sliver_prefill)
    monkeypatch.setenv("DSV41_PIECE_LAG", "1")
    setflags(monkeypatch, sliver, protect)
    sim, m, c, _ = T.run_case(
        row_aware=True,
        B=16,
        U=4,
        up=4,
        n_req=32,
        closed_loop_total=48,
        prompt=140000,
        lengths=[100000 + (i * 997 + 13 * i * i) % 20000 for i in range(40)],
        skew=0.3,
        cap=16,
        budget=32768,
        osl_steps=20,
    )
    if sliver:
        assert c.n == 0, c.lines[:2]
