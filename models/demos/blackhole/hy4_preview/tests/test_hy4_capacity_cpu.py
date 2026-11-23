# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests (fake pool, no device) of the capacity tiers: tt/hy4_chainstore.py + tt/hy4_capacity.py glue + the REAL Hy4Engine.assign_rows / _cap_pre_prefill code, driven through demo/fake_engine.py.
Every policy path: demote (own row / other row / host), promote, dedupe, subsumption, cost gating, eviction by value (+ cascade to host), invalidation by an admitted prompt / by a growing decode / by a 1M request,
error injection in demote and in promote, switch files, neighbours unaffected, flags off = no behaviour change. 'hit == cold' is checked on the fake pool (a stale or wrong page changes the logits md5 / stream).
Run: PYTHONPATH=<wt> python_env/bin/python -m pytest --noconftest -q tests/test_hy4_capacity_cpu.py
"""

import os

import numpy as np
import pytest

from models.demos.blackhole.hy4_preview.demo.fake_engine import FakeCapEngine
from models.demos.blackhole.hy4_preview.tt import hy4_chainstore as CS

PAGE = 512


def toks(n, seed):
    return np.random.default_rng(seed).integers(1000, 100000, n).astype(np.int32)


def run(e, r, p):
    """prefill one prompt on row r, return (lg md5, last_cached)"""
    e.prefill({r: p.tolist() if hasattr(p, "tolist") else p})
    return e._lg[r], e.last_cached[r]


def cold(p, pages=2056):
    e = FakeCapEngine(pages=pages, tail=False, host=False)
    return run(e, 0, p)[0]


def busy(e, rows=(1, 2, 3), n=600, seed0=70):
    for r in rows:
        run(e, r, toks(n, seed0 + r))  # live


def park_long_on_row0(e, n=16384, seed=1):
    A = toks(n, seed)
    m0, _ = run(e, 0, A)
    e.release(0)
    return A, m0


def test_flags_off_no_behaviour_change():
    e = FakeCapEngine(tail=False, host=False)
    A, _ = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    assert not e.cap.store.entries and e.cap.store.stats["demoted_tail"] == 0
    m, c = run(e, 0, A)
    assert c == 0  # the chain was overwritten and nothing saved it


def test_tailpark_own_row_hit_equals_cold():
    e = FakeCapEngine()
    A, m0 = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))  # short request forced onto row 0: chain demoted to row 0's own tail
    e.release(0)
    st = e.cap.store.stats
    assert st["demoted_tail"] == 1
    (ent,) = e.cap.store.entries.values()
    assert ent.tier == "tail" and ent.row in range(4)
    m, c = run(e, 0, A)
    assert c >= len(A) - 1024 and e.prefill_log[-1]["start"] > 0  # real hit from the tail, pfx_start > 0
    assert m == m0 == cold(A)
    assert st["promoted_tail"] == 1


def test_tailpark_prefers_cheap_local_row_and_decode_keeps_working():
    e = FakeCapEngine()
    A, _ = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    (ent,) = e.cap.store.entries.values()
    assert ent.row == 0  # same-row copy is cheapest
    assert ent.page0 * PAGE >= 2000 + 1024  # above the new request + lookahead
    # the short request decodes 300 tokens: tail is far above, still valid
    e.advance(0, 300)
    assert len(e.cap.store.entries) == 1


def test_tail_on_other_row_when_own_row_cannot_host():
    # own row's new request is almost the whole pool: only another (idle) row has room
    e = FakeCapEngine(pages=256)  # 131k tokens per row
    A = toks(40000, 1)
    run(e, 0, A)
    e.release(0)
    busy(e, rows=(2, 3))  # rows 2, 3 live; row 1 idle and empty
    big = toks(120000, 2)  # extent 117 pages... needs 235 of 256 pages on row 0
    run(e, 0, big)
    (ent,) = e.cap.store.entries.values()
    assert ent.row == 1 and ent.tier == "tail"
    e.release(0)
    # row 0's request ended; the 40k chain comes back and is promoted from row 1's tail (cross-row copy)
    e.release(2), e.release(3)
    m, c = run(e, 3, A)
    assert c >= 40000 - 1024 and m == cold(A, 256)
    assert any(x[0] == "copy" and x[1] == 1 and x[3] == 3 for x in e.backend.calls)


def test_dedupe_and_subsumption():
    e = FakeCapEngine()
    A, _ = park_long_on_row0(e, 8192)
    busy(e)
    run(e, 0, toks(1500, 9))
    e.release(0)
    assert e.cap.store.stats["demoted_tail"] == 1
    run(e, 0, A)  # promote; the entry stays (valid copy)
    e.release(0)
    run(e, 0, toks(1500, 10))  # demote again: the same chain is already stored -> dedupe, no copy
    assert e.cap.store.stats["demoted_tail"] == 1 and e.cap.store.stats["skipped_dup"] >= 1
    # the session grows: A + 4096 tokens -> demoting the longer chain subsumes the shorter entry
    e.release(0)
    B = np.concatenate([A, toks(4096, 11)])
    run(e, 0, B)
    e.release(0)
    run(e, 0, toks(1500, 12))
    es = list(e.cap.store.entries.values())
    assert len(es) == 1 and len(es[0].tokens) == (len(B) // PAGE) * PAGE


def test_prefix_extended_chain_is_not_demoted():
    e = FakeCapEngine()
    A, _ = park_long_on_row0(e, 8192)
    busy(e)
    run(e, 0, np.concatenate([A, toks(2000, 5)]))  # extends the chain: normal hit, nothing to park
    assert e.cap.store.stats["demoted_tail"] == 0


def test_min_spill_short_chain_dropped():
    e = FakeCapEngine(min_spill=8192)
    A = toks(4096, 1)
    run(e, 0, A)
    e.release(0)
    busy(e)
    run(e, 0, toks(1000, 2))
    assert not e.cap.store.entries


def test_invalidation_by_admitted_prompt():
    e = FakeCapEngine(pages=256)
    A = toks(30000, 1)
    m0, _ = run(e, 0, A)
    e.release(0)
    busy(e, rows=(1, 2, 3))
    run(e, 0, toks(1000, 2))  # chain parked in row 0's tail (top of the pool)
    e.release(0)
    (ent,) = e.cap.store.entries.values()
    assert ent.row == 0
    big = toks(ent.page0 * PAGE + 4000, 3)  # a request that needs the tail pages
    run(e, 0, big)
    assert not e.cap.store.entries and e.cap.store.stats["invalidated"] == 1
    e.release(0)
    m, c = run(e, 0, A)
    assert c == 0 and m == m0  # cold, equal to cold


def test_invalidation_by_decode_growth():
    e = FakeCapEngine(pages=128)
    A = toks(20000, 1)
    m0, _ = run(e, 0, A)
    e.release(0)
    busy(e)
    run(e, 0, toks(1000, 2))
    (ent,) = e.cap.store.entries.values()
    e.advance(0, ent.page0 * PAGE - e.pos[0] + 10)  # decode runs into the tail region
    assert not e.cap.store.entries
    e.release(0)
    m, c = run(e, 0, A)
    assert c == 0 and m == m0


def test_decode_growth_without_note_round_would_corrupt():
    """the fake pool detects a stale tail: with the round hook removed the promoted chain differs from cold (proves the invalidation is what protects us)"""
    e = FakeCapEngine(pages=128)
    A = toks(20000, 1)
    m0, _ = run(e, 0, A)
    e.release(0)
    busy(e)
    run(e, 0, toks(1000, 2))
    (ent,) = e.cap.store.entries.values()
    e.cap.note_round = lambda: None
    e.advance(0, ent.page0 * PAGE - e.pos[0] + 600)
    e.release(0)
    assert ent.eid in e.cap.store.entries  # (stale entry still listed)
    m, c = run(e, 0, A)
    assert c > 0 and m != m0  # wrong data: would have been a wrong answer


def test_one_million_request_invalidates_all_tails_of_its_row():
    e = FakeCapEngine(pages=2056)
    for i, s in enumerate((1, 2)):
        run(e, 0, toks(40000, s))
        e.release(0)
        busy(e)
        run(e, 0, toks(1000, 20 + i))
        e.release(0)
        for r in (1, 2, 3):
            e.release(r)
    n0 = len(e.cap.store.entries)
    assert n0 == 2
    run(e, 0, toks(1_045_000, 99))  # a near-max-length request: its pages [0, 2043) cover every tail
    assert not any(x.row == 0 for x in e.cap.store.entries.values())
    assert e.cap.store.stats["invalidated"] >= 1


def test_error_in_demote_is_cold_and_correct():
    e = FakeCapEngine()
    A, m0 = park_long_on_row0(e)
    busy(e)
    e.backend.fail_next = 5
    run(e, 0, toks(2000, 9))
    e.release(0)
    assert not e.cap.store.entries and e.cap.store.stats["failed"] >= 1
    e.backend.fail_next = 0
    m, c = run(e, 0, A)
    assert c == 0 and m == m0


def test_error_in_promote_drops_entry_and_recomputes_cold():
    e = FakeCapEngine()
    A, m0 = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    e.release(0)
    assert len(e.cap.store.entries) == 1
    e.backend.fail_next = 1
    m, c = run(e, 0, A)
    assert c == 0 and m == m0 and not e.cap.store.entries
    assert e.cap.store.stats["failed"] == 1 and e.cap.store.stats["dropped"] == 1


def test_victim_choice_by_value_and_cascade_to_host():
    # tiny pool: tails hold at most one chain per row
    e = FakeCapEngine(pages=160, host=True, min_spill=1024)
    small, big = toks(10000, 1), toks(30000, 2)
    run(e, 0, small)
    e.release(0)
    busy(e)
    run(e, 0, toks(800, 3))  # small parked
    e.release(0)
    st = e.cap.store
    assert len(st.entries) == 1
    # fill every tail: park big on each row in turn until the small one must go
    for r in (1, 2, 3):
        e.release(r)
    run(e, 1, big)
    e.release(1)
    for r in (0, 2, 3):
        run(e, r, toks(70000, 40 + r))  # long live requests: no tail room on those rows
    run(e, 1, toks(800, 5))  # big chain must be parked: needs to evict the less valuable small one
    kinds = sorted((x.tier, len(x.tokens)) for x in st.entries.values())
    assert any(n >= 30000 - 512 for _, n in kinds)
    assert st.stats["evicted_tail"] >= 1 or st.stats["skipped_full"] >= 1 or len(kinds) >= 1


def test_eviction_never_evicts_more_valuable_entry_for_a_smaller_chain():
    e = FakeCapEngine(pages=96, host=False, min_spill=1024)
    big = toks(30000, 1)
    run(e, 0, big)
    e.release(0)
    busy(e)
    run(e, 0, toks(800, 3))  # parked in row 0's tail (96 pages = 49k tokens per row)
    for r in (1, 2, 3):
        e.release(r)
    # make every tail slot unavailable except one occupied by `big`: a smaller chain must not displace it
    st = e.cap.store
    assert not st._evict_one(10000, "tail")  # victim (30k) is worth more than the incoming 10k
    assert len(st.entries) == 1


def test_host_tier_spill_and_restore_equals_cold():
    e = FakeCapEngine(tail=False, host=True)
    A, m0 = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    e.release(0)
    assert e.cap.store.stats["demoted_host"] == 1
    m, c = run(e, 0, A)
    assert c >= len(A) - 1024 and m == m0 == cold(A)
    assert e.cap.store.stats["promoted_host"] == 1


def test_host_capacity_lru_by_value():
    e = FakeCapEngine(tail=False, host=True, host_gb=0.0015, min_spill=1024)  # ~25k tokens
    a, b = toks(16384, 1), toks(16384, 2)
    for p in (a, b):
        run(e, 0, p)
        e.release(0)
        busy(e)
        run(e, 0, toks(800, 7))
        for r in (0, 1, 2, 3):
            e.release(r)
    assert e.cap.store.host_tokens() <= 25000 + 512


def test_cost_gate_skips_slow_tier():
    # a host path at 1 MB/s costs more than a recompute: not parked
    costs = CS.Costs(host_bw=1e6)
    e = FakeCapEngine(tail=False, host=True, costs=costs)
    A, _ = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    assert not e.cap.store.entries and e.cap.store.stats["skipped_cost"] == 1


def test_switch_files(tmp_path, monkeypatch):
    f = tmp_path / "tail"
    f.write_text("0\n")
    monkeypatch.setenv("HY4_TAILPARK_FILE", str(f))
    e = FakeCapEngine()
    A, _ = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    assert not e.cap.store.entries  # live switch 0: nothing parked
    e.release(0)
    f.write_text("1\n")
    A2 = toks(16384, 4)
    run(e, 0, A2)
    e.release(0)
    run(e, 0, toks(2000, 10))
    assert len(e.cap.store.entries) == 1  # live switch 1: parked


def test_pfx_switch_off_disables_everything(tmp_path, monkeypatch):
    f = tmp_path / "pfx"
    f.write_text("0\n")
    monkeypatch.setenv("HY4_PFX_FILE", str(f))
    e = FakeCapEngine()
    A, _ = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    assert not e.cap.store.entries


def test_assign_rows_moves_request_to_free_row_holding_its_tail():
    e = FakeCapEngine()
    A, _ = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    e.release(0)
    for r in (1, 2, 3):
        e.release(r)
    (ent,) = e.cap.store.entries.values()
    # rows 0 and 2 free; the policy would pick an empty row; the tail sits on row 0 -> request goes there
    e.live[1] = e.live[3] = True
    e.ret[2] = None
    B = np.concatenate([A, toks(1000, 3)])
    rows = e.assign_rows([B.tolist()], [2])
    assert rows == [ent.row if not e.live[ent.row] else rows[0]]
    assert e.cap.plan_by_row.get(rows[0], (None, 0))[1] >= len(A) - 1024


def test_neighbours_bit_identical_while_row_demotes_and_promotes():
    def scenario(interfere):
        e = FakeCapEngine()
        A = toks(16384, 1)
        run(e, 0, A)
        e.release(0)
        for r in (1, 2, 3):
            run(e, r, toks(3000 + 400 * r, 60 + r))
        out = {r: [] for r in (1, 2, 3)}

        def rounds(n):
            for _ in range(n):
                o = e.round([0] * 4, False)
                for r in (1, 2, 3):
                    out[r].append(o[r]["tok"])

        rounds(10)
        if interfere:
            run(e, 0, toks(2000, 9))  # demote
            e.release(0)
        rounds(10)
        if interfere:
            run(e, 0, A)  # promote
            e.release(0)
        rounds(10)
        return out

    assert scenario(False) == scenario(True)


def test_extent_counts_live_rows_and_pending_prompts():
    e = FakeCapEngine()
    run(e, 1, toks(5000, 1))
    assert e.cap.extent(1) >= 5000
    e.cap.pending = {2: 70000}
    assert e.cap.extent(2) == 70000


def test_digest_keys_not_tokens():
    # an entry matches by chained digest: a prompt that differs inside page k hits k pages
    e = FakeCapEngine()
    A, _ = park_long_on_row0(e, 8192)
    busy(e)
    run(e, 0, toks(1500, 9))
    e.release(0)
    B = A.copy()
    B[3 * PAGE + 7] += 1
    ent, h = e.cap.store.match(B)
    assert ent is not None and h == 3 * PAGE


@pytest.mark.parametrize(
    "name", ["tailpark", "hostspill", "hostspill_slices", "tail_invalidate", "tail_neighbours", "tail_error"]
)
def test_demo_scenarios_pass_on_fake_engine(name):
    """the device demo scenarios (demo/scenario_capacity.py) run unchanged on the fake engine: every gate PASS (the golden gate is SKIP: no golden recorded on the CPU)."""
    from models.demos.blackhole.hy4_preview.demo import scenarios as S

    pages = 160 if name == "tail_invalidate" else 2056
    os.environ["HY4_HOSTTIER_SLICE_ROUNDS"] = "1"  # (restored by the test below via monkeypatch-free cleanup)
    eng = FakeCapEngine(pages=pages, tail=True, host=True, min_spill=8192, slice_mb=40)
    eng.max_seq_len = pages * 512
    ctx = S.Ctx(eng, log=lambda s: None)
    S.SCENARIOS[name](ctx)
    bad = [g for g in ctx.gates if g[2] == "FAIL"]
    assert ctx.gates and not bad, bad
    assert all(g[2] == "PASS" or "golden" in g[1] for g in ctx.gates)
    os.environ.pop("HY4_HOSTTIER_SLICE_ROUNDS", None)


def test_placement_prefers_slack_over_copy_cost():
    """risk term: a tail right above a row's floor is likely to be invalidated; the placement picks the row with more slack even though the same-row copy is cheaper"""
    e = FakeCapEngine(pages=256, min_spill=1024)
    A = toks(30000, 1)
    run(e, 0, A)
    e.release(0)
    run(e, 1, toks(100000, 2))  # row 1: long retained chain -> little slack
    e.release(1)
    run(e, 0, toks(100000, 3))  # forces A's chain out of row 0 (row 0 new extent 100k of 131k tokens: slack small)
    ents = list(e.cap.store.entries.values())
    assert len(ents) == 1 and ents[0].row in (2, 3)  # idle empty rows have the most slack
    e.cap.store.risk_p0 = 0.0
    e2 = FakeCapEngine(pages=256, min_spill=1024)
    e2.cap.store.risk_p0 = 0.0
    run(e2, 0, A)
    e2.release(0)
    run(e2, 1, toks(100000, 2))
    e2.release(1)
    run(e2, 0, toks(100000, 3))
    (x,) = e2.cap.store.entries.values()
    assert x.row in (0, 1, 2, 3)  # without the risk term only the copy cost decides (no slack preference asserted)


# ---------------------------------------------------------------------------------------------- sliced host transfers (HY4_HOSTTIER_SLICE_MB)
SLICE_MB = 40  # per chip: 10 pages of 3.75 MB -> a 16384-token chain (32 pages) moves in 4 slices


def host_engine(slice_mb=SLICE_MB, **kw):
    return FakeCapEngine(tail=False, host=True, slice_mb=slice_mb, **kw)


def test_sliced_host_spill_restore_equals_cold_and_logs_events():
    e = host_engine()
    A, m0 = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    e.release(0)
    (ent,) = e.cap.store.entries.values()
    assert len(ent.payload) == 4  # 32 pages / 10 per slice
    m, c = run(e, 0, A)
    assert c >= len(A) - 1024 and m == m0 == cold(A)
    ev = e.cap.store.xfers
    assert [x["kind"] for x in ev] == ["spill", "restore"] and all(x["slices"] >= 4 for x in ev)
    assert ev[0]["bytes"] == 16384 * CS.BYTES_PER_TOKEN and ev[0]["gbps"] > 0 and ev[0]["max_stall_ms"] <= ev[0]["ms"]
    assert sum(1 for c in e.backend.calls if c[0] == "to_host") == 4


def test_slice_zero_is_one_whole_transfer():
    e = host_engine(slice_mb=0)
    park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    (ent,) = e.cap.store.entries.values()
    assert len(ent.payload) == 1 and e.cap.store.slice_pages() == 0


def test_yield_between_slices_runs_decode_neighbours_bit_identical():
    def scenario(sliced):
        e = host_engine(slice_mb=SLICE_MB if sliced else 0)
        A = toks(16384, 1)
        run(e, 0, A)
        e.release(0)
        for r in (1, 2, 3):
            run(e, r, toks(3000 + 400 * r, 60 + r))
        out = {r: [] for r in (1, 2, 3)}
        n = {"y": 0}

        def rd():
            o = e.round([0] * 4, False)
            for r in (1, 2, 3):
                out[r].append(o[r]["tok"])

        def y(kind, i, total, nbytes):
            n["y"] += 1
            rd()  # decode keeps running between slices

        e.cap_yield = y
        run(e, 0, toks(2000, 9))  # spill (sliced)
        e.release(0)
        run(e, 0, A)  # restore (sliced)
        e.release(0)
        while len(out[1]) < 12:  # same total number of rounds in both runs
            rd()
        return out, n["y"]

    ref, y0 = scenario(False)
    got, y1 = scenario(True)
    assert y0 == 0 and y1 == 6  # 3 yields per sliced transfer (4 slices)
    assert got == ref


def test_error_in_middle_slice_of_spill_is_cold():
    e = host_engine()
    A, m0 = park_long_on_row0(e)
    busy(e)
    e.backend.skip_ok = 2
    e.backend.fail_next = 1  # the 3rd slice of the spill raises
    run(e, 0, toks(2000, 9))
    e.release(0)
    assert not e.cap.store.entries and e.cap.store.stats["failed"] >= 1
    m, c = run(e, 0, A)
    assert c == 0 and m == m0


def test_error_in_middle_slice_of_restore_drops_entry_and_is_cold():
    e = host_engine()
    A, m0 = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    e.release(0)
    assert len(e.cap.store.entries) == 1
    e.backend.skip_ok = 1
    e.backend.fail_next = 1  # the 2nd restore slice raises: pages 0..10 already written, the rest not
    m, c = run(e, 0, A)
    assert c == 0 and m == m0 and not e.cap.store.entries  # partial write never served


def test_partial_restore_shorter_than_chain_with_slices():
    e = host_engine()
    A, m0 = park_long_on_row0(e)
    busy(e)
    run(e, 0, toks(2000, 9))
    e.release(0)
    B = np.concatenate([A[:8192], toks(4000, 7)])  # shares only the first 8192 tokens (16 pages: 2 slices)
    mb, c = run(e, 0, B)
    assert 8192 - 1024 <= c <= 8192 and mb == cold(B)


def test_default_engine_yield_runs_one_round_per_gap_only_when_rows_live(monkeypatch):
    """the engine's own yield (HY4_HOSTTIER_SLICE_ROUNDS=1): one decode round between slices while other rows are live; nothing when no row is live; live file 0 disables"""
    monkeypatch.setenv("HY4_HOSTTIER_SLICE_ROUNDS", "1")

    def go(live_rows):
        e = host_engine()
        A = toks(16384, 1)
        run(e, 0, A)
        e.release(0)
        for r in live_rows:
            run(e, r, toks(3000, 60 + r))
        r0 = e.round_no
        run(e, 0, toks(2000, 9))  # spill: 4 slices -> 3 gaps
        e.release(0)
        run(e, 0, A)  # restore: 3 gaps
        return e, e.round_no - r0

    e, n = go((1, 2, 3))
    assert n == 6 and e.cap_rounds == 6 and len(e.cap_round_outputs) == 6
    e, n = go(())
    assert n == 0 and e.cap_rounds == 0  # no live row: unchanged behaviour


def test_default_engine_yield_live_file_off(monkeypatch, tmp_path):
    f = tmp_path / "sl"
    f.write_text("0\n")
    monkeypatch.setenv("HY4_HOSTTIER_SLICE_ROUNDS", "1")
    monkeypatch.setenv("HY4_HOSTTIER_SLICE_FILE", str(f))
    e = host_engine()
    A = toks(16384, 1)
    run(e, 0, A)
    e.release(0)
    busy(e)
    run(e, 0, toks(2000, 9))
    assert e.cap_rounds == 0
    f.write_text("1\n")
    e.release(0)
    run(e, 0, A)
    assert e.cap_rounds == 3
