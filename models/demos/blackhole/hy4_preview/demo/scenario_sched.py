# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler / serving-feature scenarios of the Hy4 demo (program hy4sched, 2026-11-23). Registered into ``demo.scenarios.SCENARIOS`` (names ``sched_*``). Same contract as scenarios.py:
drives ``Hy4Engine`` in process, no vLLM; every gate is a ``DEMO_GATE`` line; a missing engine feature is a SKIP, never a PASS. No ttnn / torch import at module level (CPU-testable with a fake engine,
tests/test_hy4_sched_cpu.py).

* sched_xrow5   E6, the whole-server cross-row copy gate: 5 sessions on 4 rows, the returning turn's chain sits on a LIVE row. Gates: planner picks the free row and one copy happens, copy ms per 100k tokens (printed
                next to the DeepSeek d2d figure 183 ms / 100k [M, DS runs, RESULT_DSB4B]), md5 and stream == cold, the live neighbours (rows 0, 2, 3) are bit-identical to a run without the copy, source chain intact.
* sched_cached  usage cached_tokens: the schedule-time peek (hook.note_scheduled) equals the engine's actual last_cached, the fingerprint claim returns it, a cold prompt reports 0.
* sched_victim  victim by cost + tier counters: a cold prompt overwrites the cheapest chain (shortest), the long chain survives and later hits == cold; pfx_tiers counters add up.
Size knobs: DEMO_XROW_N (prompt tokens of the shared chain, default 8192; 100000 for the per-100k figure).
"""
import os
import tempfile

DS_D2D_MS_PER_100K = 183.0  # DeepSeek d2d cross-row copy [M] (program/RESULT_DSB4B.md, DeepSeek B=4 runs)


def _S():
    from models.demos.blackhole.hy4_preview.demo import scenarios as S

    return S


def _switch(name, on):
    """live-switch file of a hy4sched feature (tt/hy4_sched_flags.py) in a temp file, returns nothing; cache reset so the change is seen at once."""
    from models.demos.blackhole.hy4_preview.tt import hy4_sched_flags as SF

    env = SF.FEATURES[name][0] + "_FILE"
    f = os.environ.get(env)
    if not f:
        f = tempfile.NamedTemporaryFile("w", suffix="." + name, delete=False).name
        os.environ[env] = f
    with open(f, "w") as fh:
        fh.write("1\n" if on else "0\n")
    SF.reset_cache()


def sched_xrow5(ctx, n=None, tail=1024, rounds=8, seed=11):
    S = _S()
    s = "sched_xrow5"
    eng = ctx.eng
    if not getattr(eng, "xrow_on", False) or getattr(eng, "xrow", None) is None:
        return ctx.gate(s, "xrow_enabled", None, "HY4_XROW=1 not set")
    n = int(os.environ.get("DEMO_XROW_N", "8192")) if n is None else n
    A = S.prompt(seed, n)
    Bp = A + S.prompt(seed + 1, tail)  # the returning turn (same prefix, new tail)
    others = {1: S.prompt(seed + 2, n // 2), 2: S.prompt(seed + 3, 600), 3: S.prompt(seed + 4, 600)}

    def phase(copy):
        """session 0 (chain A) and 2, 3 decode; session 1 finishes (row 1 free); the returning turn arrives; 4 sessions + the 5th (returning) = 5 sessions over 4 rows. -> streams, md5, stats"""
        S.release_all(eng)
        S.set_pfx(ctx, False)  # setup prefills are cold in both phases (bookkeeping still retains the chains)
        first = {}
        for r, p in [(0, A)] + sorted(others.items()):
            f, _, _, _ = S.prefill(ctx, {r: p})
            first.update(f)
        pre, _, _ = S.decode(ctx, first, 4, rows=[0, 2, 3])
        eng.release(1)  # session 1 done: its chain stays on row 1 until overwritten
        S.set_pfx(ctx, bool(copy))
        want = eng.assign_rows([Bp], [1]) if copy else [1]
        plan = dict(getattr(eng, "xrow_plan", {}) or {})
        st0 = dict(eng.xrow_stats)
        f1, m1, dt, cached = S.prefill(ctx, {want[0]: Bp})
        st1 = dict(eng.xrow_stats)
        streams, _, _ = S.decode(
            ctx, {**{r: pre[r][-1] for r in (0, 2, 3)}, want[0]: f1[want[0]]}, rounds, rows=[0, 2, 3, want[0]]
        )
        return dict(want=want, plan=plan, md5=m1.get(want[0]), streams=streams, st0=st0, st1=st1, dt=dt, cached=cached)

    ref = phase(False)
    cp = phase(True)
    row = cp["want"][0]
    ctx.gate(s, "planner_picks_free_row_1", cp["want"] == [1] and bool(cp["plan"]), f"{cp['want']} plan={cp['plan']}")
    d = {k: cp["st1"][k] - cp["st0"][k] for k in ("copies", "failed", "tokens", "seconds")}
    ctx.gate(s, "one_copy_no_failure", d["copies"] == 1 and d["failed"] == 0, f"{d}")
    if d["copies"] and d["tokens"]:
        ms100k = d["seconds"] * 1e3 / d["tokens"] * 1e5
        ctx.perf.append(
            dict(
                kind="sched_xrow5",
                tokens=d["tokens"],
                copy_ms=d["seconds"] * 1e3,
                ms_per_100k=ms100k,
                ds_ms_per_100k=DS_D2D_MS_PER_100K,
            )
        )
        ctx.gate(
            s,
            "copy_ms_per_100k_vs_ds",
            True,
            f"{ms100k:.0f} ms per 100k tokens [measured here, {d['tokens']} tokens] vs DeepSeek {DS_D2D_MS_PER_100K:.0f}",
        )
    ctx.gate(s, "copy_hit_served", cp["cached"].get(row, 0) >= n - 512, f"last_cached={cp['cached']}")
    ctx.gate(s, "md5_equals_cold", cp["md5"] is not None and cp["md5"] == ref["md5"], f"{cp['md5']} vs {ref['md5']}")
    ctx.gate(
        s,
        "returning_stream_equals_cold",
        cp["streams"].get(row) == ref["streams"].get(row),
        S.md5_ints(cp["streams"].get(row, [])),
    )
    same = all(cp["streams"].get(r) == ref["streams"].get(r) for r in (0, 2, 3))
    ctx.gate(s, "live_neighbours_unaffected", same, "rows 0,2,3 streams equal to the run without the copy")
    ctx.gate(
        s,
        "source_chain_intact",
        eng.ret[0] is not None and len(eng.ret[0]) >= n - 512,
        f"len={None if eng.ret[0] is None else len(eng.ret[0])}",
    )
    S.set_pfx(ctx, True)


def sched_cached(ctx, n=4096, extra=1024, seed=12):
    S = _S()
    s = "sched_cached"
    eng = ctx.eng
    if not getattr(eng, "pfx", False):
        return ctx.gate(s, "pfx_enabled", None, "HY4_PFX=1 not set")
    from models.demos.blackhole.hy4_preview.tt.hy4_apc_hook import Hy4ApcHook

    hook = Hy4ApcHook(eng, 4)
    S.release_all(eng)
    S.set_pfx(ctx, True)
    A = S.prompt(seed, n)
    Bp = A + S.prompt(seed + 1, extra)
    hook.note_scheduled("cold", A)
    f, _, _, c0 = S.prefill(ctx, {0: A})
    hook.note_prefill_fp([A], [0], [c0.get(0, 0)])
    got, src = hook.claim("cold", A)
    ctx.gate(s, "cold_reports_zero", got == 0 and c0.get(0, 0) == 0, f"claim={got} ({src}) last_cached={c0.get(0, 0)}")
    S.decode(ctx, f, 2, rows=[0])
    eng.release(0)
    _fp, pred = hook.note_scheduled("turn2", Bp)
    f, _, _, c1 = S.prefill(ctx, {0: Bp})
    hook.note_prefill_fp([Bp], [0], [c1.get(0, 0)])
    got, src = hook.claim("turn2", Bp)
    ctx.gate(
        s,
        "claim_equals_engine_actual",
        got == c1.get(0, 0) and src == "actual" and got > 0,
        f"claim={got} ({src}) last_cached={c1.get(0, 0)} of {len(Bp)}",
    )
    ctx.gate(s, "schedule_time_peek_equals_actual", pred == c1.get(0, 0), f"peek={pred} actual={c1.get(0, 0)}")
    S.decode(ctx, f, 2, rows=[0])
    eng.release(0)


def sched_victim(ctx, n=8192, seed=13):
    S = _S()
    s = "sched_victim"
    eng = ctx.eng
    if not getattr(eng, "pfx", False):
        return ctx.gate(s, "pfx_enabled", None, "HY4_PFX=1 not set")
    S.release_all(eng)
    S.set_pfx(ctx, False)  # chains are laid down cold
    _switch("evict_cost", True)
    _switch("tier_log", True)
    L = {0: S.prompt(seed, n), 1: S.prompt(seed + 1, n // 4), 2: S.prompt(seed + 2, n // 2)}
    first = {}
    for r, p in L.items():
        f, _, _, _ = S.prefill(ctx, {r: p})
        first.update(f)
    S.prefill(ctx, {3: S.prompt(seed + 3, 600)})  # row 3 busy
    for r in (0, 1, 2):
        eng.release(
            r
        )  # all three free; row 0 holds the LONGEST chain and is the OLDEST (an LRU policy would overwrite it)
    S.set_pfx(ctx, True)
    C = S.prompt(seed + 4, 1024)
    want = eng.assign_rows([C], [0])
    ctx.gate(
        s,
        "cost_victim_is_shortest_chain",
        want == [1],
        f"want={want} chain lengths {[None if eng.ret[r] is None else len(eng.ret[r]) for r in range(4)]}",
    )
    f, _, _, _ = S.prefill(ctx, {want[0]: C})
    eng.release(want[0])
    ctx.gate(
        s,
        "long_chain_survives",
        eng.ret[0] is not None and len(eng.ret[0]) >= n - 512,
        f"len0={None if eng.ret[0] is None else len(eng.ret[0])}",
    )
    Bp = L[0] + S.prompt(seed + 5, 1024)
    f1, m1, _, c1 = S.prefill(ctx, {0: Bp})
    st1, _, _ = S.decode(ctx, f1, 8, rows=[0])
    eng.release(0)
    S.set_pfx(ctx, False)
    f2, m2, _, c2 = S.prefill(ctx, {0: Bp})
    st2, _, _ = S.decode(ctx, f2, 8, rows=[0])
    eng.release(0)
    S.set_pfx(ctx, True)
    ctx.gate(s, "hit_on_surviving_chain", c1.get(0, 0) >= n - 512, f"last_cached={c1.get(0, 0)}")
    ctx.gate(s, "hit_md5_equals_cold", m1.get(0) is not None and m1.get(0) == m2.get(0), f"{m1.get(0)} vs {m2.get(0)}")
    ctx.gate(s, "hit_stream_equals_cold", st1[0] == st2[0], S.md5_ints(st1[0]))
    t = getattr(eng, "tier_stats", {})
    ctx.gate(s, "tier_counters", t.get("row_tokens", 0) >= n - 512 and t.get("cold_reqs", 0) >= 1, f"{t}")
    _switch("evict_cost", False)
    _switch("tier_log", False)


SCENARIOS = {"sched_xrow5": sched_xrow5, "sched_cached": sched_cached, "sched_victim": sched_victim}
