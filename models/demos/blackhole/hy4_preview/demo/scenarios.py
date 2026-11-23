# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview in-process DEMO scenarios (step 0 of HY4_PORT_PLAN: demo first, vLLM server last).

Everything here drives ``Hy4Engine`` (tt/hy4_serve_engine.py) DIRECTLY: ``prefill(prompts_by_row)``, ``round(caps, want_sampled)``, ``release(row)``, ``assign_rows``; no vLLM, plugin, scheduler or client.
No ttnn / torch import at module level: the control flow is unit-testable on the CPU with a fake engine (tests/test_hy4_demo_cpu.py). Seeded from product_engine_smoke.py (basic / masked / cap0),
prefix_check4.py (hit vs cold), stash_check.py (stash), b45x_xrow_unit.py (cross-row copy) and the hy4_pfx_loss / tail-park / chunked-prefill designs of program/HY4_PORT_PLAN.md.

Gate = one line ``DEMO_GATE <scenario> <name> PASS|FAIL|SKIP <value>``. A scenario that needs an engine feature that does not exist yet (prefill_begin, tail-park) reports SKIP, never PASS.
Prompts are random token ids (seeded): bit-exact comparisons (logits md5 of the last prompt row, greedy token streams) do not depend on the tokenizer.
"""
import hashlib
import os
import random
import tempfile
import time

R = 4


# ------------------------------------------------------------------------------------------------ plumbing
class Ctx:
    def __init__(self, eng, golden=None, record=False, log=print):
        self.eng, self.golden, self.record, self.log = eng, golden if golden is not None else {}, record, log
        self.gates, self.perf = [], []
        self.md5_calls = []  # one {row: md5(last prompt logits row)} per Hy4PoolPrefill.prefill call
        self.pfx_file = None
        self._tap()

    def _tap(self):
        """record the md5 of the logits row of every prefill call (the engine prints the same value as lg_md5 in its 'HY4 prefill call' line); wraps eng.pp.prefill, no engine change."""
        pp = getattr(self.eng, "pp", None)
        if pp is None or getattr(pp, "_demo_tapped", False):
            return
        orig = pp.prefill
        calls = self.md5_calls

        def wrapped(*a, **k):
            res = orig(*a, **k)
            d = {}
            for r in k.get("keep_rows") or range(R):
                try:
                    d[r] = hashlib.md5(res.logits[r].numpy().tobytes()).hexdigest()[:12]
                except Exception:  # noqa: BLE001
                    pass
            calls.append(d)
            return res

        pp.prefill = wrapped
        pp._demo_tapped = True

    def gate(self, scen, name, ok, value=""):
        st = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.gates.append((scen, name, st, value))
        self.log(f"DEMO_GATE {scen} {name} {st} {value}")
        return ok

    def golden_check(self, scen, key, value):
        """greedy golden: --record stores, otherwise compare when a golden exists (a missing golden is a SKIP, not a pass)."""
        g = self.golden.setdefault(scen, {})
        if self.record:
            g[key] = value
            return self.gate(scen, f"golden:{key}", True, "recorded")
        if key not in g:
            return self.gate(scen, f"golden:{key}", None, "no golden recorded for this config")
        return self.gate(scen, f"golden:{key}", g[key] == value, f"{value} vs {g[key]}")


def prompt(seed, n):
    rnd = random.Random(seed)
    return [rnd.randrange(1000, 100000) for _ in range(n)]


def md5_ints(xs):
    return hashlib.md5(",".join(map(str, xs)).encode()).hexdigest()[:12]


def release_all(eng):
    for r in range(R):
        eng.release(r)


def set_pfx(ctx, on):
    """live prefix-cache switch (engine reads HY4_PFX_FILE per call)."""
    if ctx.pfx_file is None:
        ctx.pfx_file = tempfile.NamedTemporaryFile("w", suffix=".pfx", delete=False).name
        os.environ["HY4_PFX_FILE"] = ctx.pfx_file
    with open(ctx.pfx_file, "w") as f:
        f.write("1\n" if on else "0\n")


def prefill(ctx, prompts_by_row):
    """-> (first tokens {row: tok}, md5 {row: md5 of last-prompt logits row}, seconds, last_cached {row: tokens not recomputed})"""
    n0 = len(ctx.md5_calls)
    t0 = time.perf_counter()
    first = ctx.eng.prefill(prompts_by_row)
    dt = time.perf_counter() - t0
    md5 = {}
    for d in ctx.md5_calls[n0:]:
        md5.update(d)
    return first, md5, dt, dict(getattr(ctx.eng, "last_cached", {}) or {})


def decode(ctx, first, n_rounds, rows=None, caps=None):
    """greedy spec rounds on the engine; rows = rows whose streams are kept. -> (streams {row: [tokens]}, ms per round [float], accepted per round)"""
    eng = ctx.eng
    rows = sorted(first) if rows is None else rows
    st = {r: [first[r]] for r in rows if r in first}
    ms, acc = [], []
    for _ in range(n_rounds):
        c = caps if caps is not None else [eng.K if eng.live[r] else 0 for r in range(R)]
        t0 = time.perf_counter()
        out = eng.round(c, False)
        ms.append((time.perf_counter() - t0) * 1e3)
        for r in rows:
            if r in st:
                em = out[r]["a"][: out[r]["m"] + 1]
                st[r] += em
                acc.append(len(em))
    return st, ms, acc


def common_prefix_equal(a, b):
    n = min(len(a), len(b))
    return n > 0 and a[:n] == b[:n]


# ------------------------------------------------------------------------------------------------ scenarios
def multirow(ctx, lens=(100, 700, 2600, 5000), rounds=24, seed=1):
    """4 prompts of different lengths in one call (lock-step rows), greedy rounds. Golden: logits md5 per row + token-stream md5 per row. Perf: decode ms/round, tokens/round."""
    s = "multirow"
    release_all(ctx.eng)
    P = {r: prompt(seed * 100 + r, lens[r]) for r in range(R)}
    first, md5, dt, _ = prefill(ctx, P)
    ctx.gate(s, "prefill_returns_4_tokens", len(first) == R, f"{dt:.1f}s")
    st, ms, acc = decode(ctx, first, rounds)
    for r in range(R):
        ctx.golden_check(s, f"lg_md5_row{r}", md5.get(r))
        ctx.golden_check(s, f"stream_md5_row{r}", md5_ints(st[r]))
    ctx.perf.append(
        dict(kind="decode", scen=s, users=R, ms_per_round=sum(ms) / len(ms), tok_per_round=sum(acc) / max(len(acc), 1))
    )
    ctx.gate(
        s, "tokens_valid", all(0 <= t < 200000 for r in st for t in st[r]), f"{sum(len(v) for v in st.values())} tokens"
    )
    return st


def masked(ctx, lens=(100, 700, 2600, 5000), rounds=12, mid_row=2, mid_len=900, seed=1):
    """product_engine_smoke 'masked': row mid_row is prefilled anew after 3 rounds while the other rows decode; their streams must equal the uninterrupted run."""
    s = "masked"
    eng = ctx.eng
    release_all(eng)
    P = {r: prompt(seed * 100 + r, lens[r]) for r in range(R)}
    first, _, _, _ = prefill(ctx, P)
    base, _, _ = decode(ctx, first, rounds)
    release_all(eng)
    first, _, _, _ = prefill(ctx, P)
    st, _, _ = decode(ctx, first, 3)
    f2, _, _, _ = prefill(ctx, {mid_row: prompt(7, mid_len)})
    more, _, _ = decode(
        ctx, {r: st[r][-1] for r in first if r != mid_row}, rounds - 3, rows=[r for r in range(R) if r != mid_row]
    )
    for r in range(R):
        if r == mid_row:
            continue
        got = st[r] + more[r][1:]
        ctx.gate(s, f"row{r}_stream_equals_uninterrupted", common_prefix_equal(base[r], got), f"{len(got)} tokens")


def cap0(ctx, lens=(100, 700, 2600, 5000), rounds=12, seed=1):
    """lossless spec: caps 0 (one token per round) stream equals the speculative stream over their common length."""
    s = "cap0"
    eng = ctx.eng
    release_all(eng)
    P = {r: prompt(seed * 100 + r, lens[r]) for r in range(R)}
    first, _, _, _ = prefill(ctx, P)
    a, _, _ = decode(ctx, first, rounds)
    release_all(eng)
    first, _, _, _ = prefill(ctx, P)
    b, _, _ = decode(ctx, first, rounds * 3, caps=[0] * R)
    for r in range(R):
        ctx.gate(s, f"row{r}_cap0_equals_spec", common_prefix_equal(a[r], b[r]), f"{min(len(a[r]), len(b[r]))} tokens")


def pfx_hit_cold(ctx, n=4096, extra=1024, rounds=16, row=0, seed=3):
    """prefix_check4 in-process: prompt cold -> same prompt again (hit) -> hit with a new suffix -> cold with the cache off; logits md5 and 16-round greedy stream must be equal (hit == cold)."""
    s = "pfx_hit_cold"
    eng = ctx.eng
    if not getattr(eng, "pfx", False):
        return ctx.gate(s, "pfx_enabled", None, "HY4_PFX=1 not set")
    release_all(eng)
    A = prompt(seed, n)
    set_pfx(ctx, True)
    f0, m0, t0, c0 = prefill(ctx, {row: A})
    s0, _, _ = decode(ctx, f0, rounds, rows=[row])
    eng.release(row)
    f1, m1, t1, c1 = prefill(ctx, {row: A})
    s1, _, _ = decode(ctx, f1, rounds, rows=[row])
    eng.release(row)
    ctx.gate(s, "real_hit", c1.get(row, 0) > 0, f"last_cached={c1.get(row, 0)} of {n}; cold {t0:.1f}s hit {t1:.1f}s")
    ctx.gate(
        s,
        "hit_md5_equals_cold",
        m1.get(row) == m0.get(row) and m0.get(row) is not None,
        f"{m1.get(row)} vs {m0.get(row)}",
    )
    ctx.gate(s, "hit_stream_equals_cold", s1[row] == s0[row], md5_ints(s1[row]))
    B = A + prompt(seed + 1, extra)  # multi-turn style: same prefix, new tail
    f2, m2, t2, c2 = prefill(ctx, {row: B})
    s2, _, _ = decode(ctx, f2, rounds, rows=[row])
    eng.release(row)
    set_pfx(ctx, False)
    f3, m3, t3, c3 = prefill(ctx, {row: B})
    s3, _, _ = decode(ctx, f3, rounds, rows=[row])
    eng.release(row)
    set_pfx(ctx, True)
    ctx.gate(s, "suffix_hit", c2.get(row, 0) >= (n // 512 - 1) * 512, f"last_cached={c2.get(row, 0)}")
    ctx.gate(
        s,
        "suffix_hit_md5_equals_cold",
        m2.get(row) == m3.get(row) and m3.get(row) is not None,
        f"{m2.get(row)} vs {m3.get(row)}",
    )
    ctx.gate(s, "suffix_hit_stream_equals_cold", s2[row] == s3[row], md5_ints(s2[row]))
    ctx.gate(s, "cold_off_has_no_hit", c3.get(row, 0) == 0, f"last_cached={c3.get(row, 0)}")
    ctx.golden_check(s, "cold_lg_md5", m0.get(row))


def xrow(ctx, n=8192, rounds=16, seed=4):
    """b45x_xrow_unit at engine level: chain A on row 0, row 0 stays LIVE (decoding), a request with A's prefix arrives while only row 1 is free -> device copy row 0 -> 1, md5 equals a cold prefill."""
    s = "xrow"
    eng = ctx.eng
    if not getattr(eng, "xrow_on", False) or getattr(eng, "xrow", None) is None:
        return ctx.gate(s, "xrow_enabled", None, "HY4_XROW=1 not set")
    release_all(eng)
    A = prompt(seed, n)
    B = A + prompt(seed + 1, 1024)
    set_pfx(ctx, True)
    f0, _, _, _ = prefill(ctx, {0: A})
    for r in (2, 3):  # rows 2, 3 busy with their own requests
        prefill(ctx, {r: prompt(50 + r, 600)})
    ctx.eng.live[0] = ctx.eng.live[2] = ctx.eng.live[3] = True
    want = eng.assign_rows([B], [1])
    ctx.gate(s, "planner_picks_free_row_1", want == [1], f"{want}; plan={getattr(eng, 'xrow_plan', None)}")
    c0 = eng.xrow_stats["copies"]
    f1, m1, dt, cached = prefill(ctx, {1: B})
    ctx.gate(
        s, "copy_happened", eng.xrow_stats["copies"] == c0 + 1 and eng.xrow_stats["failed"] == 0, f"{eng.xrow_stats}"
    )
    s1, _, _ = decode(ctx, f1, rounds, rows=[1])
    eng.release(1)
    set_pfx(ctx, False)
    f2, m2, _, _ = prefill(ctx, {1: B})
    s2, _, _ = decode(ctx, f2, rounds, rows=[1])
    set_pfx(ctx, True)
    ctx.gate(s, "copy_md5_equals_cold", m1.get(1) == m2.get(1) and m1.get(1) is not None, f"{m1.get(1)} vs {m2.get(1)}")
    ctx.gate(s, "copy_stream_equals_cold", s1[1] == s2[1], md5_ints(s1[1]))
    ctx.gate(
        s,
        "source_chain_intact",
        eng.ret[0] is not None and len(eng.ret[0]) >= n - 512,
        f"len={None if eng.ret[0] is None else len(eng.ret[0])}",
    )


def stash(ctx, n_long=16384, n_short=1024, rounds=16, seed=5):
    """stash_check in-process (HY4_STASH=1 + HY4_BLOCKS=1; set HY4_STASH_MIN low enough for n_long): short request on the row holding a long chain, long prompt returns, hit == cold."""
    s = "stash"
    eng = ctx.eng
    if not getattr(eng, "stash_on", False):
        return ctx.gate(s, "stash_enabled", None, "HY4_STASH=1 HY4_BLOCKS=1 not set")
    release_all(eng)
    A = prompt(seed, n_long)
    set_pfx(ctx, True)
    f0, m0, _, _ = prefill(ctx, {0: A})
    s0, _, _ = decode(ctx, f0, rounds, rows=[0])
    eng.release(0)
    f1, m1, _, c1 = prefill(ctx, {0: A})
    eng.release(0)
    ctx.gate(s, "chain_hit_before", c1.get(0, 0) > 0, f"last_cached={c1.get(0, 0)}")
    st0 = dict(eng.stash_stats)
    fs, _, _, _ = prefill(ctx, {0: prompt(seed + 9, n_short)})  # short request forced onto row 0
    decode(ctx, fs, 8, rows=[0])
    eng.release(0)  # restores the stash
    ctx.gate(
        s,
        "stash_saved_and_restored",
        eng.stash_stats["saved"] > st0["saved"]
        and eng.stash_stats["restored"] > st0["restored"]
        and eng.stash_stats["failed"] == 0,
        f"{eng.stash_stats}",
    )
    f2, m2, _, c2 = prefill(ctx, {0: A})
    s2, _, _ = decode(ctx, f2, rounds, rows=[0])
    eng.release(0)
    ctx.gate(s, "hit_after_stash", c2.get(0, 0) > 0, f"last_cached={c2.get(0, 0)} (stash keeps the head only)")
    ctx.gate(s, "md5_equals_cold", m2.get(0) == m0.get(0), f"{m2.get(0)} vs {m0.get(0)}")
    ctx.gate(s, "stream_equals_cold", s2[0] == s0[0], md5_ints(s2[0]))


def chunked_pf(ctx, **kw):
    """E7: chunked prefill interleaved with decode. Implemented in demo/scenario_chunked.py (HY4_CHUNKED_PF=1); reports SKIP when the engine has no prefill_begin / prefill_chunks / prefill_end."""
    from models.demos.blackhole.hy4_preview.demo.scenario_chunked import chunked_pf as impl

    return impl(ctx, **kw)


def tailpark(ctx, n_long=16384, n_short=2048, rounds=16, seed=8):
    """E2/E3 of the plan (HY4_TAILPARK=1; NOT implemented yet -> SKIP). Long chain on row A, short request forced on A (chain demoted to the tail), long prompt returns: hit from the tail, md5 == cold."""
    s = "tailpark"
    eng = ctx.eng
    if not getattr(eng, "tail_on", False):
        return ctx.gate(s, "tailpark_enabled", None, "engine has no tail_on (HY4_TAILPARK=1 not implemented yet)")
    release_all(eng)
    A = prompt(seed, n_long)
    set_pfx(ctx, True)
    f0, m0, _, _ = prefill(ctx, {0: A})
    s0, _, _ = decode(ctx, f0, rounds, rows=[0])
    eng.release(0)
    for r in (1, 2, 3):  # other rows busy: the short request must land on row 0
        prefill(ctx, {r: prompt(70 + r, 600)})
        eng.live[r] = True
    fs, _, _, _ = prefill(ctx, {0: prompt(seed + 9, n_short)})
    decode(ctx, fs, 8, rows=[0])
    eng.release(0)
    ctx.gate(
        s, "chain_demoted", getattr(eng, "tail_stats", {}).get("demoted", 0) >= 1, f"{getattr(eng, 'tail_stats', None)}"
    )
    f2, m2, _, c2 = prefill(ctx, {0: A})
    s2, _, _ = decode(ctx, f2, rounds, rows=[0])
    eng.release(0)
    ctx.gate(s, "hit_from_tail", c2.get(0, 0) >= n_long - 1024, f"last_cached={c2.get(0, 0)} of {n_long}")
    ctx.gate(s, "md5_equals_cold", m2.get(0) == m0.get(0), f"{m2.get(0)} vs {m0.get(0)}")
    ctx.gate(s, "stream_equals_cold", s2[0] == s0[0], md5_ints(s2[0]))


def long_ctx(ctx, lens=(16384,), rounds=16, seed=9):
    """long context (default 16k; set DEMO_LONG=65536,200000 on a host window): prefill ms/token, decode ms/round at depth, hit == cold at depth, golden md5. The 1M needle stays a server-level gate (needle_*.out)."""
    s = "long"
    eng = ctx.eng
    for L in lens:
        release_all(eng)
        A = prompt(seed, L)
        set_pfx(ctx, False)
        f0, m0, t0, _ = prefill(ctx, {0: A})
        s0, ms, acc = decode(ctx, f0, rounds, rows=[0])
        eng.release(0)
        ctx.perf.append(dict(kind="prefill", scen=s, len=L, rows=1, seconds=t0, ms_per_token=1e3 * t0 / L))
        ctx.perf.append(
            dict(
                kind="decode",
                scen=s,
                users=1,
                depth=L,
                ms_per_round=sum(ms) / len(ms),
                tok_per_round=sum(acc) / max(len(acc), 1),
            )
        )
        ctx.gate(
            s,
            f"L{L}:tokens_valid",
            len(s0[0]) > rounds and all(0 <= t < 200000 for t in s0[0]),
            f"prefill {t0:.1f}s = {1e3 * t0 / L:.2f} ms/token",
        )
        ctx.golden_check(s, f"L{L}:lg_md5", m0.get(0))
        ctx.golden_check(s, f"L{L}:stream_md5", md5_ints(s0[0]))
        if getattr(eng, "pfx", False):
            set_pfx(ctx, True)
            fa, ma, _, _ = prefill(ctx, {0: A})
            eng.release(0)
            fb, mb, tb, cb = prefill(ctx, {0: A})
            sb, _, _ = decode(ctx, fb, rounds, rows=[0])
            eng.release(0)
            ctx.gate(
                s,
                f"L{L}:hit_md5_equals_cold",
                cb.get(0, 0) > 0 and mb.get(0) == m0.get(0),
                f"hit {tb:.1f}s last_cached={cb.get(0, 0)} {mb.get(0)} vs {m0.get(0)}",
            )
            ctx.gate(s, f"L{L}:hit_stream_equals_cold", sb[0] == s0[0], md5_ints(sb[0]))


def perf(ctx, sizes=(2048, 8192, 16384), rounds=32):
    """perf mode: prefill ms/token (1 row and 4 rows in one call: the call costs the same, so 4 rows = 1/4 ms/token per row) and decode ms/round, tokens/round at 1 and 4 users."""
    s = "perf"
    eng = ctx.eng
    set_pfx(ctx, False)
    for L in sizes:
        release_all(eng)
        f, _, dt, _ = prefill(ctx, {0: prompt(11, L)})
        ctx.perf.append(dict(kind="prefill", scen=s, len=L, rows=1, seconds=dt, ms_per_token=1e3 * dt / L))
        eng.release(0)
        f, _, dt4, _ = prefill(ctx, {r: prompt(12 + r, L) for r in range(R)})
        ctx.perf.append(dict(kind="prefill", scen=s, len=L, rows=R, seconds=dt4, ms_per_token=1e3 * dt4 / (L * R)))
        st, ms, acc = decode(ctx, f, rounds)
        ctx.perf.append(
            dict(
                kind="decode",
                scen=s,
                users=R,
                depth=L,
                ms_per_round=sum(ms) / len(ms),
                tok_per_round=sum(acc) / (len(acc) or 1),
            )
        )
    set_pfx(ctx, True)
    ctx.gate(s, "perf_recorded", True, f"{len(ctx.perf)} rows")


SCENARIOS = {
    "multirow": multirow,
    "masked": masked,
    "cap0": cap0,
    "pfx_hit_cold": pfx_hit_cold,
    "xrow": xrow,
    "stash": stash,
    "chunked_pf": chunked_pf,
    "tailpark": tailpark,
    "long": long_ctx,
    "perf": perf,
}
DEFAULT = ["multirow", "masked", "cap0", "pfx_hit_cold"]

try:  # hy4sched scenarios (demo/scenario_sched.py): sched_xrow5, sched_cached, sched_victim
    from models.demos.blackhole.hy4_preview.demo import scenario_sched as _SS

    SCENARIOS.update(_SS.SCENARIOS)
except Exception:  # noqa: BLE001  (never break the base scenarios)
    pass

# capacity tiers (tail-park / host tier): demo/scenario_capacity.py overrides the SKIP stub ``tailpark`` above
from models.demos.blackhole.hy4_preview.demo.scenario_capacity import SCENARIOS as _CAP  # noqa: E402

SCENARIOS.update(_CAP)
