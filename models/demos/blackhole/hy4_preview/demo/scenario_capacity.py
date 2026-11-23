# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capacity-tier demo scenarios (agent hy4cap): tail-park (HY4_TAILPARK=1) and host tier (HY4_HOSTTIER=1). Registered in demo/scenarios.SCENARIOS (this module overrides the SKIP stub ``tailpark``).

Drive ``Hy4Engine`` through its API only (prefill / round / release / assign_rows / last_cached / cap), so the SAME functions run on the device engine (text_demo --scenarios tailpark,hostspill,...) and on the CPU fake
(demo/fake_engine.FakeCapEngine; ``python -m models.demos.blackhole.hy4_preview.demo.scenario_capacity`` runs all of them on the fake). Gate line: ``DEMO_GATE <scenario> <name> PASS|FAIL|SKIP <value>``.

Device notes: the chain sizes default to 16k tokens (cold 55 s at 3.4 ms/token [M]); export HY4_CAP_MIN_SPILL=8192 (or lower) so a 16k chain qualifies; HY4_XROW=1 for cross-row tails and the host tier
(stage tensors); the host tier scenario switches the tail off through HY4_TAILPARK_FILE. ``hit == cold``: the logits md5 of the hit equals the md5 of the first (cold) prefill of the same prompt AND of a prefill with
the prefix cache switched off (HY4_PFX_FILE), and the 48 greedy tokens equal.
"""
from __future__ import annotations

import os
import tempfile

from models.demos.blackhole.hy4_preview.demo import scenarios as S

R = 4
NTOK = 48


def _cap(eng):
    return getattr(eng, "cap", None)


def _need(ctx, s, flag, attr):
    eng = ctx.eng
    if _cap(eng) is None or not getattr(eng, attr, False):
        ctx.gate(s, f"{flag}_enabled", None, f"{flag}=1 not set / capacity tiers not built")
        return False
    # the scenarios test ONE tier: the other is switched off live (HY4_HOSTTIER_FILE / HY4_TAILPARK_FILE) so the chain cannot go to the other tier
    if flag == "HY4_TAILPARK":
        _switch_file(ctx, "HY4_HOSTTIER_FILE", False)
        _switch_file(ctx, "HY4_TAILPARK_FILE", True)
    else:
        _switch_file(ctx, "HY4_HOSTTIER_FILE", True)
    return True


def _busy(ctx, rows, n=600, seed=70):
    """rows busy with their own requests (live): the next request must land on the remaining free row."""
    for r in rows:
        S.prefill(ctx, {r: S.prompt(seed + r, n)})
        ctx.eng.live[r] = True


def _free_all(ctx):
    S.release_all(ctx.eng)


def _stat(eng, k):
    return _cap(eng).store.stats[k]


def _switch_file(ctx, var, on):
    f = getattr(ctx, "_sw", None) or {}
    ctx._sw = f
    if var not in f:
        f[var] = tempfile.NamedTemporaryFile("w", suffix=".sw", delete=False).name
        os.environ[var] = f[var]
    with open(f[var], "w") as fh:
        fh.write("1\n" if on else "0\n")


def tailpark(ctx, n_long=16384, n_short=2048, rounds=NTOK, seed=8):
    """long chain on row A, short request forced on A (rows B-D busy), long prompt returns: hit from the tail (last_cached >= n_long - 1024, pfx_start > 0), logits md5 and 48 greedy tokens == cold."""
    s = "tailpark"
    if not _need(ctx, s, "HY4_TAILPARK", "tail_on"):
        return
    eng = ctx.eng
    _free_all(ctx)
    S.set_pfx(ctx, True)
    A = S.prompt(seed, n_long)
    f0, m0, t0, _ = S.prefill(ctx, {0: A})
    s0, _, _ = S.decode(ctx, f0, rounds, rows=[0])
    eng.release(0)
    _busy(ctx, (1, 2, 3))
    d0 = _stat(eng, "demoted_tail")
    want = eng.assign_rows([S.prompt(seed + 9, n_short)], [0])
    ctx.gate(s, "short_forced_on_row0", want == [0], f"assign_rows={want}")
    fs, _, _, _ = S.prefill(ctx, {0: S.prompt(seed + 9, n_short)})
    S.decode(ctx, fs, 8, rows=[0])
    eng.release(0)
    ctx.gate(s, "chain_demoted_to_tail", _stat(eng, "demoted_tail") == d0 + 1, f"{eng.cap.store.snapshot()}")
    f2, m2, t2, c2 = S.prefill(ctx, {0: A})
    s2, _, _ = S.decode(ctx, f2, rounds, rows=[0])
    eng.release(0)
    ctx.gate(
        s,
        "hit_from_tail",
        c2.get(0, 0) >= n_long - 1024,
        f"last_cached={c2.get(0, 0)} of {n_long}; cold {t0:.1f}s hit {t2:.1f}s",
    )
    ctx.gate(s, "promoted_tail_counted", _stat(eng, "promoted_tail") >= 1, f"{eng.cap.last or ''}")
    ctx.gate(s, "hit_md5_equals_cold", m2.get(0) == m0.get(0) and m0.get(0) is not None, f"{m2.get(0)} vs {m0.get(0)}")
    ctx.gate(s, "hit_48_tokens_equal_cold", s2[0] == s0[0], S.md5_ints(s2[0]))
    # same with a new suffix, against a cache-off cold prefill
    B = A + S.prompt(seed + 1, 1024)
    _busy(ctx, (1, 2, 3))
    S.prefill(ctx, {0: S.prompt(seed + 10, n_short)})
    eng.release(0)
    f3, m3, _, c3 = S.prefill(ctx, {0: B})
    s3, _, _ = S.decode(ctx, f3, rounds, rows=[0])
    eng.release(0)
    S.set_pfx(ctx, False)
    f4, m4, _, c4 = S.prefill(ctx, {0: B})
    s4, _, _ = S.decode(ctx, f4, rounds, rows=[0])
    eng.release(0)
    S.set_pfx(ctx, True)
    ctx.gate(s, "suffix_hit_from_tail", c3.get(0, 0) >= n_long - 1024, f"last_cached={c3.get(0, 0)}")
    ctx.gate(s, "suffix_md5_equals_cache_off_cold", m3.get(0) == m4.get(0), f"{m3.get(0)} vs {m4.get(0)}")
    ctx.gate(s, "suffix_48_tokens_equal_cold", s3[0] == s4[0], S.md5_ints(s3[0]))
    ctx.gate(s, "cache_off_has_no_hit", c4.get(0, 0) == 0, f"last_cached={c4.get(0, 0)}")
    ctx.golden_check(s, "cold_lg_md5", m0.get(0))
    _free_all(ctx)


def hostspill(ctx, n_long=16384, n_short=2048, rounds=NTOK, seed=11):
    """host tier only (tail switched off by HY4_TAILPARK_FILE): the chain is read to host when the short request overwrites it and restored on the return: hit == cold."""
    s = "hostspill"
    if not _need(ctx, s, "HY4_HOSTTIER", "host_on"):
        return
    eng = ctx.eng
    _free_all(ctx)
    S.set_pfx(ctx, True)
    _switch_file(ctx, "HY4_TAILPARK_FILE", False)
    A = S.prompt(seed, n_long)
    f0, m0, _, _ = S.prefill(ctx, {0: A})
    s0, _, _ = S.decode(ctx, f0, rounds, rows=[0])
    eng.release(0)
    _busy(ctx, (1, 2, 3))
    d0 = _stat(eng, "demoted_host")
    fs, _, _, _ = S.prefill(ctx, {0: S.prompt(seed + 9, n_short)})
    S.decode(ctx, fs, 8, rows=[0])
    eng.release(0)
    ctx.gate(s, "chain_spilled_to_host", _stat(eng, "demoted_host") == d0 + 1, f"{eng.cap.store.snapshot()}")
    f2, m2, t2, c2 = S.prefill(ctx, {0: A})
    s2, _, _ = S.decode(ctx, f2, rounds, rows=[0])
    eng.release(0)
    ctx.gate(
        s, "hit_from_host", c2.get(0, 0) >= n_long - 1024, f"last_cached={c2.get(0, 0)} of {n_long}; prefill {t2:.1f}s"
    )
    ctx.gate(s, "promoted_host_counted", _stat(eng, "promoted_host") >= 1, "")
    ctx.gate(s, "hit_md5_equals_cold", m2.get(0) == m0.get(0), f"{m2.get(0)} vs {m0.get(0)}")
    ctx.gate(s, "hit_48_tokens_equal_cold", s2[0] == s0[0], S.md5_ints(s2[0]))
    _switch_file(ctx, "HY4_TAILPARK_FILE", True)
    _free_all(ctx)


def tail_invalidate(ctx, n_long=16384, n_short=2048, rounds=NTOK, seed=12):
    """a chain parked in a tail is invalidated by a request that needs those pages: no stale hit afterwards (a later prompt with that prefix is cold and equals cold). Device: needs a request that reaches the tail
    page (pool of the boot MAXLEN); only run on a small MAXLEN boot (262144: the tail starts near the top of the pool); on a 1M boot the gate is SKIP.
    """
    s = "tail_invalidate"
    if not _need(ctx, s, "HY4_TAILPARK", "tail_on"):
        return
    eng = ctx.eng
    _free_all(ctx)
    S.set_pfx(ctx, True)
    A = S.prompt(seed, n_long)
    f0, m0, _, _ = S.prefill(ctx, {0: A})
    eng.release(0)
    _busy(ctx, (1, 2, 3))
    S.prefill(ctx, {0: S.prompt(seed + 9, n_short)})
    eng.release(0)
    ents = list(eng.cap.store.entries.values())
    if not ents:
        return ctx.gate(s, "tail_parked", False, f"{eng.cap.store.snapshot()}")
    e = ents[0]
    need = e.row if e.tier == "tail" else None
    if need is None:
        return ctx.gate(s, "entry_in_tail", None, "entry went to host")
    reach = e.page0 * 512 + 1024
    if reach > int(getattr(eng, "max_seq_len", 1 << 30)) - 4096:
        return ctx.gate(
            s, "tail_reachable", None, f"tail at {reach} tokens is above max_seq_len - 4096 (use a small MAXLEN boot)"
        )
    big = S.prompt(seed + 3, reach)
    i0 = _stat(eng, "invalidated")
    S.prefill(ctx, {0: big})
    eng.release(0)
    ctx.gate(
        s,
        "tail_invalidated_by_request",
        _stat(eng, "invalidated") == i0 + 1 and not eng.cap.store.entries,
        f"{eng.cap.store.snapshot()}",
    )
    f2, m2, _, c2 = S.prefill(ctx, {0: A})
    s2, _, _ = S.decode(ctx, f2, rounds, rows=[0])
    eng.release(0)
    ctx.gate(s, "no_stale_hit", c2.get(0, 0) == 0, f"last_cached={c2.get(0, 0)}")
    ctx.gate(s, "cold_md5_equals_first_cold", m2.get(0) == m0.get(0), f"{m2.get(0)} vs {m0.get(0)}")
    _free_all(ctx)


def tail_neighbours(ctx, n_long=16384, n_short=2048, rounds=24, seed=13):
    """rows B-D decode (rounds running) while row A demotes and promotes: their greedy streams are bit-identical to the uninterrupted run (masked-scenario pattern)."""
    s = "tail_neighbours"
    if not _need(ctx, s, "HY4_TAILPARK", "tail_on"):
        return
    eng = ctx.eng

    def go(interfere):
        _free_all(ctx)
        S.set_pfx(ctx, True)
        A = S.prompt(seed, n_long)
        S.prefill(ctx, {0: A})
        eng.release(0)
        P = {r: S.prompt(seed + 20 + r, 1500 + 300 * r) for r in (1, 2, 3)}
        first = S.prefill(ctx, P)[0]
        st = {r: [first[r]] for r in (1, 2, 3)}

        def rd(n):
            for _ in range(n):
                out = eng.round([eng.K if eng.live[r] else 0 for r in range(R)], False)
                for r in (1, 2, 3):
                    st[r] += out[r]["a"][: out[r]["m"] + 1]

        rd(rounds // 2)
        if interfere:
            S.prefill(ctx, {0: S.prompt(seed + 9, n_short)})  # demote of A's chain (row 0 idle -> chosen by the caller)
            eng.release(0)
        rd(rounds // 4)
        if interfere:
            S.prefill(ctx, {0: A})  # promote
            eng.release(0)
        rd(rounds // 4)
        return st

    ref = go(False)
    n0 = _stat(eng, "demoted_tail") + _stat(eng, "promoted_tail")
    got = go(True)
    moved = _stat(eng, "demoted_tail") + _stat(eng, "promoted_tail") - n0
    ctx.gate(s, "tail_traffic_happened", moved >= 2, f"demoted+promoted={moved}")
    for r in (1, 2, 3):
        ctx.gate(s, f"row{r}_stream_bit_identical", got[r] == ref[r], S.md5_ints(got[r]))
    _free_all(ctx)


def tail_error(ctx, n_long=16384, n_short=2048, rounds=NTOK, seed=14):
    """error injection: the copy raises (demote, then promote) -> the chain is dropped, the prompt is recomputed cold and equals cold, nothing wrong is ever served. Wraps the backend's copy_pages."""
    s = "tail_error"
    if not _need(ctx, s, "HY4_TAILPARK", "tail_on"):
        return
    eng = ctx.eng
    be = eng.cap.store.be
    orig = be.copy_pages
    box = {"n": 0}

    def flaky(*a, **k):
        if box["n"] > 0:
            box["n"] -= 1
            raise RuntimeError("injected copy failure")
        return orig(*a, **k)

    be.copy_pages = flaky
    _switch_file(ctx, "HY4_HOSTTIER_FILE", False)  # the failing tail must not cascade to the host tier in this scenario
    try:
        _free_all(ctx)
        S.set_pfx(ctx, True)
        A = S.prompt(seed, n_long)
        f0, m0, _, _ = S.prefill(ctx, {0: A})
        eng.release(0)
        _busy(ctx, (1, 2, 3))
        f1 = _stat(eng, "failed")
        box["n"] = 8  # every candidate fails
        S.prefill(ctx, {0: S.prompt(seed + 9, n_short)})
        eng.release(0)
        box["n"] = 0
        ctx.gate(
            s,
            "demote_error_counted_no_entry",
            _stat(eng, "failed") > f1 and not eng.cap.store.entries,
            f"{eng.cap.store.snapshot()}",
        )
        f2, m2, _, c2 = S.prefill(ctx, {0: A})
        eng.release(0)
        ctx.gate(
            s,
            "after_failed_demote_cold_equals_cold",
            c2.get(0, 0) == 0 and m2.get(0) == m0.get(0),
            f"last_cached={c2.get(0, 0)} {m2.get(0)} vs {m0.get(0)}",
        )
        # now a good demote, then the promote fails
        _busy(ctx, (1, 2, 3))
        S.prefill(ctx, {0: S.prompt(seed + 10, n_short)})
        eng.release(0)
        ok = len(eng.cap.store.entries) == 1
        box["n"] = 1
        f3, m3, _, c3 = S.prefill(ctx, {0: A})
        s3, _, _ = S.decode(ctx, f3, rounds, rows=[0])
        eng.release(0)
        ctx.gate(
            s,
            "promote_error_falls_back_cold",
            ok and c3.get(0, 0) == 0 and m3.get(0) == m0.get(0),
            f"parked={ok} last_cached={c3.get(0, 0)} {m3.get(0)} vs {m0.get(0)}",
        )
        ctx.gate(s, "failed_entry_dropped", not eng.cap.store.entries, f"{eng.cap.store.snapshot()}")
    finally:
        be.copy_pages = orig
        _switch_file(ctx, "HY4_HOSTTIER_FILE", True)
        _free_all(ctx)


SCENARIOS = {
    "tailpark": tailpark,
    "hostspill": hostspill,
    "tail_invalidate": tail_invalidate,
    "tail_neighbours": tail_neighbours,
    "tail_error": tail_error,
}


def main(argv=None):
    """CPU run on the fake engine (no device): every scenario, one line per gate; exit 1 on a FAIL or a SKIP."""
    import sys

    from models.demos.blackhole.hy4_preview.demo.fake_engine import FakeCapEngine

    ok = True
    for name, fn in SCENARIOS.items():
        pages = 2056 if name != "tail_invalidate" else 160
        eng = FakeCapEngine(pages=pages, tail=True, host=True, min_spill=8192)
        eng.max_seq_len = pages * 512
        ctx = S.Ctx(eng)
        fn(ctx)
        ok &= all(g[2] == "PASS" for g in ctx.gates)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
