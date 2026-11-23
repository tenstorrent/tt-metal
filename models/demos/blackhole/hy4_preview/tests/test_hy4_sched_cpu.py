# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the hy4sched features (flags + live files, usage cached_tokens wrapper, victim by cost, tier counters, demo scenarios sched_*). No device.
Run: PYTHONPATH=<worktree> python -m pytest --noconftest tests/test_hy4_sched_cpu.py"""
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from models.demos.blackhole.hy4_preview.demo import scenarios as S
from models.demos.blackhole.hy4_preview.tests.test_hy4_demo_cpu import FakeEngine, R
from models.demos.blackhole.hy4_preview.tt import hy4_pfx_policy as PP
from models.demos.blackhole.hy4_preview.tt import hy4_sched_flags as SF
from models.demos.blackhole.hy4_preview.tt.hy4_apc_hook import Hy4ApcHook, fingerprint, install_cached_tokens_report
from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import Hy4Engine
from models.demos.blackhole.hy4_preview.tt.hy4_xrow import plan_copies


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for v in (
        "HY4_CACHED_TOKENS",
        "HY4_RIDMAP",
        "HY4_PFX_EVICT_COST",
        "HY4_PFX_TIER_LOG",
        "HY4_PFX_EVICT",
        "HY4_XROW",
        "HY4_XROW_FILE",
        "HY4_PFX_FILE",
    ):
        monkeypatch.delenv(v, raising=False)
        monkeypatch.delenv(v + "_FILE", raising=False)
    monkeypatch.setenv("HY4_SCHED_DIR", str(tmp_path / "sw"))
    SF.reset_cache()
    yield
    SF.reset_cache()


# ------------------------------------------------------------------------------------------------ flags
def test_flags_default_off_env_and_file_override(monkeypatch, tmp_path):
    for name in SF.FEATURES:
        assert SF.enabled(name) is False
    monkeypatch.setenv("HY4_CACHED_TOKENS", "1")
    SF.reset_cache()
    assert SF.enabled("cached_tokens")
    f = tmp_path / "c"
    f.write_text("0\n")
    monkeypatch.setenv("HY4_CACHED_TOKENS_FILE", str(f))
    SF.reset_cache()
    assert not SF.enabled("cached_tokens")  # the file overrides env in the off direction
    monkeypatch.delenv("HY4_CACHED_TOKENS")
    f.write_text("1\n")
    SF.reset_cache()
    assert SF.enabled("cached_tokens")  # ... and in the on direction (booted with nothing set)
    f.write_text("garbage")
    SF.reset_cache()
    assert not SF.enabled("cached_tokens")  # unreadable value = env value (off)


def test_flags_default_dir_file(monkeypatch, tmp_path):
    d = tmp_path / "sw"
    d.mkdir()
    (d / "ridmap").write_text("1")
    SF.reset_cache()
    assert SF.enabled("ridmap") and not SF.enabled("tier_log")


def test_flag_cache_ttl(monkeypatch, tmp_path):
    f = tmp_path / "x"
    f.write_text("1")
    monkeypatch.setenv("HY4_PFX_TIER_LOG_FILE", str(f))
    SF.reset_cache()
    assert SF.enabled("tier_log")
    f.write_text("0")
    assert SF.enabled("tier_log")  # cached for 0.2 s
    time.sleep(0.25)
    assert not SF.enabled("tier_log")


# ------------------------------------------------------------------------------------------------ usage cached_tokens
def mk_engine(ret=None):
    e = object.__new__(Hy4Engine)
    e.ret = ret or [None] * 4
    e.ret_t = [0.0] * 4
    e.live = [False] * 4
    e.pfx, e.pfx_redo, e.pf_rows = True, True, 512
    e._pfx_on = lambda: True
    return e


def fresh_sched_cls():
    """the wrapper installs once per class (idempotent): every test gets its own class"""
    return type("Sched", (Sched,), {})


class Sched:
    def __init__(self, reqs):
        self.requests = {
            k: NS(
                prefill_stats=NS(
                    num_prompt_tokens=n, num_cached_tokens=0, num_local_cached_tokens=0, num_computed_tokens=n
                )
            )
            for k, n in reqs.items()
        }
        self.out = []
        self.finished = []

    def schedule(self, *a, **k):
        return NS(scheduled_new_reqs=self.out)

    def update_from_output(self, so, mro):
        return "ok"


def test_fingerprint_stable_and_distinct():
    a = list(range(1000, 3000))
    assert fingerprint(a) == fingerprint(np.array(a, dtype=np.int64)) == fingerprint(np.array(a, dtype=np.int32))
    assert fingerprint(a) != fingerprint(a[:-1]) and fingerprint(a) != fingerprint(a[:-1] + [7])


def test_cached_tokens_fingerprint_path_and_flag(monkeypatch):
    A = np.arange(5000, 5000 + 4096, dtype=np.int32)
    e = mk_engine([A.copy(), None, None, None])  # row 0 retains A
    hook = Hy4ApcHook(e, 4)
    B = np.concatenate([A, np.arange(9, 9 + 700, dtype=np.int32)])
    SC = fresh_sched_cls()
    s = SC({"r1": len(B)})
    assert install_cached_tokens_report(hook, SC) is True and install_cached_tokens_report(hook, SC) is False
    s.out = [NS(req_id="r1", prompt_token_ids=B.tolist())]
    # flag off: pass-through, nothing recorded, nothing changed
    s.schedule()
    hook.note_prefill_fp([B], [0], [3584])
    assert (
        s.update_from_output(NS(scheduled_new_reqs=s.out), None) == "ok"
        and s.requests["r1"].prefill_stats.num_cached_tokens == 0
        and not hook.sched_peek
    )
    hook.fp_done.clear()
    monkeypatch.setenv("HY4_CACHED_TOKENS", "1")
    SF.reset_cache()
    so = s.schedule()
    assert hook.sched_peek["r1"][1] == 4096 - 512  # schedule-time peek = hit minus the REDO chunk
    hook.note_prefill_fp([B], [0], [3584])  # prefill_forward ran (no request ids)
    s.update_from_output(NS(scheduled_new_reqs=so.scheduled_new_reqs), None)
    ps = s.requests["r1"].prefill_stats
    assert ps.num_cached_tokens == ps.num_local_cached_tokens == 3584 and ps.num_computed_tokens == len(B) - 3584
    assert hook.report["actual"] == 1 and not hook.fp_done and not hook.sched_peek


def test_cached_tokens_peek_fallback_and_clamp(monkeypatch):
    monkeypatch.setenv("HY4_CACHED_TOKENS", "1")
    SF.reset_cache()
    A = np.arange(100, 100 + 2048, dtype=np.int32)
    e = mk_engine([None, A.copy(), None, None])
    hook = Hy4ApcHook(e, 4)
    B = np.concatenate([A, np.arange(7, 7 + 100, dtype=np.int32)])
    SC = fresh_sched_cls()
    s = SC({"q": len(B), "z": 300})
    install_cached_tokens_report(hook, SC)
    s.out = [NS(req_id="q", prompt_token_ids=B.tolist()), NS(req_id="z", prompt_token_ids=list(range(300)))]
    s.schedule()  # no prefill_forward note arrives (e.g. a cancelled prefill): the peek is used
    s.update_from_output(NS(scheduled_new_reqs=s.out), None)
    assert (
        s.requests["q"].prefill_stats.num_cached_tokens == 2048 - 512
        and s.requests["z"].prefill_stats.num_cached_tokens == 0
    )
    assert hook.report["peek"] == 1 and hook.report["none"] == 1
    # clamped to the prompt
    s2 = SC({"w": 100})
    hook.cached["w"] = 5000
    s2.out = [NS(req_id="w", prompt_token_ids=list(range(100)))]
    s2.update_from_output(NS(scheduled_new_reqs=s2.out), None)
    assert s2.requests["w"].prefill_stats.num_cached_tokens == 100


def test_ridmap_flag_fills_rid_row(monkeypatch):
    e = mk_engine()
    hook = Hy4ApcHook(e, 4)
    p = list(range(2000, 2600))
    hook.note_prefill_fp([p], [2], [0])
    assert hook.claim("a", p)[1] == "actual" and "a" not in hook.rid_row  # flag off: no map
    hook.note_prefill_fp([p], [2], [0])
    monkeypatch.setenv("HY4_RIDMAP", "1")
    SF.reset_cache()
    hook.claim("a", p)
    assert hook.rid_row["a"] == 2
    e.live = [False, False, True, False]
    assert hook.free_per_row(["a"]) == [1, 1, 1, 1]  # the finished request frees ITS row


def test_duplicate_prompts_claimed_in_order():
    e = mk_engine()
    hook = Hy4ApcHook(e, 4)
    p = list(range(300))
    hook.note_prefill_fp([p, p], [1, 3], [0, 512])
    assert hook.claim("a", p)[0] == 0 and hook.claim("b", p)[0] == 512


# ------------------------------------------------------------------------------------------------ victim by cost + tiers
def test_cost_model_monotone_and_anchors():
    assert PP.recompute_cost_s(0) == 0
    xs = [PP.recompute_cost_s(n) for n in (512, 4096, 65536, 262144, 1048576)]
    assert xs == sorted(xs)
    assert abs(PP.recompute_cost_s(1048576) - 4841) < 1  # the long-context anchor
    assert abs(PP.recompute_cost_s(1000) - 3.4) < 0.01 * 3.4 + 2e-3  # the short-context anchor


def test_rank_victims_cost_age_protected():
    now = 1000.0
    items = [
        ("long_old", 400000, 0.0, False),
        ("short_new", 20000, 990.0, False),
        ("mid", 100000, 500.0, False),
        ("tip", 5000, 999.0, True),
    ]
    assert PP.rank_victims(items, now) == ["short_new", "mid", "long_old", "tip"]  # pure cost: protected last
    # with a reuse discount an ancient long chain is cheaper to lose than a fresh mid one
    items2 = [("ancient_long", 400000, 0.0, False), ("fresh_mid", 150000, 990.0, False)]
    assert PP.rank_victims(items2, now, tau_s=100.0)[0] == "ancient_long"
    assert PP.rank_victims(items2, now, tau_s=0.0)[0] == "fresh_mid"


def test_pick_victim_cost_vs_lru():
    ret = [np.zeros(8192), np.zeros(2048), np.zeros(4096), None]
    ret_t = [1.0, 2.0, 3.0, 0.0]
    assert PP.pick_victim([0, 1, 2], ret, ret_t, 0, "lru") == 0  # oldest = the longest chain
    assert PP.pick_victim([0, 1, 2], ret, ret_t, 0, "cost") == 1
    assert PP.pick_victim([0, 1, 2, 3], ret, ret_t, 0, "cost") == 3  # an empty row is free
    assert PP.pick_victim([0, 1, 2], ret, ret_t, 0, "short") == 1


def test_xrow_planner_uses_cost(monkeypatch):
    rets = [
        np.arange(8192, dtype=np.int32),
        np.arange(100000, 102048, dtype=np.int32),
        np.arange(200000, 204096, dtype=np.int32),
        None,
    ]
    live = [False, False, False, True]
    p = np.arange(900000, 900600, dtype=np.int32)
    rows, copies = plan_copies(rets, live, set(), [p], [0], own_hit=[1.0, 2.0, 3.0, 0.0], evict="cost")
    assert rows == [1] and not copies
    rows, _ = plan_copies(rets, live, set(), [p], [0], own_hit=[1.0, 2.0, 3.0, 0.0])
    assert rows == [0]  # lru default unchanged


def test_tier_account_and_line():
    tier = {}
    c = PP.tier_account(
        tier, hits={0: 4096, 1: 8192, 2: 0}, start=3584, plan={1: (0, 8192)}, lens={0: 4500, 1: 9000, 2: 700}
    )
    assert (
        c["row_tokens"] == 3584
        and c["xrow_tokens"] == 3584
        and c["cold_tokens"] == (4500 - 3584) + (9000 - 3584) + 700
        and c["cold_reqs"] == 1
    )
    c2 = PP.tier_account(tier, {0: 0}, 0, {}, {0: 1000})
    assert tier["cold_reqs"] == 2 and "pfx_tiers" in PP.tier_line(c2, tier)


# ------------------------------------------------------------------------------------------------ demo scenarios with a fake engine
class Fake2(FakeEngine):
    """FakeEngine + the real row planner (assign_rows policy), a cross-row copy and the tier counters."""

    xrow_on = True
    pf_rows = 512
    _evict_mode = Hy4Engine._evict_mode
    _evict_tau = Hy4Engine._evict_tau
    match_len = Hy4Engine.match_len

    def __init__(self, **k):
        super().__init__(**k)
        self.xrow = object()
        self.xrow_plan = {}
        self.xrow_stats = dict(copies=0, failed=0, tokens=0, seconds=0.0)
        self.ret_t = [0.0] * R
        self.tier_stats = {}
        self.pfx_redo = True

    def assign_rows(self, prompts, cur_rows):
        from models.demos.blackhole.hy4_preview.tt import hy4_xrow as XR

        arrs = [np.asarray(p, dtype=np.int32) for p in prompts]
        rets = [None if r is None else np.asarray(r, dtype=np.int32) for r in self.ret]
        rows, copies = (
            plan_copies(
                rets,
                self.live,
                set(),
                arrs,
                cur_rows,
                own_hit=self.ret_t,
                evict=self._evict_mode(),
                tau_s=self._evict_tau(),
            )
            if XR.enabled()
            else (list(cur_rows), {})
        )
        self.xrow_plan = {rows[i]: sc for i, sc in copies.items()}
        return rows

    def prefill(self, prompts_by_row, sampling=None, rids=None):
        plan, self.xrow_plan = self.xrow_plan, {}
        for r, (src, h) in plan.items():
            self.ret[r] = np.array(self.ret[src][:h], dtype=np.int32)
            self.xrow_stats["copies"] += 1
            self.xrow_stats["tokens"] += h
            self.xrow_stats["seconds"] += 0.0183 * h / 1e5
        out = super().prefill(prompts_by_row, sampling, rids)
        hits = dict(self.last_cached)
        for r, h in hits.items():
            if h > 0 and self.pfx_redo:
                self.last_cached[r] = h - 512  # the real engine recomputes the last chunk of a hit
        if SF.enabled("tier_log"):
            start = min(self.last_cached.values()) if self.last_cached else 0
            PP.tier_account(
                self.tier_stats,
                hits,
                hits and min(h - 512 if h else 0 for h in hits.values()) or 0,
                plan,
                {r: len(p) for r, p in prompts_by_row.items()},
            )
        now = time.time()
        for r in prompts_by_row:
            n = len(prompts_by_row[r]) // 512 * 512
            self.ret[r] = np.array(prompts_by_row[r][:n], dtype=np.int32) if n else None
            self.ret_t[r] = now
        return out


def gates(ctx):
    return {g[1]: g[2] for g in ctx.gates}


def run(scen, eng, **kw):
    ctx = S.Ctx(eng, log=lambda *_: None)
    S.SCENARIOS[scen](ctx, **kw)
    return ctx


def test_registered():
    assert {"sched_xrow5", "sched_cached", "sched_victim"} <= set(S.SCENARIOS)


def test_sched_absent_features_skip():
    for scen in ("sched_xrow5",):
        ctx = run(scen, FakeEngine())
        assert ctx.gates and all(g[2] == "SKIP" for g in ctx.gates)
    e = FakeEngine(pfx=False)
    for scen in ("sched_cached", "sched_victim"):
        assert all(g[2] == "SKIP" for g in run(scen, e).gates)


def test_sched_xrow5_pass(monkeypatch):
    monkeypatch.setenv("HY4_XROW", "1")
    ctx = run("sched_xrow5", Fake2(), n=4096, tail=512, rounds=3)
    st = gates(ctx)
    assert all(v == "PASS" for v in st.values()), ctx.gates
    assert "copy_ms_per_100k_vs_ds" in st and ctx.perf and abs(ctx.perf[0]["ms_per_100k"] - 18.3) < 1e-6


def test_sched_xrow5_detects_corrupt_copy(monkeypatch):
    monkeypatch.setenv("HY4_XROW", "1")
    ctx = run("sched_xrow5", Fake2(corrupt_hit=True), n=4096, tail=512, rounds=3)
    assert gates(ctx)["md5_equals_cold"] == "FAIL"


def test_sched_victim_pass(monkeypatch):
    monkeypatch.setenv("HY4_XROW", "1")
    ctx = run("sched_victim", Fake2(), n=4096)
    st = gates(ctx)
    assert all(v == "PASS" for v in st.values()), ctx.gates


def test_sched_cached_pass():
    e = Fake2()
    e.pfx_redo = True
    ctx = run("sched_cached", e, n=2048, extra=512)
    assert all(v == "PASS" for v in gates(ctx).values()), ctx.gates


def test_depth_note_flag_and_buckets(monkeypatch, capsys):
    from models.demos.blackhole.hy4_preview.tt import generator_vllm as G

    dummy = NS()
    note = lambda *a: G.HYV4ForCausalLM._depth_note(dummy, *a)
    note([(0, 0)], [70000], 1, 50.0)
    assert not hasattr(dummy, "_depth")  # flag off: nothing recorded
    monkeypatch.setenv("HY4_POS_BUCKETS", "1")
    SF.reset_cache()
    note([(0, 0), (1, 1)], [70000, 5000], 2, 60.0)
    note([(0, 0), (1, 1)], [70100, 5000], 2, 40.0)
    assert dummy._depth == {(64, 2): [100.0, 2]}
