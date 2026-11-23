# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the hy4pfx prefix-cache policy (tt/hy4_pfx_policy.py) and of the engine stash / release / eviction wiring. No device.
Run: PYTHONPATH=<worktree> python -m pytest --noconftest tests/test_hy4_pfx_policy_cpu.py"""
import itertools

import numpy as np

from models.demos.blackhole.hy4_preview.tt import hy4_pfx_policy as PP
from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import ROWS, Hy4Engine
from models.demos.blackhole.hy4_preview.tt.hy4_xrow import plan_copies


def cost(groups, starts, lens):
    return sum(max(lens[r] for r in g) - min(starts[r] for r in g) for g in groups)


def test_split_real_case_finax():
    # 04:24:56 call of the finax run: three rows with a deep hit and one common minimum of 65024
    starts = {1: 65024, 2: 314368 - 512, 3: 433664 - 512}
    lens = {1: 65702, 2: 315064, 3: 436615}
    for mode in ("thr", "dp"):
        g = PP.plan_groups(starts, lens, mode)
        assert len(g) == 3 and g[0] == [1]
        assert cost(g, starts, lens) < 0.01 * (max(lens.values()) - min(starts.values())) * 3
    assert cost([[1, 2, 3]], starts, lens) > 350000  # the lock-step call: 1250 s


def test_split_does_not_split_when_it_costs_more():
    starts = {0: 0, 1: 9000}
    lens = {0: 400000, 1: 400000}
    assert PP.plan_groups(starts, lens, "dp") == [[0, 1]]
    assert sorted(PP.plan_groups(starts, lens, "thr")) == [
        [0],
        [1],
    ]  # the 8k-gap rule splits and pays 391k extra tokens


def test_dp_is_optimal_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 5))
        starts = {r: int(rng.integers(0, 40)) * 512 for r in range(n)}
        lens = {r: starts[r] + int(rng.integers(1, 60)) * 512 for r in range(n)}
        rows = sorted(starts, key=lambda r: (starts[r], r))
        best = min(
            cost([[rows[i] for i in range(n) if lab[i] == k] for k in range(max(lab) + 1) if k in lab], starts, lens)
            for lab in itertools.product(range(n), repeat=n)
        )
        assert cost(PP.plan_groups(starts, lens, "dp"), starts, lens) == best


def test_victim():
    ret = [np.zeros(300000), np.zeros(70000), None, np.zeros(16000)]
    t = [5.0, 1.0, 0.0, 9.0]
    assert PP.pick_victim([0, 1, 3], ret, t, 0, "lru") == 1  # oldest
    assert PP.pick_victim([0, 1, 3], ret, t, 0, "short") == 3  # shortest chain
    assert PP.pick_victim([0, 1, 2, 3], ret, t, 0, "short") == 2  # an empty row first
    assert PP.pick_victim([0], ret, t, 0, "short") == 0


def test_stash_pages_and_restore():
    assert PP.stash_pages(332800, 17556, 64) == 43  # (17556 + 4096) / 512 -> 43 pages
    assert PP.stash_pages(332800, 40000, 64) == 0  # footprint beyond the shadow pages / a prompt worth its own chain
    assert PP.stash_pages(332800, 37788, 64) == 0
    assert PP.stash_pages(20000, 3000, 64) == 0  # chain too short to protect
    ch = np.arange(332800)
    assert len(PP.restored_chain(ch, 43, 17700, 8, 512)) == 332800  # decode stayed inside the saved pages
    assert len(PP.restored_chain(ch, 43, 23000, 8, 512)) == 43 * 512  # decode ran over: first 43 pages only


class FakeCopy:
    def __init__(self):
        self.calls = []

    def copy(self, row, pairs):
        self.calls.append((row, list(pairs)))
        return 0.01


def eng():
    e = object.__new__(Hy4Engine)
    e.pfx, e.pf_rows, e.pfx_redo, e.S = True, 512, True, 4
    e.ret, e.ret_t, e.live = [None] * ROWS, [0.0] * ROWS, [False] * ROWS
    e.pend, e.pos, e.drafts, e.K, e.dirty = [0] * ROWS, [0] * ROWS, [[-1] * 3 for _ in ROWS * [0]], 3, False
    e.stash_on, e.stash, e.stash_stats = True, {}, dict(saved=0, restored=0, truncated=0, failed=0, seconds=0.0)
    e.bcopy, e.blocks_pages, e.pages, e.inflight = FakeCopy(), 64, 2048, None
    e.drain = lambda: None
    return e


def test_engine_stash_roundtrip():
    e = eng()
    chain = np.arange(332800, dtype=np.int32)
    e.ret[3] = chain
    e._stash_save(3, 17556)
    ((row, pairs),) = e.bcopy.calls
    assert row == 3 and pairs[0] == (0, 2048) and len(pairs) == 43 and 3 in e.stash
    e.ret[3] = None  # prefill clears it, then sets the short prompt
    e.live[3], e.pos[3] = True, 17700
    e.release(3)
    assert e.bcopy.calls[1] == (3, [(2048 + p, p) for p in range(43)])
    assert len(e.ret[3]) == 332800 and e.pos[3] == 332800 and not e.stash
    # decode ran past the saved pages: only the head of the chain stays valid
    e.ret[3] = chain
    e._stash_save(3, 17556)
    e.live[3], e.pos[3] = True, 30000
    e.release(3)
    assert len(e.ret[3]) == 43 * 512 and e.stash_stats["truncated"] == 1


def test_engine_stash_failure_drops_chain():
    e = eng()
    e.ret[2] = np.arange(100000, dtype=np.int32)

    def bad(row, pairs):
        raise RuntimeError("x")

    e.bcopy.copy = bad
    e._stash_save(2, 5000)
    assert not e.stash and e.stash_stats["failed"] == 1


def test_xrow_plan_copies_evict_short(monkeypatch):
    rets = [np.arange(300000, dtype=np.int32), np.arange(70000, dtype=np.int32), None, np.arange(20000, dtype=np.int32)]
    live = [True, False, False, False]
    new = np.arange(1000, dtype=np.int32) + 10**6
    monkeypatch.setenv("HY4_PFX_EVICT", "short")
    rows, _ = plan_copies(rets, live, set(), [new], [1], own_hit=[0.0, 5.0, 9.0, 1.0])
    assert rows == [2]  # empty row first
    rets[2] = np.arange(500, dtype=np.int32)
    rows, _ = plan_copies(rets, live, set(), [new], [1], own_hit=[0.0, 5.0, 9.0, 1.0])
    assert rows == [2]  # shortest retained chain, not the oldest (row 1)
    monkeypatch.setenv("HY4_PFX_EVICT", "lru")
    rows, _ = plan_copies(rets, live, set(), [new], [1], own_hit=[0.0, 5.0, 9.0, 1.0])
    assert rows == [3]


def test_cached_tokens_report_wrapper(monkeypatch):
    from types import SimpleNamespace as NS

    from models.demos.blackhole.hy4_preview.tt import hy4_sched_flags as SF

    monkeypatch.setenv("HY4_CACHED_TOKENS", "1")  # hy4sched: the feature is flag gated (default off)
    SF.reset_cache()

    from models.demos.blackhole.hy4_preview.tt.hy4_apc_hook import Hy4ApcHook, install_cached_tokens_report

    class Sched:
        def __init__(self):
            self.requests = {
                "a": NS(
                    prefill_stats=NS(
                        num_prompt_tokens=1000, num_cached_tokens=0, num_local_cached_tokens=0, num_computed_tokens=1000
                    )
                ),
                "b": NS(
                    prefill_stats=NS(
                        num_prompt_tokens=800, num_cached_tokens=0, num_local_cached_tokens=0, num_computed_tokens=800
                    )
                ),
            }
            self.called = 0

        def update_from_output(self, so, mro):
            self.called += 1
            return "ok"

    e = object.__new__(Hy4Engine)
    e.ret = [None] * ROWS
    hook = Hy4ApcHook(e, ROWS)
    hook.cached = {"a": 512, "b": 5000}
    assert install_cached_tokens_report(hook, Sched) is True and install_cached_tokens_report(hook, Sched) is False
    s = Sched()
    assert s.update_from_output(NS(scheduled_new_reqs=[NS(req_id="a"), NS(req_id="b"), NS(req_id="zz")]), None) == "ok"
    assert (
        s.requests["a"].prefill_stats.num_cached_tokens == 512
        and s.requests["a"].prefill_stats.num_computed_tokens == 488
    )
    assert s.requests["b"].prefill_stats.num_cached_tokens == 800  # clamped to the prompt
    assert s.update_from_output(NS(scheduled_new_reqs=None), None) == "ok" and s.called == 2
