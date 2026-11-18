# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Hy4 plugin prefix hook (tt/hy4_apc_hook.py: peek_home over the row-local chains) and of the loss-reason classifier (Hy4Engine.classify_loss); no device, no vLLM.
Run: PYTHONPATH=<worktree> python -m pytest --noconftest tests/test_hy4_apc_hook_cpu.py"""
import numpy as np

from models.demos.blackhole.hy4_preview.tt.hy4_apc_hook import Hy4ApcHook
from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import ROWS, Hy4Engine


def eng():
    e = object.__new__(Hy4Engine)
    e.pfx, e.pf_rows, e.pfx_redo = True, 512, True
    e.ret, e.ret_t, e.live = [None] * ROWS, [0.0] * ROWS, [False] * ROWS
    e.pfx_stats = dict(calls=0, hit_calls=0, hit_tokens=0, prompt_tokens=0, computed_tokens=0)
    e.hist, e.hist_max, e.loss_log = [], 24, True
    return e


def toks(n, seed):
    return np.random.default_rng(seed).integers(1000, 100000, n).astype(np.int32)


def test_hit_alignment_and_one_token_left():
    e = eng()
    h = Hy4ApcHook(e)
    a = toks(5000, 1)
    e.ret[2] = a[:4608].copy()
    assert h.peek_home("r", a, 5000) == (4608, 2, [1, 1, 1, 1])  # 4608 = 9 pages, 392 tokens left
    assert h.peek_home("r", a[:4608], 4608)[0] == 4096  # the prompt equals the chain: one token must stay
    b = a.copy()
    b[1500] += 1
    assert h.peek_home("r", b, 5000)[:2] == (1024, 2)
    assert h.peek_home("r", toks(5000, 9), 5000)[:2] == (0, -1)


def test_num_tokens_truncates_prompt_list():
    e = eng()
    h = Hy4ApcHook(e)
    a = toks(3000, 3)
    e.ret[0] = a[:2560].copy()
    assert h.peek_home("r", list(a) + [7] * 50, 3000)[:2] == (2560, 0)


def test_free_flags_live_and_finished_ids():
    e = eng()
    h = Hy4ApcHook(e)
    a = toks(3000, 1)
    e.ret[1] = a[:2560].copy()
    e.live = [True, True, False, True]
    assert h.peek_home("r", a, 3000)[2] == [0, 0, 1, 0]
    h.note_prefill(["x0", "x1"], [0, 1])
    # the scheduler finished request x1 (row 1) that the model has not released yet: row 1 counts as free
    assert h.peek_home("r", a, 3000, finished_ids=["x1", "unknown"])[2] == [0, 1, 1, 0]


def test_tie_prefers_free_row_and_busy_home_is_reported():
    e = eng()
    h = Hy4ApcHook(e)
    a = toks(3000, 1)
    e.ret[0], e.ret[3] = a[:2560].copy(), a[:2560].copy()
    e.live = [True, False, False, False]
    hit, home, free = h.peek_home("r", a, 3000)
    assert (hit, home) == (2560, 3) and free[0] == 0
    e.live = [True, False, False, True]  # both holders busy: home = first, free flag 0 -> the plugin holds
    hit, home, free = h.peek_home("r", a, 3000)
    assert hit == 2560 and free[home] == 0


def test_touch_refreshes_lru_of_home_row_only_when_asked():
    e = eng()
    h = Hy4ApcHook(e)
    a = toks(3000, 1)
    e.ret[2] = a[:2560].copy()
    e.ret_t[2] = 5.0
    h.peek_home("r", a, 3000)
    assert e.ret_t[2] == 5.0
    h.peek_home("r", a, 3000, touch=True)
    assert e.ret_t[2] > 5.0
    # tip protection effect: with no match the engine's LRU choice among free rows skips the touched chain
    e.ret[0], e.ret_t[0] = toks(1024, 5), 9e18
    e.ret[1], e.ret_t[1] = toks(1024, 6), 1.0
    e.ret[3], e.ret_t[3] = toks(1024, 7), 2.0
    assert e.assign_rows([toks(2000, 11)], [0]) == [1]


def test_switch_file_off_means_no_hit(tmp_path, monkeypatch):
    e = eng()
    h = Hy4ApcHook(e)
    a = toks(3000, 1)
    e.ret[1] = a[:2560].copy()
    f = tmp_path / "pfx"
    f.write_text("0")
    monkeypatch.setenv("HY4_PFX_FILE", str(f))
    assert h.peek_home("r", a, 3000) == (0, -1, [1, 1, 1, 1])
    f.write_text("1")
    assert h.peek_home("r", a, 3000)[:2] == (2560, 1)


def test_plan_hit_zero_cancel_noop_and_garbage_is_a_miss():
    e = eng()
    h = Hy4ApcHook(e)
    assert h.plan_hit("r", [1, 2, 3], 3, max_hit=512) == (0, None)
    assert h.cancel("r") is None
    assert h.peek_home("r", [], 0)[:2] == (0, -1)
    e.pfx = False
    assert h.peek_home("r", [1] * 600, 600)[:2] == (0, -1)


# ------------------------------------------------------------------ plugin contract replay (Hy4 constants: 4 rows, 1 user per row)
def plugin_hold(info, waited, t_max=60.0, h_min=32768, factor=0.4, rate=192.0):
    """mirror of TTScheduler._hold_for_home_row's decision for one request (plugin a8962b0)."""
    hit, home, free = int(info[0]), int(info[1]), list(info[2])
    if hit >= h_min and 0 <= home < len(free) and free[home] <= 0:
        return waited < min(t_max, factor * hit / rate)
    return False


def test_busy_row_scenario_hold_then_hit():
    e = eng()
    h = Hy4ApcHook(e)
    conv = [toks(40000, 100 + r) for r in range(4)]
    for r in range(4):
        e.ret[r] = conv[r][:39936].copy()
        e.live[r] = True  # 4 users decoding their turn
    nxt = np.concatenate([conv[2], toks(2000, 55)])  # row 2's conversation, next turn (30k+ prefix)
    info = h.peek_home("t", nxt, len(nxt), touch=True)
    assert info[:2] == (39936, 2) and info[2] == [0, 0, 0, 0] and plugin_hold(info, 0.0)
    assert plugin_hold(info, 30.0) and not plugin_hold(
        info, 100.0
    )  # bound = min(60, 0.4*39936/192 = 83 s) = 60 s: then admitted cold, no starvation
    e.live[2] = False  # the home row's user finished
    info = h.peek_home("t", nxt, len(nxt))
    assert info[2][2] == 1 and not plugin_hold(info, 1.0)
    assert e.assign_rows([nxt], [0]) == [2]  # and the engine's row-match puts it on row 2: hit
    assert e.hit_len(2, nxt) == 39936


def test_small_hit_or_other_row_free_is_not_held():
    e = eng()
    h = Hy4ApcHook(e)
    a = toks(9000, 1)
    e.ret[1] = a[:8704].copy()
    e.live = [True, True, False, True]
    assert not plugin_hold(h.peek_home("t", a, 9000), 0.0)  # below H_min
    b = toks(50000, 2)
    e.ret[1] = b[:49664].copy()
    e.live = [False, True, True, True]
    assert plugin_hold(h.peek_home("t", b, 50000), 0.0)


# ------------------------------------------------------------------ loss classifier
def call(e, prompts_by_row, live_before=None, start=0, hits=None):
    hits = (
        hits
        if hits is not None
        else {r: e.hit_len(r, np.asarray(p, dtype=np.int32)) for r, p in prompts_by_row.items()}
    )
    lb = live_before if live_before is not None else list(e.live)
    return e.classify_loss(prompts_by_row, hits, start, lb)


def test_loss_first_seen_then_hit_then_evicted():
    e = eng()
    a = toks(4000, 1)
    d = call(e, {0: a.tolist()})
    assert d[0]["cause"] == "first_seen"
    e.remember_prompts({0: a.tolist()}, [0])
    e.ret[0] = a[:3584].copy()
    a2 = np.concatenate([a, toks(600, 2)])
    d = call(e, {0: a2.tolist()}, hits={0: 3584}, start=3584 - 512)
    assert d[0]["cause"] in ("hit", "partial", "lockstep")  # (REDO start is one chunk lower: counted as served by hits)
    e.ret[0] = None  # overwritten by another prompt
    d = call(e, {1: a2.tolist()}, hits={1: 0}, start=0)
    assert d[1]["cause"] == "evicted" and d[1]["potential"] == 3584
    assert e.pfx_stats["loss_evicted"] == 3584


def test_loss_busy_home_and_lockstep_and_policy_off(tmp_path, monkeypatch):
    e = eng()
    a = toks(4000, 1)
    e.remember_prompts({2: a.tolist()}, [2])
    e.ret[2] = a[:3584].copy()
    a2 = np.concatenate([a, toks(600, 2)])
    e.live[2] = True
    d = call(e, {0: a2.tolist()}, hits={0: 0}, start=0)
    assert d[0]["cause"] == "busy_home"
    e.live[2] = False
    # the request got its home row but shares the call with a cold row: start = 0
    d = call(e, {2: a2.tolist()}, hits={2: 3584}, start=0)
    assert d[2]["cause"] == "lockstep"
    f = tmp_path / "pfx"
    f.write_text("0")
    monkeypatch.setenv("HY4_PFX_FILE", str(f))
    d = call(e, {0: a2.tolist()}, hits={0: 0}, start=0)
    assert d[0]["cause"] == "policy_off"


def test_history_is_bounded():
    e = eng()
    e.hist_max = 3
    for i in range(10):
        e.remember_prompts({0: toks(1024, i).tolist()}, [0])
    assert len(e.hist) == 3
