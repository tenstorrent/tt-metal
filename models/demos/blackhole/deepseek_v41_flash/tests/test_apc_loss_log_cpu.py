# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the per-event APC loss log (``loss_log`` of ``BlockIndex``, env DSV41_APC_LOSS_LOG): one line per lost / shortened hit, counters unchanged. Run: python -m pytest tests/test_apc_loss_log_cpu.py -q"""
import os

from test_apc_paged_index import AP, C, mk, run, toks


def _scenario(idx):
    """A is resident on row 0; row 0 is then full (users 0, 1 busy), row 1 free; A2 = A + more tokens arrives -> busy-row loss. -> plan of A2"""
    a = toks(1, 6 * C + 300)
    run(idx, "A", a, user=0)
    idx.bind("B", 0, user=0, prompt_len=1000)
    idx.bind("C", 0, user=1, prompt_len=1000)
    a2 = list(a) + list(toks(2, 4 * C))
    plan = idx.plan_hit("A2", a2, len(a2))
    return plan, len(a2)


def test_one_line_per_busy_row_loss_and_counters_unchanged():
    lines = []
    idx1, idx0 = mk(loss_log=lines.append), mk()
    p1, n = _scenario(idx1)
    p0, _ = _scenario(idx0)
    assert p1.hit_tokens == p0.hit_tokens == 0  # behaviour unchanged: the request is still planned cold
    assert idx1.counters == idx0.counters  # no new counters, same values
    assert idx0.counters["busy_loss"] == 1
    busy = [l for l in lines if "reason=busy_row" in l]
    assert len(busy) == 1 and "req=A2" in busy[0] and f"n={n}" in busy[0]
    assert "home_row=0" in busy[0] and "free_row=-1" in busy[0] and "attempt=1" in busy[0]
    assert f"lost_busy={idx1.counters['busy_loss_tokens']}" in busy[0] and f"recompute={n}" in busy[0]
    assert "free_users=[0, 2]" in busy[0]
    # a second plan of the same request is a second event (the scheduler retries), attempt=2
    a2 = list(toks(1, 6 * C + 300)) + list(toks(2, 4 * C))
    idx1.plan_hit("A2", a2, len(a2))
    assert len([l for l in lines if "reason=busy_row" in l]) == 2 and "attempt=2" in lines[-1]
    assert idx1.counters["busy_loss"] == 2


def test_no_line_without_loss_and_default_off():
    lines = []
    idx = mk(loss_log=lines.append)
    t = toks(3, 5 * C + 10)
    run(idx, "X", t, user=0)
    plan = idx.plan_hit("X2", t, len(t))  # a clean hit on a free row
    assert plan.hit_tokens > 0 and lines == []
    assert mk().loss_log is None


def test_eviction_line():
    lines = []
    idx = mk(loss_log=lines.append, pages_per_row=120, rows=1, users_per_row=2)
    a = toks(5, 8 * C + 10)
    run(idx, "A", a, user=0)
    for i in range(12):  # churn the one row until A's blocks are gone
        run(idx, f"F{i}", toks(100 + i, 8 * C + 10), user=0)
    assert idx.counters["ev_block"] > 0
    plan = idx.plan_hit("A2", a, len(a))
    ev = [l for l in lines if "req=A2" in l]
    assert plan.hit_tokens == 0
    assert len(ev) == 1 and "reason=evicted" in ev[0] and "lost_evicted=" in ev[0]


def test_no_lookup_line():
    lines = []
    idx = mk(loss_log=lines.append)
    idx.bind("N", 0, user=0, prompt_len=4096)  # bound without a plan_hit: the plugin never asked the model
    assert len(lines) == 1 and "reason=no_lookup" in lines[0] and "n=4096" in lines[0]
    idx.plan_hit("P", toks(9, 3 * C), 3 * C)  # planned (cold) then bound: not a no_lookup
    idx.bind("P", 0, user=1, prompt_len=3 * C)
    assert len(lines) == 1
