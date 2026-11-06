# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) of the placement of paged-APC prefix hits with the row-aware placement (B=16: DSV41_PREFILL_ROW_AWARE=1, Up 2 < U 4): the REAL ``_place_row_aware`` (taken from the source of
tt/generator_vllm.py) must keep the start of a hit (the cached prefix is restored, only [start, end) is computed) and bind the request to the user the index reserved. Regression of the first B=16
server test: the row-aware branch reset every hit to start 0 ("chunk continuation at N without carried state: recomputing from position 0"): hit TTFT == cold TTFT, 4 concurrent hits 2x cold.
Counts the prefill chunks (512 tokens) computed for a full-hit prompt and for 4 concurrent hits. Run (from the worktree root):
  python_env/bin/python -m pytest --noconftest -c /dev/null --rootdir=/tmp tests/test_apc_paged_hit_chunks_cpu.py -q"""

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace

import torch

from models.demos.blackhole.deepseek_v41_flash.tt.apc_paged import HitPlan
from models.demos.blackhole.deepseek_v41_flash.tt.vllm_state import SlotTable

C, U, UP, ROWS = 512, 4, 2, 4
PROMPT = 65536


class _Log:
    def info(self, *a, **k):
        self.msgs.append(a[0])

    warning = info
    msgs = []


def _place_fn():
    src = (Path(__file__).resolve().parents[1] / "tt" / "generator_vllm.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_place_row_aware")
    log = _Log()
    log.msgs = []
    ns = {"logger": log}
    exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)
    return ns["_place_row_aware"], log


class FakeIdx:
    def __init__(self, plans):
        self.plans = plans

    def reservation(self, rid):
        return self.plans.get(rid)


class FakeInprog:
    users = {}

    def find(self, *a, **k):
        return None


def make(plans):
    fn, log = _place_fn()
    slots = SlotTable(16, 16)
    g = SimpleNamespace(
        U=U,
        m=SimpleNamespace(__dict__={"pf_resume": {}}),
        inprog=FakeInprog(),
        slots=slots,
        paged=SimpleNamespace(idx=FakeIdx(plans)),
        piece_lag=False,
        pf_done=set(),
        row_stats={"placed": 0, "conflicts": 0},
        _busy_rows=lambda: set(),
    )
    return fn, g, log


def chunks(items):
    return sum(-(-(e - st) // C) for _, _, st, e in items)


def waves(items):
    by_row = {}
    for b, *_ in items:
        by_row[b // U] = by_row.get(b // U, 0) + 1
    return -(-max(by_row.values()) // UP)


def toks(n):
    return torch.zeros(n, PROMPT, dtype=torch.int64)


def test_full_hit_computes_only_the_tail():
    hit = 61440
    fn, g, _ = make({"r0": HitPlan(hit_tokens=hit, row=2, user=9)})
    items, phys = fn(g, toks(1), [hit], [PROMPT], [0], ["r0"], ())
    assert phys == [9] and items[0][2] == hit  # reserved user, start kept
    assert chunks(items) == 8  # cold: 128
    assert g.hit_rows_last == {0}


def test_cold_prompt_is_128_chunks():
    fn, g, _ = make({})
    items, _ = fn(g, toks(1), [0], [PROMPT], [0], ["r0"], ())
    assert chunks(items) == 128


def test_start_without_reservation_still_recomputes():
    fn, g, log = make({})
    items, _ = fn(g, toks(1), [61440], [PROMPT], [0], ["rX"], ())
    assert items[0][2] == 0 and chunks(items) == 128
    assert any("recomputing" in m for m in log.msgs)
    fn, g, _ = make({})
    items, _ = fn(g, toks(1), [61440], [PROMPT], [0], None, ())  # (no request ids: the old behaviour)
    assert items[0][2] == 0


def test_reservation_with_other_hit_length_is_not_used():
    fn, g, _ = make({"r0": HitPlan(hit_tokens=4096, row=0, user=1)})
    items, _ = fn(g, toks(1), [61440], [PROMPT], [0], ["r0"], ())
    assert items[0][2] == 0


def test_four_concurrent_hits_on_four_rows_one_wave():
    hit = 61440
    plans = {f"r{i}": HitPlan(hit_tokens=hit, row=i, user=i * U) for i in range(4)}
    fn, g, _ = make(plans)
    items, phys = fn(g, toks(4), [hit] * 4, [PROMPT] * 4, [0, 1, 2, 3], [f"r{i}" for i in range(4)], ())
    assert phys == [0, 4, 8, 12] and all(it[2] == hit for it in items)
    assert chunks(items) == 32  # 4 x 8, cold: 4 x 128
    assert waves(items) == 1


def test_four_concurrent_hits_in_one_row_two_waves_of_tail_chunks():
    hit = 61440
    plans = {f"r{i}": HitPlan(hit_tokens=hit, row=1, user=4 + i) for i in range(4)}
    fn, g, _ = make(plans)
    items, phys = fn(g, toks(4), [hit] * 4, [PROMPT] * 4, [0, 1, 2, 3], [f"r{i}" for i in range(4)], ())
    assert sorted(phys) == [4, 5, 6, 7] and all(it[2] == hit for it in items)
    assert chunks(items) == 32 and waves(items) == 2  # 2 waves x 8 chunks, not 2 x 128


def test_new_prompt_avoids_a_user_reserved_for_a_hit():
    fn, g, _ = make({})
    reserved = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    items, phys = fn(g, toks(1), [0], [4096], [0], ["rN"], reserved)
    assert phys == [15]
