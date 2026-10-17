# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests (no device) for prefix caching of the MTP spec prefill: resume-position rule, planner, engine threading.

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_prefix_cache_host.py -q
"""
import types

import pytest
import torch

from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.prefix_cache import spec_learn_positions, spec_resume_cap, spec_save_points
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine


@pytest.mark.parametrize(
    "start,L,cap",
    [
        (0, 500, 0),
        (64, 500, 0),  # start < 128: nothing to resume
        (127, 500, 0),
        (128, 500, 0),  # exactly on a boundary: slot 127 would fuse a stale token -> one block back = 0
        (129, 500, 128),
        (192, 500, 128),
        (256, 500, 128),  # on a boundary: s = start - 128
        (320, 500, 256),
        (4096, 9000, 3968),
        (4160, 9000, 4096),
        (129, 130, 0),  # fewer than 3 tokens would remain after s=128 -> walks down
        (129, 131, 128),
    ],
)
def test_resume_cap(start, L, cap):
    assert spec_resume_cap(start, L) == cap
    assert cap % 128 == 0 and (start <= 0 or cap <= start - 1) and (cap == 0 or L - cap >= 3)


def test_learn_and_save_points():
    assert spec_learn_positions(256, 1000) == {128, 256}
    assert spec_learn_positions(320, 1000) == {256}
    assert spec_learn_positions(0, 1000) == set() and spec_learn_positions(64, 1000) == set()
    assert spec_save_points(1000, 0, set()) == []  # short prompt, no chunk boundary: no split
    assert spec_save_points(5000, 0, set()) == [4096]  # last full-chunk boundary
    assert spec_save_points(10000, 0, set()) == [8192, 9984]  # stride point + end-of-prompt
    assert spec_save_points(20000, 128, {256}) == [256, 8320, 16512, 19968]
    assert spec_save_points(1000, 0, {128, 256, 2000}) == [128, 256]  # learn points beyond L-3 dropped


class _Cache:
    def __init__(self, entries):
        self.entries, self.queries = entries, []

    def lookup(self, tokens, max_pos):
        self.queries.append(max_pos)
        ok = [p for p in self.entries if p <= max_pos]
        return (max(ok), 7) if ok else (0, None)


def _plan(cache, start, L=3000, B=4):
    m = types.SimpleNamespace(
        _prefix_cache=cache, args=types.SimpleNamespace(max_batch_size=B), _chunked_chunk_size=2048
    )
    return Qwen36Model.plan_spec_prefix(m, list(range(L)), L, start)


def test_plan_picks_snapshot_below_cap():
    c = _Cache([128, 256, 384])
    s, src, pts = _plan(c, 256)
    assert c.queries == [128] and s == 128 and src == (c, 7)  # NOT the snapshot at 256 == start
    assert 256 in pts  # learn at start for longer hits
    s, src, pts = _plan(_Cache([128, 256, 384]), 384 + 64)
    assert s == 384
    assert _plan(_Cache([]), 256)[:2] == (0, None)  # no snapshot: full recompute (+ learn points)


def test_plan_none_cases():
    assert _plan(None, 256) is None  # no cache
    assert _plan(_Cache([128]), None) is None  # caller not part of prefix caching
    assert _plan(_Cache([128]), 256, B=1) is None  # B == 1 keeps recomputing from 0
    assert _plan(_Cache([]), 0, L=1000) is None  # short prompt, no hit, nothing to learn


class _Stop(Exception):
    pass


def _engine_with_fake_model(plan_ret):
    seen = {}
    row = torch.zeros(1, 8, dtype=torch.int32)

    def plan_spec_prefix(tokens, L, start):
        seen["plan_args"] = (tokens.tolist(), L, start)
        return plan_ret

    def prefill_for_spec(prompt, pt, Tp, on_chunk, slot=0, **kw):
        seen["pfs"] = (Tp, slot, kw)
        raise _Stop("stop")

    model = types.SimpleNamespace(
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt, dtype=torch.int32),
        plan_spec_prefix=plan_spec_prefix,
        prefill_for_spec=prefill_for_spec,
    )
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e.block_size, e.nb = 2, 2, 3, 64, 8
    e._offered, e._no_draft, e._speculated, e._withheld_streak = set(), set(), set(), {}
    e._parked, e._prompts, e._fresh_dirty = {}, {}, set()
    e._last_hidden, e._scratch_block, e.model = [None, None], 10_000, model
    e.prefill_recorded = False
    return e, row, seen


def test_engine_threads_start_pos_and_replays_it(expect_error):
    plan = (128, ("cache", 3), [256])
    e, row, seen = _engine_with_fake_model(plan)
    with expect_error(_Stop, "stop"):
        e.prefill(1, list(range(300)), row, start_pos=192)
    assert seen["plan_args"][1:] == (300, 192) and seen["pfs"][2] == {"plan": plan}
    assert e._prompts[1][2] == 192
    # _reprefill replays with the same start_pos
    calls = []
    e.prefill = lambda slot, prompt, r, start_pos=None: calls.append((slot, start_pos))
    e._reprefill(1)
    assert calls == [(1, 192)]


def test_engine_without_start_pos_is_unchanged(expect_error):
    e, row, seen = _engine_with_fake_model((128, None, []))
    with expect_error(_Stop, "stop"):
        e.prefill(0, list(range(300)), row)
    assert "plan_args" not in seen and seen["pfs"][2] == {}
    assert len(e._prompts[0]) == 2
    calls = []
    e.prefill = lambda slot, prompt, r, **kw: calls.append(kw)
    e._reprefill(0)
    assert calls == [{}]


def test_engine_no_plan_calls_without_plan_kw(expect_error):
    e, row, seen = _engine_with_fake_model(None)
    with expect_error(_Stop, "stop"):
        e.prefill(0, list(range(300)), row, start_pos=0)
    assert seen["plan_args"][2] == 0 and seen["pfs"][2] == {}
