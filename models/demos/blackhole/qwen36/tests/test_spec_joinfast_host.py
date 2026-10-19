# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_SPEC_JOIN_FAST=1 (MTPSpecEngine._seed_fresh_rows host path, work/notes/55_mtp_join_fast.md):
the join stages only the fresh rows' MTP page tables and warms their slot Tp-1 MTP KV in one batched drafter call.
No device: the staged tensors / indices of the new path are compared with the old path using fakes
(ttnn tensor ops are monkeypatched; nothing opens a mesh).

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_joinfast_host.py -q
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine, join_warm_plan
from models.demos.blackhole.qwen36.tt.tp_common import spec_join_fast_enabled

NB = 16


class _Buf:
    def __init__(self, name):
        self.name, self.content = name, None


class _Model:
    """model.ttnn_mtp_decode_forward stand-in: records (hidden rows, tokens, positions, page table rows)."""

    def __init__(self):
        self.decode_calls = []

    def ttnn_mtp_decode_forward(self, hidden, token_id, position, page_table, need_logits=True):
        assert need_logits is False
        self.decode_calls.append((hidden, token_id, position, page_table))
        return None, "h_next"


class _Gdn:
    def seed_spec_rows_select(self, rows, sel):
        pass


def _engine(B, K=3):
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e.block_size, e.nb = B, K, K + 1, 64, NB
    e._fresh, e._prompts, e._fresh_dirty = {}, {}, set()
    e._gdn, e._kc, e._fj_sel, e.mesh = [], 4, "SEL", "MESH"
    e.copies = []
    e._mtp_pt = _Buf("pt")
    e._mtp_pt_rows = [_Buf(f"row{u}") for u in range(B)]
    e._last_hidden = [f"h{u}" for u in range(B)]
    e.model = _Model()
    e.warm_single = []
    e.nb = NB

    def stage(host, dev, dtype, layout):
        e.copies.append(dev.name)
        dev.content = host.clone()

    e._stage = stage
    e._warm_mtp_last = lambda h, tok, slot, pt: e.warm_single.append(
        (h, int(tok), int(slot), pt.content.clone().tolist())
    )
    return e


@pytest.fixture
def fake_ttnn(monkeypatch):
    """Rows are tracked as Python lists of row names: to_layout keeps them, concat joins them, from_torch keeps the torch
    table."""
    freed = []
    monkeypatch.setattr(ttnn, "to_layout", lambda t, layout: t)
    monkeypatch.setattr(ttnn, "concat", lambda ts, dim: [r for t in ts for r in (t if isinstance(t, list) else [t])])
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: "REP")
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: t.clone())
    monkeypatch.setattr(ttnn, "deallocate", lambda t: freed.append(t))
    return freed


def _case(B, fresh_rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 1000, (B, 4), generator=g)
    pt = torch.randint(0, 5000, (B, NB), generator=g).to(torch.int32)
    fresh_now = {u: 30 + 7 * u for u in fresh_rows}
    start = torch.tensor([fresh_now.get(u, 100 + u) for u in range(B)])
    return tokens, start, pt, fresh_now


def _run(monkeypatch, flag, B, fresh_rows, rows=None, pre=None):
    monkeypatch.setenv("QWEN36_SPEC_JOIN_FAST", flag)
    e = _engine(B)
    tokens, start, pt, fresh_now = _case(B, fresh_rows)
    e._fresh = dict(fresh_now)
    if pre is not None:  # device already holds the previous step's tables
        MTPSpecEngine._stage_mtp_tables(e, pre)
        e.copies.clear()
    out = MTPSpecEngine._seed_fresh_rows(e, tokens, start, pt, fresh_now, list(range(B)) if rows is None else rows)
    return e, out, tokens, pt, fresh_now


def test_flag_read(monkeypatch):
    for val, want in (("1", True), ("0", False), (None, True)):
        if val is None:
            monkeypatch.delenv("QWEN36_SPEC_JOIN_FAST", raising=False)
        else:
            monkeypatch.setenv("QWEN36_SPEC_JOIN_FAST", val)
        assert spec_join_fast_enabled() is want


@pytest.mark.parametrize("fresh", [[0], [3], [1, 4], [0, 2, 5, 7], list(range(8))])
def test_join_warm_plan_equals_per_row_inputs(fresh):
    B = 8
    tokens, start, pt, fresh_now = _case(B, fresh, seed=len(fresh))
    rows, toks, slots, pt_rows = join_warm_plan(fresh_now, tokens, pt)
    assert rows == sorted(fresh) and pt_rows.dtype == torch.int32 and pt_rows.is_contiguous()
    for i, u in enumerate(rows):  # what the old per-row call receives
        assert toks[i] == int(tokens[u, 0]) and slots[i] == fresh_now[u] - 1
        assert torch.equal(pt_rows[i : i + 1], pt[u : u + 1])


@pytest.mark.parametrize("fresh", [[0], [5], [1, 4], [0, 2, 5, 7], list(range(8))])
def test_new_path_warms_same_rows_as_old_path(monkeypatch, fake_ttnn, fresh):
    """Old: one (hidden, token, slot, table) call per fresh row. New: one call for R >= 2 whose rows equal those calls in
    order, the single-row call for R == 1."""
    B = 8
    old, old_out, tokens, pt, fresh_now = _run(monkeypatch, "0", B, fresh)
    new, new_out, *_ = _run(monkeypatch, "1", B, fresh)
    assert old_out == new_out == sorted(fresh)
    assert old.model.decode_calls == []
    assert [(h, t, s) for h, t, s, _ in old.warm_single] == [
        (f"h{u}", int(tokens[u, 0]), fresh_now[u] - 1) for u in sorted(fresh)
    ]
    if len(fresh) == 1:
        assert new.warm_single == old.warm_single and new.model.decode_calls == []
    else:
        assert new.warm_single == [] and len(new.model.decode_calls) == 1
        hid, toks, slots, pt_tt = new.model.decode_calls[0]
        assert hid == [f"h{u}" for u in sorted(fresh)]
        assert toks == [t for _, t, _, _ in old.warm_single] and slots == [s for _, _, s, _ in old.warm_single]
        for i, (_, _, _, row_pt) in enumerate(old.warm_single):
            assert pt_tt[i : i + 1].tolist() == row_pt
        assert pt_tt.shape == (len(fresh), NB) and pt_tt.dtype == torch.int32
        assert len(fake_ttnn) >= 3  # temporaries freed
    # bookkeeping identical
    assert old._fresh == new._fresh == {} and old._prompts == new._prompts == {}


@pytest.mark.parametrize("fresh", [[2], [1, 4], [0, 2, 5, 7]])
def test_new_path_stages_only_fresh_rows_with_old_contents(monkeypatch, fake_ttnn, fresh):
    """Staged tensors: every fresh row's [1, nb] table equals the old path's; old path also copies the [B, nb] table and
    all changed rows, the new one only the fresh rows."""
    B = 8
    old, *_ = _run(monkeypatch, "0", B, fresh)
    new, *_ = _run(monkeypatch, "1", B, fresh)
    for u in fresh:
        assert torch.equal(new._mtp_pt_rows[u].content, old._mtp_pt_rows[u].content)
    assert sorted(new.copies) == sorted(f"row{u}" for u in fresh)
    assert len(old.copies) == 1 + B and new.copies and len(new.copies) < len(old.copies)
    # rows left out keep what they held (nothing staged); the next propose stages them (content keyed)
    for u in set(range(B)) - set(fresh):
        assert new._mtp_pt_rows[u].content is None


def test_new_path_second_stage_by_propose_matches_old_state(monkeypatch, fake_ttnn):
    """After the join the next propose calls _stage_mtp_tables(pt): with the new path every buffer then holds the same
    content as with the old path."""
    B, fresh = 8, [1, 4]
    old, _, _, pt, _ = _run(monkeypatch, "0", B, fresh)
    new, *_ = _run(monkeypatch, "1", B, fresh)
    for e in (old, new):
        MTPSpecEngine._stage_table(e, pt, e._mtp_pt, None, None)
        for u in range(B):
            MTPSpecEngine._stage_table(e, pt[u : u + 1].contiguous(), e._mtp_pt_rows[u], None, None)
    assert torch.equal(old._mtp_pt.content, new._mtp_pt.content)
    for u in range(B):
        assert torch.equal(old._mtp_pt_rows[u].content, new._mtp_pt_rows[u].content)


def test_start_pos_mismatch_still_asserts(monkeypatch, fake_ttnn, expect_error):
    monkeypatch.setenv("QWEN36_SPEC_JOIN_FAST", "1")
    e = _engine(4)
    tokens, start, pt, fresh_now = _case(4, [1, 2])
    start[2] += 1
    with expect_error(AssertionError, "start_pos"):
        MTPSpecEngine._seed_fresh_rows(e, tokens, start, pt, fresh_now, [0, 1, 2, 3])


def test_warmup_compiles_every_batched_row_count(monkeypatch, fake_ttnn):
    """Before capture the engine runs the R-row batched warm for R = 2 .. B over the scratch block (the device tests run
    with program-cache misses disallowed, so a join must never see a new row count); flag 0 or B = 1 warms nothing."""
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    B = 4
    monkeypatch.setenv("QWEN36_SPEC_JOIN_FAST", "1")
    e = _engine(B)
    e._scratch_block = 99
    MTPSpecEngine._join_warm_warmup(e)
    assert [len(h) for h, *_ in e.model.decode_calls] == [2, 3, 4]
    for _, toks, slots, pt_tt in e.model.decode_calls:
        assert set(toks) == {0} and set(slots) == {0} and bool((pt_tt == 99).all())
    monkeypatch.setenv("QWEN36_SPEC_JOIN_FAST", "0")
    e2 = _engine(B)
    MTPSpecEngine._join_warm_warmup(e2)
    assert e2.model.decode_calls == []
