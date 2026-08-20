# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python tests for the decode timing ledger. No device, no ttnn.

The ledger's whole job is to survive call sites it does not control: spans that close out of order,
spans that never close, a decode that raises mid-flight. Those paths are what these tests pin --
plus the one invariant every reported number rests on, that self-times partition the root exactly.
"""

from __future__ import annotations

import pytest

from models.tt_dit.utils import decode_tree as dt


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    """ENABLED is an import-time constant (it gates device syncs in the real helpers), so tests set
    it directly rather than through the environment."""
    monkeypatch.setattr(dt, "ENABLED", True)
    dt.reset()
    yield
    dt.reset()


def _close(span, ms):
    dt.close_span(span, ms)


def test_nesting_and_self_time():
    root = dt.open_span("decode", root=True)
    stage = dt.open_span("stage", category=None)
    attn = dt.open_span("attention", category=dt.ATTENTION)
    kv = dt.open_span("kv-allgather", category=dt.ALLGATHER)
    _close(kv, 40.0)
    _close(attn, 100.0)
    _close(stage, 150.0)
    _close(root, 200.0)

    (node,) = dt.roots()
    assert node.label == "decode" and node.incl_ms == 200.0
    stage_node = node.children[0]
    attn_node = stage_node.children[0]
    # inclusive time nests; self time is what the level itself did not hand to a child
    assert attn_node.incl_ms == 100.0
    assert attn_node.self_ms == 60.0  # 100 attention - 40 kv-allgather
    assert stage_node.self_ms == 50.0
    assert node.self_ms == 50.0


def test_self_times_partition_the_root():
    """The roll-up's correctness claim: every ms is charged to exactly one node."""
    root = dt.open_span("decode", root=True)
    a = dt.open_span("a", category=dt.ATTENTION)
    b = dt.open_span("kv", category=dt.ALLGATHER)
    _close(b, 30.0)
    _close(a, 70.0)
    c = dt.open_span("mlp", category=dt.MLP)
    _close(c, 20.0)
    _close(root, 120.0)

    (node,) = dt.roots()
    totals, _ = dt.category_totals(node)
    assert sum(totals.values()) == pytest.approx(node.incl_ms)
    assert totals[dt.ALLGATHER] == 30.0
    assert totals[dt.ATTENTION] == 40.0  # 70 - 30 handed to the child
    assert totals[dt.MLP] == 20.0
    assert totals["other (uncategorized)"] == 30.0  # the root's own unattributed remainder


def test_siblings_pool_by_exact_label_at_render():
    """attention fires once per band; those two spans are one row with n=2. Distinctly labelled
    siblings (block 0 vs block 1) stay distinct rows -- nothing is merged for containing a number."""
    root = dt.open_span("decode", root=True)
    for block in range(2):
        blk = dt.open_span(f"block {block}")
        for _band in range(2):
            attn = dt.open_span("attention", category=dt.ATTENTION)
            _close(attn, 10.0)
        _close(blk, 25.0)
    _close(root, 60.0)

    (node,) = dt.roots()
    rows = dt._rows(node.label, [node], node.incl_ms, node.incl_ms)
    labels = [r[0].strip() for r in rows]
    assert any(lbl.endswith("block 0") for lbl in labels)
    assert any(lbl.endswith("block 1") for lbl in labels)  # not collapsed together
    attn_rows = [r for r in rows if r[0].strip().endswith("attention")]
    assert len(attn_rows) == 2  # one per block
    assert all(r[4] == 2 for r in attn_rows)  # each pools the two bands
    assert all(r[1] == 20.0 for r in attn_rows)


def test_out_of_order_close_marks_the_orphan_and_keeps_the_parent_honest():
    root = dt.open_span("decode", root=True)
    outer = dt.open_span("outer")
    inner = dt.open_span("inner-that-leaks")
    _close(outer, 100.0)  # inner never closed
    _close(root, 120.0)

    (node,) = dt.roots()
    outer_node = node.children[0]
    orphan = outer_node.children[0]
    assert "unclosed" in orphan.flags
    assert orphan.label == "inner-that-leaks"  # named, because the label was set at open
    assert orphan.incl_ms == 0.0
    assert outer_node.self_ms == 100.0  # the leaked span claims none of the parent's time
    dt.close_span(inner, 999.0)  # late close after its parent: dropped, not applied
    assert orphan.incl_ms == 0.0


def test_double_close_is_dropped():
    root = dt.open_span("decode", root=True)
    child = dt.open_span("child")
    _close(child, 10.0)
    _close(child, 999.0)  # second close: must not re-enter the stack or change the node
    _close(root, 20.0)

    (node,) = dt.roots()
    assert node.children[0].incl_ms == 10.0
    assert dt.root_count() == 1


def test_abort_unwinds_and_still_records_a_marked_root():
    root = dt.open_span("decode", root=True)
    stage = dt.open_span("stage")
    dt.open_span("attention", category=dt.ATTENTION)  # still open when the decode raises
    dt.abort_span(stage)
    dt.abort_span(root)

    (node,) = dt.roots()
    assert "aborted" in node.flags
    assert "unclosed" in node.children[0].children[0].flags
    assert dt._stack() == []  # a raise must not leave the stack deeper than it started


def test_new_root_resets_a_dirty_stack():
    """Two decodes in one process: the second must not inherit the first's half-open spans."""
    first = dt.open_span("decode", root=True)
    dt.open_span("leaked")
    _close(first, 10.0)

    second = dt.open_span("decode", root=True)
    child = dt.open_span("clean")
    _close(child, 3.0)
    _close(second, 5.0)

    assert dt.root_count() == 2
    latest = dt.roots()[-1]
    assert [c.label for c in latest.children] == ["clean"]


def test_disabled_is_inert(monkeypatch):
    monkeypatch.setattr(dt, "ENABLED", False)
    span = dt.open_span("decode", root=True)
    assert span is None
    dt.close_span(span, 5.0)  # must tolerate None without a guard at the call site
    dt.abort_span(span)
    assert dt.root_count() == 0


def test_render_reports_the_remainder_and_reconciles():
    root = dt.open_span("decode", root=True)
    attn = dt.open_span("attention", category=dt.ATTENTION)
    kv = dt.open_span("kv-allgather", category=dt.ALLGATHER)
    _close(kv, 40.0)
    _close(attn, 100.0)
    _close(root, 200.0)

    text = dt.render(dt.roots()[-1], title="unit", measured_ms=199.0)
    assert "· other (unattributed)" in text  # attention's 60 ms that kv-allgather does not explain
    assert "CATEGORY ROLL-UP" in text
    assert "test-measured 199 ms" in text
