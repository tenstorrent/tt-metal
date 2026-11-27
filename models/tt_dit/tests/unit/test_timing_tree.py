# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python tests for the timing tree. No device: the sync is stubbed out.

The ledger's whole job is to survive call sites it does not control: spans that close out of order,
spans that never close, a decode that raises mid-flight. Those paths are what these tests pin --
plus the one invariant every reported number rests on, that self-times partition the root exactly.
The ``span`` helper is tested against a stubbed ``synchronize_device``: what they must
get right is the gate, the two syncs and the abort path, none of which needs hardware.
"""

from __future__ import annotations

import pytest

from models.tt_dit.utils import timing_tree as dt


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
    assert dt._STACK == []  # a raise must not leave the stack deeper than it started


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


def test_rows_carry_the_category_the_rollup_charges_them_to():
    """The tree names labels, the roll-up names categories; the column is what joins the two views."""
    root = dt.open_span("decode", root=True)
    attn = dt.open_span("attention", category=dt.ATTENTION)
    kv = dt.open_span("kv-allgather", category=dt.ALLGATHER)
    _close(kv, 30.0)
    _close(attn, 100.0)
    _close(root, 120.0)

    by_label = {lbl.strip(): cat for lbl, _, _, _, _, cat in dt._rows("decode", [dt.roots()[-1]], 120.0, 120.0)}
    assert by_label["decode"] == "-"  # uncategorised -> the roll-up's "other" row
    assert by_label["├─ attention"] == dt.ATTENTION
    assert by_label["│  ├─ kv-allgather"] == dt.ALLGATHER
    # the remainder row carries its PARENT's category, because that is where its ms are charged
    assert by_label["│  └─ · other (unattributed)"] == dt.ATTENTION


def test_live_lines_stream_progress(monkeypatch, capsys):
    """TT_DIT_STAGE_LOG: one line per open and per close, depth as indent, and a hang -- a span
    that never closes -- leaves a ">" with no "<", which is the whole point."""
    monkeypatch.setattr(dt, "LIVE", True)
    root = dt.open_span("decode", root=True)
    attn = dt.open_span("attention", category=dt.ATTENTION)
    dt.open_span("kv-allgather", category=dt.ALLGATHER)  # never closes: the hang
    _close(attn, 12.34)
    _close(root, 20.0)
    lines = [ln.split("] ", 1)[1] for ln in capsys.readouterr().out.splitlines()]
    assert lines == [
        "> decode",
        "  > attention",
        "    > kv-allgather",
        "  < attention  12.3 ms",
        "< decode  20.0 ms",
    ]
    assert dt.roots()[-1].children[0].children[0].flags == {"unclosed"}


def test_live_off_prints_nothing(capsys):
    root = dt.open_span("decode", root=True)
    _close(root, 1.0)
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------------------------ span / timed


class _FakeDevice:
    pass


@pytest.fixture
def syncs(monkeypatch):
    """Record every synchronize_device call instead of touching a device."""
    calls = []
    monkeypatch.setattr(dt.ttnn, "synchronize_device", calls.append)
    return calls


def test_span_records_one_node_between_two_syncs(syncs):
    dev = _FakeDevice()
    with dt.span(dev, "decode", root=True):
        pass
    (node,) = dt.roots()
    assert node.label == "decode" and node.incl_ms >= 0.0 and not node.flags
    assert syncs == [dev, dev]


def test_span_is_inert_when_disabled(monkeypatch, syncs):
    monkeypatch.setattr(dt, "ENABLED", False)
    with dt.span(_FakeDevice(), "decode", root=True):
        pass
    assert dt.roots() == [] and syncs == []


def test_deep_span_needs_deep(monkeypatch, syncs):
    monkeypatch.setattr(dt, "DEEP", False)
    with dt.span(_FakeDevice(), "shallow", root=True):
        with dt.span(_FakeDevice(), "deep", deep=True):
            pass
    (root,) = dt.roots()
    assert root.children == [] and len(syncs) == 2

    monkeypatch.setattr(dt, "DEEP", True)
    with dt.span(_FakeDevice(), "shallow", root=True):
        with dt.span(_FakeDevice(), "deep", deep=True):
            pass
    root = dt.roots()[-1]
    assert [c.label for c in root.children] == ["deep"]


def test_span_aborts_on_raise_and_propagates(syncs, expect_error):
    with expect_error(RuntimeError, "boom"):
        with dt.span(_FakeDevice(), "decode", root=True):
            with dt.span(_FakeDevice(), "stage"):
                raise RuntimeError("boom")
    (root,) = dt.roots()
    assert "aborted" in root.flags and "aborted" in root.children[0].flags
    assert dt._STACK == []


class _Model:
    def __init__(self, device):
        self.mesh_device = device

    @dt.span("mesh_device", "decode", root=True)
    def decode(self, x):
        """the docstring"""
        return x + 1

    @dt.span("mesh_device", lambda self, stage, *a, **k: f"stage {stage}", category=dt.SETUP)
    def stage(self, stage):
        return stage


def test_decorator_resolves_device_and_label_from_the_call(syncs):
    dev = _FakeDevice()
    model = _Model(dev)
    with dt.span(dev, "decode", root=True):
        assert model.stage(3) == 3
    (root,) = dt.roots()
    (child,) = root.children
    assert child.label == "stage 3" and child.category == dt.SETUP
    assert syncs == [dev] * 4


def test_decorator_root_and_wraps(syncs):
    model = _Model(_FakeDevice())
    assert model.decode(1) == 2
    (root,) = dt.roots()
    assert root.label == "decode" and root.parent is None
    assert _Model.decode.__name__ == "decode" and _Model.decode.__doc__ == "the docstring"


def test_decorator_is_inert_when_disabled(monkeypatch, syncs):
    monkeypatch.setattr(dt, "ENABLED", False)
    model = _Model(_FakeDevice())
    assert model.decode(1) == 2
    assert dt.roots() == [] and syncs == []


def test_decorator_device_may_be_a_callable(syncs):
    dev = _FakeDevice()

    @dt.span(lambda q, *a, **k: q.device, "op", root=True)
    def op(q):
        return q

    class _Q:
        device = dev

    op(_Q())
    (root,) = dt.roots()
    assert root.label == "op" and syncs == [dev, dev]
