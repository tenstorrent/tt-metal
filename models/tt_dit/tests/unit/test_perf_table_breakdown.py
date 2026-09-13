# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the LTX perf table expands its VAE decode row from a recorded decode tree."""

from __future__ import annotations

import pytest

from models.tt_dit.utils import decode_tree as dt
from models.tt_dit.utils.ltx import decode_breakdown_rows, decode_category_rows, print_ltx_timing_table


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(dt, "ENABLED", True)
    dt.reset()
    yield
    dt.reset()


def _tree():
    """decode TOTAL 10 s = det 4 s (+ stage 1..2) + stage5 5.5 s (blocks, each attention+mlp) + 0.5 s self."""
    root = dt.open_span("decode TOTAL", root=True)
    det = dt.open_span("det stages TOTAL (forward_context)")
    for i, ms in ((1, 1500.0), (2, 2500.0)):
        s = dt.open_span(f"det stage {i}", category=dt.MLP)
        dt.close_span(s, ms)
    dt.close_span(det, 4000.0)
    s5 = dt.open_span("stage5 TOTAL (forward)")
    for i in range(2):
        b = dt.open_span(f"  stage5 block {i}")
        a = dt.open_span("attention", category=dt.ATTENTION)
        ag = dt.open_span("kv-allgather", category=dt.ALLGATHER)
        dt.close_span(ag, 500.0)
        dt.close_span(a, 2000.0)
        m = dt.open_span("mlp", category=dt.MLP)
        dt.close_span(m, 700.0)
        dt.close_span(b, 2750.0)
    dt.close_span(s5, 5500.0)
    dt.close_span(root, 10000.0)
    return dt.roots()[-1]


class _Pipe:
    def __init__(self, root):
        self.last_timings = [("Stage 1 denoise", 2.0), ("VAE decode", 9.0), ("Audio decode", 0.5)]
        self.last_decode_tree = root


def _print(monkeypatch, capsys, depth, root=None):
    monkeypatch.setenv("LTX_PERF_BREAKDOWN", str(depth))
    print_ltx_timing_table(
        _Pipe(root if root is not None else _tree()),
        label="T",
        num_frames=1,
        height=32,
        width=32,
        mesh_shape=(1, 1),
        sp_axis=0,
        tp_axis=1,
        topology="Ring",
        output_path="x.mp4",
        prompt="p",
    )
    return capsys.readouterr().out


def test_breakdown_levels_sum_to_parent():
    root = _tree()
    rows = decode_breakdown_rows(root, 9.0, depth=2)
    labels = [lbl for lbl, _ in rows]
    assert labels[0].startswith("  · tree total") and "+1.00 s vs wall" in labels[0]
    level1 = {lbl.strip("│ ├└─"): s for lbl, s in rows if lbl.startswith("  ├─") or lbl.startswith("  └─")}
    assert level1["det stages TOTAL (forward_context)"] == pytest.approx(4.0)
    assert level1["stage5 TOTAL (forward)"] == pytest.approx(5.5)
    assert level1["· other"] == pytest.approx(0.5)
    assert sum(level1.values()) == pytest.approx(10.0)
    # depth 2 reaches the blocks, not the attention/mlp inside them
    assert any("stage5 block 1" in lbl for lbl in labels)
    assert not any("attention" in lbl for lbl in labels)


def test_depth_zero_and_missing_tree_keep_flat_table(monkeypatch, capsys):
    out = _print(monkeypatch, capsys, 0)
    assert "tree total" not in out and "det stages" not in out
    monkeypatch.setenv("LTX_PERF_BREAKDOWN", "2")
    pipe = _Pipe(None)
    print_ltx_timing_table(
        pipe,
        label="T",
        num_frames=1,
        height=32,
        width=32,
        mesh_shape=(1, 1),
        sp_axis=0,
        tp_axis=1,
        topology="Ring",
        output_path="x.mp4",
        prompt="p",
    )
    out = capsys.readouterr().out
    assert "VAE decode" in out and "tree total" not in out
    assert "│ Total" in out and "11.50 s" in out


def test_table_shows_subrows_and_total_excludes_them(monkeypatch, capsys):
    out = _print(monkeypatch, capsys, 1)
    assert "├─ det stages TOTAL (forward_context)" in out
    assert "stage5 block" not in out  # depth 1 only
    assert "11.50 s" in out  # 2.0 + 9.0 + 0.5, sub-rows not double counted
    # every line of the box has the same width
    box = [l for l in out.splitlines() if l.startswith(("┌", "│", "├", "└"))]
    assert len({len(l) for l in box}) == 1


def test_category_block_only_under_block_prof(monkeypatch, capsys):
    out = _print(monkeypatch, capsys, 1)
    assert "by category" not in out
    monkeypatch.setattr(dt, "DEEP", True)
    out = _print(monkeypatch, capsys, 1)
    assert "VAE decode by category" in out
    rows = dict((c, (s, p)) for c, s, p in decode_category_rows(_tree()))
    # exclusive: attention self = 2 x (2000-500) = 3.0 s, allgather = 1.0 s, mlp = 4.0 (det) + 1.4 = 5.4 s
    assert rows["attention"][0] == pytest.approx(3.0)
    assert rows["allgather"][0] == pytest.approx(1.0)
    assert rows["mlp"][0] == pytest.approx(5.4)
    assert sum(s for s, _ in rows.values()) == pytest.approx(10.0)
