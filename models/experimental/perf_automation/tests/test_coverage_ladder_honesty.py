# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Issue 7: the coverage ladder reported a FAILED search as a successful measurement.

``_measure_cov`` walks rungs looking for the shallowest depth that still exercises every distinct
op type. When it finds one it returns ``(depth, [], "measured")``. When it runs out of rungs it
returned::

    return (ladder[-1] if ladder else 16), sorted(want - got), "measured"

-- the same ``"measured"`` label. So "16 covers every op type" and "16 was simply the last rung and
2 op types were never seen" are indistinguishable to every consumer, and the run proceeded on a
window that was known not to cover the model.

Since issue 1 the final rung is the model's FULL declared depth, which sharpens this considerably:
an op type cannot be absent at full depth, because full depth is the whole model. If op types are
still missing there, the depth knob is not actually slicing anything -- exactly the llama3_1_8b_p150
state, where the knob named a variable the demo never read and every rung profiled the same full
model. That deserves to be called out as a broken knob, not reported as a coverage number.
"""

import importlib.util
import json
import sys
from pathlib import Path


_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"


def _mod():
    sys.path.insert(0, str(_PA))
    spec = importlib.util.spec_from_file_location("cc_run_cov", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ALL_OPS = {"matmul", "softmax", "rmsnorm", "rare_op"}


def _wire(monkeypatch, sigs_by_depth, depth=None, tmp_path=None):
    """sigs_by_depth: depth -> set of op signatures that rung reveals."""
    m = _mod()
    seen = []

    def _probe(_repo, _env, _dev, _node, _case, d):
        seen.append(d)
        return sorted(sigs_by_depth.get(d, set())), "", []

    monkeypatch.setattr(m, "_run_op_sigs", _probe)
    root = tmp_path or Path("/nonexistent")
    if depth is not None and tmp_path is not None:
        (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": depth}))
    return m, seen, root


def _call(m, root, sigs=ALL_OPS):
    return m._measure_cov(Path("/repo"), {}, "0", "n.py::t", None, sigs, root, base_knob={"CAP": "1"})


def test_success_is_labelled_measured(tmp_path, monkeypatch):
    m, seen, root = _wire(monkeypatch, {2: {"matmul"}, 4: ALL_OPS}, depth=32, tmp_path=tmp_path)
    depth, missing, source = _call(m, root)
    assert (depth, missing, source) == (4, [], "measured")
    assert seen[: len(seen)] and 4 in seen


def test_exhausted_ladder_is_not_labelled_measured(tmp_path, monkeypatch):
    """The regression: rungs exhausted, op types still missing, reported as 'measured'.

    Uses an UNDECLARED depth so the ladder legitimately stops at 16 without that implying the knob
    is inert -- the declared-depth case is the separate, stronger verdict below.
    """
    partial = ALL_OPS - {"rare_op"}
    m, _seen, root = _wire(monkeypatch, {2: partial, 4: partial, 8: partial, 16: partial}, tmp_path=tmp_path)
    depth, missing, source = _call(m, root)
    assert missing == ["rare_op"]
    assert source != "measured", (
        "a search that never covered the model was labelled 'measured' -- indistinguishable from "
        "a real result, so the run proceeds on a window known not to cover the model"
    )


def test_declared_depth_exhausted_returns_inert_knob_not_a_number(tmp_path, monkeypatch):
    """With a declared depth the last rung IS the whole model, so missing ops mean an inert knob
    and no coverage window may be reported at all."""
    partial = ALL_OPS - {"rare_op"}
    m, _seen, root = _wire(
        monkeypatch, {2: partial, 4: partial, 8: partial, 16: partial, 32: partial}, depth=32, tmp_path=tmp_path
    )
    assert _call(m, root) is None


def test_missing_ops_at_full_depth_is_flagged_as_a_broken_knob(tmp_path, monkeypatch, capsys):
    """An op type cannot be absent at FULL depth. If it is, the knob is not slicing."""
    partial = ALL_OPS - {"rare_op"}
    m, _seen, root = _wire(
        monkeypatch, {2: partial, 4: partial, 8: partial, 16: partial, 32: partial}, depth=32, tmp_path=tmp_path
    )
    _call(m, root)
    out = capsys.readouterr().out.lower()
    assert "knob" in out and (
        "not" in out or "no" in out
    ), f"missing op types at the model's full depth were not reported as a knob failure:\n{out}"


def test_full_depth_coverage_is_a_normal_success(tmp_path, monkeypatch):
    """Covering only at the last rung is still a genuine success, not a failure."""
    m, _seen, root = _wire(
        monkeypatch,
        {2: {"matmul"}, 4: {"matmul"}, 8: {"matmul"}, 16: {"matmul"}, 32: ALL_OPS},
        depth=32,
        tmp_path=tmp_path,
    )
    depth, missing, source = _call(m, root)
    assert (depth, missing, source) == (32, [], "measured")


def test_undeclared_depth_still_distinguishes_failure(tmp_path, monkeypatch):
    """With no declared depth the ladder is 2,4,8,16 and we cannot claim the knob is broken -- but
    an exhausted search must still not be called 'measured'."""
    partial = ALL_OPS - {"rare_op"}
    m, _seen, root = _wire(monkeypatch, {2: partial, 4: partial, 8: partial, 16: partial}, tmp_path=tmp_path)
    depth, missing, source = _call(m, root)
    assert missing == ["rare_op"] and source != "measured"


def test_shallower_rung_wins(tmp_path, monkeypatch):
    m, seen, root = _wire(monkeypatch, {2: ALL_OPS, 4: ALL_OPS}, depth=32, tmp_path=tmp_path)
    depth, missing, source = _call(m, root)
    assert (depth, missing, source) == (2, [], "measured")
    assert seen == [2], "the search kept probing after it had already succeeded"


def test_empty_probe_is_still_inconclusive(tmp_path, monkeypatch):
    m, _seen, root = _wire(monkeypatch, {}, depth=32, tmp_path=tmp_path)
    assert _call(m, root) is None
