"""enumerate_graduated must recognise SHARDED graduation, not only native best_pcc.

The tensor-parallel sharded graduation path writes `.py.last_good_sharded` once the
gathered-PCC shard gate passes, but returns before persisting a best_pcc number. A
best_pcc-only gate therefore reported a fully sharded-graduated model as "no graduated
modules found", while every other detector in the tool (all disk-based, shard-aware)
saw it graduated. enumerate_graduated now reuses the shared disk detector, so it stays
in step with bring-up/categorization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.commands.module_optimize import enumerate_graduated

_NATIVE = "import ttnn\n\n\nclass T:\n    def __init__(self, d, m):\n        pass\n\n    def __call__(self, x):\n        return ttnn.clone(x)\n\n\ndef build(d, m):\n    return T(d, m)\n"


def _demo(tmp_path):
    d = tmp_path / "demo"
    (d / "_stubs").mkdir(parents=True)
    return d


def _grad_stub(d, comp, suffix):
    (d / "_stubs" / f"{comp}.py").write_text(_NATIVE)
    (d / "_stubs" / f"{comp}.py{suffix}").write_text(_NATIVE)


def test_sharded_snapshot_counts_without_best_pcc(tmp_path):
    d = _demo(tmp_path)
    _grad_stub(d, "mo_e", ".last_good_sharded")
    _grad_stub(d, "m_l_p", ".last_good_sharded")
    (d / ".bringup_cc_state.json").write_text(json.dumps({"shard_reset_done": True}))
    assert enumerate_graduated(d) == ["m_l_p", "mo_e"]


def test_native_snapshot_counts_without_best_pcc(tmp_path):
    d = _demo(tmp_path)
    _grad_stub(d, "r_m_s_norm", ".last_good_native")
    (d / ".bringup_cc_state.json").write_text("{}")
    assert enumerate_graduated(d) == ["r_m_s_norm"]


def test_best_pcc_gate_still_applies(tmp_path):
    d = _demo(tmp_path)
    (d / ".bringup_cc_state.json").write_text(json.dumps({"best_pcc": {"foo": 0.999, "bar": 0.98}}))
    assert enumerate_graduated(d) == ["foo"]


def test_union_of_best_pcc_and_disk(tmp_path):
    d = _demo(tmp_path)
    _grad_stub(d, "sharded_mod", ".last_good_sharded")
    (d / ".bringup_cc_state.json").write_text(json.dumps({"best_pcc": {"native_mod": 0.995}}))
    assert enumerate_graduated(d) == ["native_mod", "sharded_mod"]


def test_fallback_excluded_even_with_snapshot(tmp_path):
    d = _demo(tmp_path)
    _grad_stub(d, "att", ".last_good_sharded")
    (d / ".bringup_cc_state.json").write_text(json.dumps({"fallback": ["att"]}))
    assert enumerate_graduated(d) == []


def test_helper_stubs_ignored(tmp_path):
    d = _demo(tmp_path)
    _grad_stub(d, "_tt_common", ".last_good_sharded")
    _grad_stub(d, "perf_kernels", ".last_good_sharded")
    (d / ".bringup_cc_state.json").write_text("{}")
    assert enumerate_graduated(d) == []


def test_snapshot_absent_not_graduated(tmp_path):
    d = _demo(tmp_path)
    (d / "_stubs" / "no_snap.py").write_text(_NATIVE)
    (d / ".bringup_cc_state.json").write_text("{}")
    assert enumerate_graduated(d) == []


def test_empty(tmp_path):
    d = _demo(tmp_path)
    assert enumerate_graduated(d) == []
