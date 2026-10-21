# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F76: the dashboard shows a '>= limit' gate against the limit its run recorded, never the word 'limit'."""

from models.demos.common.bringup.core.gate import state_metrics
from models.demos.common.bringup.dashboard.export import red_passed, resolve_thresholds


def test_a_limit_gate_shows_the_recorded_limit():
    thr = {"pcc_swap_out": ">= limit", "x": "== 0"}
    out, note = resolve_thresholds(thr, {"pcc_swap_out": 0.9995, "pcc_swap_out_limit": 0.9923, "x": 0}, {})
    assert out == {"pcc_swap_out": ">= 0.9923", "x": "== 0"} and note is None
    out, _ = resolve_thresholds(thr, {"pcc_swap_out": 0.9995}, {"pcc_swap_out_limit": {"value": 0.99}})
    assert out["pcc_swap_out"] == ">= 0.99"


def test_a_step_gated_before_the_limits_shows_no_bar_and_a_pending_one_keeps_its_glob():
    out, note = resolve_thresholds({"pcc_a_L*": ">= limit"}, {"pcc_a_L00": 0.999}, {})
    assert out == {} and "F76" in note
    out, note = resolve_thresholds({"pcc_a_L*": ">= limit"}, {}, {})
    assert out == {} and note is None  # pending: nothing to show yet, never the word 'limit'


def test_a_step_passed_under_an_earlier_rule_is_not_red():
    # F81 renamed the gate (pcc_* -> err_*): a step passed before it has only pcc_* metrics
    vals = {"pcc_q_a_L00": 0.99997}
    out, note = resolve_thresholds({"err_q_a_L*": "<= limit"}, vals, {}, "PASS")
    assert out == {} and "earlier rule" in note
    assert red_passed([{"id": "C.q_a", "status": "PASS", "thresholds": out, "metrics": vals}]) == []


def test_red_passed_catches_what_the_page_would_paint_red():
    t = {"id": "C.x", "status": "PASS", "metrics": {"err_x_L00": 0.002}}
    assert red_passed([{**t, "thresholds": {"err_x_L*": "<= limit"}}]) == ["C.x: err_x_L* '<= limit' is not a number"]
    assert red_passed([{**t, "thresholds": {"err_x_L*": "<= 0.001"}}]) == ["C.x: err_x_L00 = 0.002 fails <= 0.001"]
    assert red_passed([{**t, "thresholds": {"err_y": "<= 0.1"}}]) == ["C.x: err_y missing"]
    assert red_passed([{**t, "thresholds": {"err_x_L*": "<= 0.01"}}]) == []


def test_state_keeps_the_limits_of_gated_metrics():
    got = {f"m{i}": {"value": i} for i in range(80)}
    got.update(pcc_swap_out={"value": 0.999}, pcc_swap_out_limit={"value": 0.99})
    kept = state_metrics({"pcc_swap_out": ">= limit"}, got)
    assert kept == {"pcc_swap_out": 0.999, "pcc_swap_out_limit": 0.99}


def test_state_keeps_only_gated_metrics_even_when_few():
    """F129: a gate with a few informational metrics no longer copies them into state.json (results/ has them)."""
    got = {"pcc_layer_L00": {"value": 0.99}, "pcc_layer_L00_limit": {"value": 0.98}, "chunk_seconds_c00": {"value": 1}}
    assert state_metrics({"pcc_layer_L*": ">= limit"}, got) == {"pcc_layer_L00": 0.99, "pcc_layer_L00_limit": 0.98}
