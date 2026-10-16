# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F76: the dashboard shows a '>= limit' gate against the limit its run recorded, never the word 'limit'."""

from models.demos.common.bringup.core.gate import state_metrics
from models.demos.common.bringup.dashboard.export import resolve_thresholds


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
    assert out == {"pcc_a_L*": ">= limit"} and note is None


def test_state_keeps_the_limits_of_gated_metrics():
    got = {f"m{i}": {"value": i} for i in range(80)}
    got.update(pcc_swap_out={"value": 0.999}, pcc_swap_out_limit={"value": 0.99})
    kept = state_metrics({"pcc_swap_out": ">= limit"}, got)
    assert kept == {"pcc_swap_out": 0.999, "pcc_swap_out_limit": 0.99}
