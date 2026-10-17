# SPDX-License-Identifier: Apache-2.0
"""termination_check band hook: opt-in, unit-matched scoring (per-profile floor scored against
per-profile device_ms, NOT the per-token trace), per-module uses its own floor, fail-open."""
import importlib.util
from pathlib import Path

_S = importlib.util.spec_from_file_location(
    "perf_mcp_ptterm", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py")
)
pm = importlib.util.module_from_spec(_S)
_S.loader.exec_module(pm)


def test_band_off_returns_none(monkeypatch):
    monkeypatch.delenv("PERF_MCP_TARGET_BAND", raising=False)
    assert pm._perf_target_status({"modeled_floor_ms": 2.0}, 2.2) is None


def test_per_module_scores_dev_against_floor_in_band(monkeypatch):
    # floor path scores dev (per-profile), NOT the per-token trace. dev 2.2 near the 2.0 floor -> IN_BAND
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    s = pm._perf_target_status({"modeled_floor_ms": 2.0}, 2.2)
    # a floor-derived target has no band, so there is no IN_BAND verdict to reach
    assert s["scope"] == "module" and s["status"] == "NO_BAND"


def test_per_module_below_band(monkeypatch):
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    assert pm._perf_target_status({"modeled_floor_ms": 2.0}, 3.5)["status"] == "NO_BAND"


def test_unit_match_no_false_above_band(monkeypatch):
    # the real regression: a per-profile floor scored against a per-token trace read ABOVE_BAND every
    # time. Measuring SLOWER than the floor must never read ABOVE_BAND -- and a floor target has no
    # band at all now, so the honest verdict is NO_BAND rather than a manufactured BELOW_BAND.
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    s = pm._perf_target_status({"modeled_floor_ms": 1000 / 522.821}, 3.533)
    assert s["status"] != "ABOVE_BAND", s
    assert s["status"] == "NO_BAND", s


def test_full_model_falls_back_to_floor_scores_dev(monkeypatch):
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.delenv("TT_PERF_MODULE_LEVEL", raising=False)
    monkeypatch.setattr(pm, "_load_perf_target_inputs", lambda: None)
    s = pm._perf_target_status({"modeled_floor_ms": 2.0}, 2.2)
    assert s["scope"] == "model" and s["status"] == "NO_BAND"


def test_fail_open_on_error(monkeypatch):
    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    monkeypatch.setenv("TT_PERF_MODULE_LEVEL", "1")
    monkeypatch.setattr(pm.perf_target, "score", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert pm._perf_target_status({"modeled_floor_ms": 2.0}, 2.0) is None


# --- the band must count the SAME unit of work as the measurement -----------------------------------


def _target(unit):
    from agent import perf_target as pt

    return pt.compute_target({"weight_bytes": 6_094_651_392, "unit": unit}, {"dram_bw_gbps": 512.0})


def test_the_ceiling_carries_the_unit_it_is_per():
    """peak_BW / active_bytes is a rate only if active_bytes is what ONE unit of work reads, and the
    unit differs by model: a token, a denoise step, one forward pass. Without it on the target,
    nothing downstream can check the measurement counts the same thing."""
    assert _target("token").unit == "token"
    assert _target("step").unit == "step"
    assert _target("inference").unit == "inference"
    assert _target("").unit == "token"  # absent -> the historical default


def test_a_step_ceiling_is_not_scored_against_a_token_reading(tmp_path, monkeypatch):
    """THE GAP: `is_llm` really means "a config ceiling exists", which is true for a diffusion model
    too, and the gate's reading was whatever it last measured. A per-step ceiling over a per-token
    reading is arithmetic, not a comparison -- and an IN_BAND from it would end a run at a target
    never tested."""
    import json as _json

    from cc_optimize import perf_mcp as pm

    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    base = tmp_path / "fullpipe.json"
    base.write_text(_json.dumps({"full_pipeline_ms": 17.0, "method": "trace", "mode": "trace+1cq", "unit": "token"}))
    monkeypatch.setattr(pm, "_FULLPIPE_BASELINE_1CQ_PATH", base)
    monkeypatch.setattr(pm, "_select_perf_target", lambda rep: (_target("step"), "model", True))

    s = pm._perf_target_status({"modeled_floor_ms": 100.0}, 500.0)
    assert s["status"] == "UNIT_MISMATCH", s
    assert s["target_unit"] == "step" and s["measured_unit"] == "token"


def test_a_matching_unit_still_scores_normally(tmp_path, monkeypatch):
    import json as _json

    from cc_optimize import perf_mcp as pm

    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    base = tmp_path / "fullpipe.json"
    base.write_text(_json.dumps({"full_pipeline_ms": 17.0, "method": "trace", "mode": "trace+1cq", "unit": "token"}))
    monkeypatch.setattr(pm, "_FULLPIPE_BASELINE_1CQ_PATH", base)
    monkeypatch.setattr(pm, "_select_perf_target", lambda rep: (_target("token"), "model", True))

    s = pm._perf_target_status({"modeled_floor_ms": 100.0}, 500.0)
    assert s["status"] == "IN_BAND", s


def test_a_reading_with_no_recorded_unit_is_accepted_only_for_the_token_ceiling(tmp_path, monkeypatch):
    """Every run and test predating TRACE_HEADLINE_UNIT measured per-token, so an unrecorded unit is
    treated as token -- but ONLY against a token ceiling. Against a step ceiling it still refuses."""
    import json as _json

    from cc_optimize import perf_mcp as pm

    monkeypatch.setenv("PERF_MCP_TARGET_BAND", "1")
    base = tmp_path / "fullpipe.json"
    base.write_text(_json.dumps({"full_pipeline_ms": 17.0, "method": "trace", "mode": "trace+1cq"}))
    monkeypatch.setattr(pm, "_FULLPIPE_BASELINE_1CQ_PATH", base)

    monkeypatch.setattr(pm, "_select_perf_target", lambda rep: (_target("token"), "model", True))
    assert pm._perf_target_status({"modeled_floor_ms": 100.0}, 500.0)["status"] == "IN_BAND"

    monkeypatch.setattr(pm, "_select_perf_target", lambda rep: (_target("step"), "model", True))
    assert pm._perf_target_status({"modeled_floor_ms": 100.0}, 500.0)["status"] == "UNIT_MISMATCH"


def test_unit_mismatch_can_never_stop_a_run():
    """exit_policy stops on IN_BAND only, so a refusal must not be mistaken for a target reached."""
    from agent import exit_policy

    src = (__import__("pathlib").Path(exit_policy.__file__)).read_text()
    assert 'status == "IN_BAND"' in src
    assert "UNIT_MISMATCH" not in src
