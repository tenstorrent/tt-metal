# SPDX-License-Identifier: Apache-2.0
"""A stage's baseline is its pinned START, not the committed-best split. Using the committed-best for
both made every Latency Breakdown bar read baseline == current (galaxy qwen: identical bars while e2e
was 3.11x faster). With no pin, baseline_ms is None (UI shows current + the e2e gain); with a pin it
is the pinned value."""
import json

from tt_hw_planner.optimize_dashboard import collect_state


def _write(d, name, obj):
    (d / name).write_text(json.dumps(obj))


def test_stage_baseline_is_the_pin_not_the_committed_best(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _write(run, "state.json", {"state": "optimize", "metric": {"name": "device_ms", "baseline": 1.0}})
    sd = tmp_path / "state"
    sd.mkdir()
    slug = "m"
    # committed-best per-stage split (this is the CURRENT time):
    _write(
        sd,
        "perf_mcp_full_pipeline_baseline_1cq_%s_main.json" % slug,
        {"full_pipeline_ms": 300.0, "unit": "pass", "stages": {"a": 100.0, "b": 200.0}},
    )
    st = {s["name"]: s for s in collect_state(run, [sd], slug)["stages"]}
    # current follows the committed-best split; baseline is NOT the same split (no pin) -> None.
    assert st["a"]["ms"] == 100.0 and st["b"]["ms"] == 200.0
    assert (
        st["a"]["baseline_ms"] is None and st["b"]["baseline_ms"] is None
    ), "baseline must not be the committed-best split, or every bar reads baseline == current"
