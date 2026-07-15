# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from models.autoports.google_gemma_4_31b.tests.generate_datatype_sweep_artifacts import (  # noqa: E402
    _annotation_roles,
    _main_panel_rows,
    _pareto,
    _plot,
)

PERF_KEY = "trace_verified_teacher_forcing_decode_t/s/u"


def _result(config_id: str, top1: float, top5: float, perf: float, passes: bool) -> dict:
    return {
        "config_id": config_id,
        "top1": top1,
        "top5": top5,
        PERF_KEY: perf,
        "pass": passes,
    }


def _rows() -> list[dict]:
    return [
        _result("selected", 0.92, 1.00, 24.50, True),
        _result("closest_pass", 0.90, 1.00, 24.40, True),
        _result("other_pass", 0.91, 1.00, 23.80, True),
        _result("high_accuracy_frontier", 0.93, 1.00, 18.00, True),
        _result("near_accuracy_failure", 0.89, 1.00, 23.00, False),
        _result("low_accuracy_outlier", 0.01, 0.01, 24.10, False),
    ]


def test_pareto_and_annotation_selection_cover_decision_rows():
    rows = _rows()

    assert [row["config_id"] for row in _pareto(rows, "top1")] == [
        "selected",
        "high_accuracy_frontier",
    ]
    assert [row["config_id"] for row in _main_panel_rows(rows, "top1", 0.90)] == [
        "selected",
        "closest_pass",
        "other_pass",
        "high_accuracy_frontier",
        "near_accuracy_failure",
    ]

    roles = _annotation_roles(rows, "selected", "top1", 0.90)
    assert "selected" in roles["selected"][1]
    assert "closest passing" in roles["closest_pass"][1]
    assert "frontier" in roles["high_accuracy_frontier"][1]
    assert "closest accuracy failure" in roles["near_accuracy_failure"][1]
    assert "fastest accuracy failure" in roles["low_accuracy_outlier"][1]


def test_plot_keeps_every_point_in_overview_and_makes_outlier_explicit(tmp_path: Path):
    output = tmp_path / "pareto.png"

    summary = _plot(_rows(), "selected", "top1", 0.90, output)

    assert summary["evaluated_count"] == summary["overview_count"] == 6
    assert summary["main_count"] == 5
    assert summary["frontier_ids"] == ["selected", "high_accuracy_frontier"]
    assert summary["selected_id"] == "selected"
    assert set(summary["annotated_ids"]) == {
        "selected",
        "near_accuracy_failure",
        "high_accuracy_frontier",
        "closest_pass",
        "other_pass",
    }
    assert summary["excluded_ids"] == ["low_accuracy_outlier"]
    assert summary["threshold"] == 0.90
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 20_000
