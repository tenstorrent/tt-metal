# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tests.perf import compare as cmp
from tests.perf import golden as golden_io
from tests.perf import report
from tests.perf.contract import MetricSpec
from tests.perf.golden import Environment, Golden
from tests.perf.unit.expect import expect_error  # noqa: F401

METRICS = {"t": MetricSpec("s", "lower", "min")}


def test_golden_round_trip_is_deterministic_with_one_case_per_line(tmp_path):
    golden = Golden(
        "pgm",
        METRICS,
        {
            "wh": Environment(
                2, {"commit": "abc"}, {"aiclk_mhz": "1000"}, {"b/x:2": {"t": 1.23456e-6}, "a/x:1": {"t": 2e-6}}
            ),
            "bh": Environment(2, {}, {}, {}),
        },
    )
    path = tmp_path / "g.json"
    golden_io.save(golden, path)
    text = path.read_text()
    assert json.loads(text)["environments"]["wh"]["cases"]["b/x:2"] == {"t": 1.235e-06}
    case_lines = [line for line in text.splitlines() if line.startswith("        ")]
    assert len(case_lines) == 2 and case_lines[0].strip().startswith('"a/x:1"')
    loaded = golden_io.load(path, "pgm")
    golden_io.save(loaded, path)
    assert path.read_text() == text


def test_missing_golden_file_is_empty(tmp_path):
    assert golden_io.load(tmp_path / "none.json", "pgm").environments == {}


@pytest.mark.parametrize(
    "document, message",
    [
        ({"schema_version": 2, "suite": "pgm", "metrics": {}}, "schema_version"),
        ({"schema_version": 1, "suite": "other", "metrics": {}}, "suite"),
        (
            {"schema_version": 1, "suite": "pgm", "metrics": {}, "environments": {"wh": {"cases": {"a": {"t": 1}}}}},
            "undeclared metric",
        ),
    ],
)
def test_invalid_goldens_are_rejected(expect_error, tmp_path, document, message):
    path = tmp_path / "g.json"
    path.write_text(json.dumps(document))
    with expect_error(golden_io.GoldenError, message):
        golden_io.load(path, "pgm")


def test_report_groups_and_lists_only_failures_by_default():
    comparison = cmp.Comparison(
        results=[
            cmp.CaseResult("BM/a/kernel_size:256/manual_time", "t", "PASS", 1e-6, 1e-6),
            cmp.CaseResult("BM/a/kernel_size:512/manual_time", "t", "REGRESSION", 1.2e-6, 1e-6, retry_actual=1.19e-6),
            cmp.CaseResult("BM/b/manual_time", "t", "PASS", 1e-6, 1e-6, retry_actual=1.0e-6, first_status="STALE"),
        ]
    )
    text = report.render(
        "pgm / wh", comparison, {"t": "s"}, context={"aiclk_mhz": "900"}, golden_context={"aiclk_mhz": "1000"}
    )
    assert "2 PASS" in text and "1 REGRESSION" in text
    assert "aiclk_mhz=900 (golden 1000)" in text
    assert "kernel_size:512" in text and "1.20 us" in text and "+20.0%" in text and "retry +19.0%" in text
    assert "PASS (first run STALE)" in text
    assert "kernel_size:256" not in text
    markdown = report.render("pgm / wh", comparison, {"t": "s"}, markdown=True)
    assert markdown.startswith("### pgm / wh") and "| REGRESSION |" in markdown


@pytest.mark.parametrize(
    "value, unit, text",
    [(1.5e-6, "s", "1.50 us"), (8.3e-9, "s", "8.30 ns"), (1.44e10, "B/s", "14.400 GB/s"), (50700, "ms", "5.07e+04 ms")],
)
def test_value_formatting(value, unit, text):
    assert report.format_value(value, unit) == text
