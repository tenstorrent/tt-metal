# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from models.tt_dit.utils.vbench import assert_scores
from models.tt_dit.utils.vbench_bundle import METRICS, aggregate, digest


@pytest.fixture
def gate(tmp_path):
    bundle = tmp_path / "input"
    bundle.mkdir()
    manifest = {
        "prompt": "test",
        "thresholds": {metric: 0.8 for metric in METRICS},
        "clips": [{"name": f"seed_{i}.mp4", "shape": [145, 1088, 1920], "sha256": "unused"} for i in range(5)],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    results = tmp_path / "scores"
    results.mkdir()
    for i in range(5):
        scores = {metric: 0.9 for metric in METRICS}
        scores["dynamic_degree"] = int(i != 0)
        row = {"index": i, "manifest_sha256": digest(bundle / "manifest.json"), "scores": scores}
        (results / f"score_{i}.json").write_text(json.dumps(row))
    return bundle, results


def test_gate_averages_all_five_seeds(gate):
    assert aggregate(*gate)["dynamic_degree"] == 0.8


@pytest.mark.parametrize("failure", ["missing", "duplicate", "metric", "nan", "wrong_input", "below_floor"])
def test_incomplete_or_bad_quality_cannot_pass(gate, failure, expect_error):
    bundle, results = gate
    path = results / "score_4.json"
    row = json.loads(path.read_text())
    if failure == "missing":
        path.unlink()
    else:
        if failure == "duplicate":
            row["index"] = 0
        elif failure == "metric":
            del row["scores"]["motion_smoothness"]
        elif failure == "nan":
            row["scores"]["motion_smoothness"] = float("nan")
        elif failure == "wrong_input":
            row["manifest_sha256"] = "different build"
        else:
            row["scores"]["dynamic_degree"] = 0
        path.write_text(json.dumps(row))
    with expect_error((ValueError, AssertionError), "VBench"):
        aggregate(bundle, results)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1])
def test_invalid_scores_fail(score, expect_error):
    with expect_error(AssertionError, "VBench quality gate failed"):
        assert_scores({"quality": score}, {"quality": 0.8})
