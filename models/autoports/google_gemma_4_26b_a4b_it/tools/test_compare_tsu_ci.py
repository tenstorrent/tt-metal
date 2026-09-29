# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for exact-shape CI evidence comparison."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "mutation", [None, "failed", "text", "short_output", "missing_text", "nan_tpot", "shape", "filtered_extra"]
)
def test_ci_comparison_rejects_incomplete_or_changed_results(tmp_path, mutation):
    data = {
        "mean_tpot_ms": 20.0,
        "mean_ttft_ms": 100.0,
        "mean_e2el_ms": 240.0,
        "median_itl_ms": 20.0,
        "completed": 1,
        "failed": 0,
        "input_lens": [16],
        "output_lens": [8],
        "generated_texts": ["same response"],
        "model_id": "model",
        "tokenizer_id": "tokenizer",
    }
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    control.mkdir()
    candidate.mkdir()
    filename = "benchmark_model_isl-16_osl-8_maxcon-1_n-1.json"
    (control / filename).write_text(json.dumps(data))
    if mutation == "failed":
        data["failed"] = 1
    elif mutation == "text":
        data["generated_texts"] = ["changed response"]
    elif mutation == "short_output":
        data["output_lens"] = [7]
    elif mutation == "missing_text":
        data["generated_texts"] = []
    elif mutation == "nan_tpot":
        data["mean_tpot_ms"] = float("nan")
    elif mutation == "shape":
        filename = filename.replace("isl-16", "isl-32")
    (candidate / filename).write_text(json.dumps(data))
    if mutation == "filtered_extra":
        other = dict(data, generated_texts=["different concurrency must not be compared"])
        (candidate / filename.replace("maxcon-1", "maxcon-8")).write_text(json.dumps(other))
    output = tmp_path / "comparison.json"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("compare_tsu_ci.py")),
            "--control",
            str(control),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
            *(["--concurrency", "1"] if mutation == "filtered_extra" else []),
        ],
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (mutation in (None, "filtered_extra")), result.stdout + result.stderr
    if mutation in (None, "filtered_extra"):
        report = json.loads(output.read_text())
        assert report["passed"] and report["rows"][0]["tsu_gain_percent"] == 0
        assert report["rows"][0]["candidate"]["tsu"] == 50
        if mutation == "filtered_extra":
            assert report["concurrency_filter"] == [1]
            assert report["excluded_shapes"]["candidate"] == [[16, 8, 8, 1]]
