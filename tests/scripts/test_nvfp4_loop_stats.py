# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

from scripts.nvfp4_loop_stats import summarize


def test_nvfp4_loop_stats_counts_retests_without_inflating_snapshots(tmp_path):
    for index, (start, end, kernel) in enumerate([(10, 15, "one"), (12, 20, "one"), (25, 27, "two")]):
        output = tmp_path / f"run-{index}"
        stage = output / "accuracy"
        stage.mkdir(parents=True)
        (output / "result.json").write_text(
            json.dumps(
                dict(
                    status="screened",
                    validation_level="screen",
                    started_unix=start,
                    finished_unix=end,
                    kernel_hashes={"kernel": kernel},
                    input_bit_permutation=None,
                )
            )
        )
        (stage / "pytest.xml").write_text(
            '<testsuites><testsuite><testcase name="pass"/><testcase name="skip"><skipped/></testcase></testsuite></testsuites>'
        )
        (output / "accuracy.json").write_text(json.dumps(dict(valid_values_per_scale_case=16, cases=[{}, {}])))
    result = summarize(tmp_path)
    assert result["evaluations"] == 3
    assert result["unique_source_and_encoding_snapshots"] == 2
    assert result["pytest_counts"] == dict(passed=3, skipped=3, failed=0)
    assert result["gate_datum_comparisons"] == 96
    assert result["evaluation_active_union_seconds"] == 12
    assert result["evaluation_window_seconds"] == 17
    assert result["detailed_timing_stages"] == 0


def test_nvfp4_loop_stats_ignores_incomplete_live_results(tmp_path):
    output = tmp_path / "writing"
    output.mkdir()
    path = output / "result.json"
    path.write_text('{"status":')
    result = summarize(tmp_path)
    assert result["evaluations"] == 0
    assert result["incomplete_artifacts"] == [str(path)]
