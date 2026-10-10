# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Protect full-model speed claims from drift, changed tokens or missing data."""

import copy

import pytest

from models.demos.qwen38_27b_qb2.demo.run_compact_gdn import compare_sweeps
from models.demos.qwen38_27b_qb2.tests.unit.test_b16_followup import report


def arms():
    return {name: copy.deepcopy(report()) for name in ("before", "compact", "after")}


def test_matching_tokens_and_stable_controls_do_not_imply_gpqa_qualification():
    data = arms()
    for cell in data["compact"]["cells"]:
        for sample in cell["samples"]:
            sample["decode_s"] = 4.0
    rows = compare_sweeps(data)
    assert all(r["comparison_qualified"] and r["target_reached"] for r in rows)
    assert all(not r["gpqa_qualified"] and not r["promoted_to_serving"] for r in rows)
    assert rows[0]["speedup"] == 2.5


@pytest.mark.parametrize("change", ["drift", "candidate_tokens", "control_tokens"])
def test_bad_comparisons_never_report_target_success(change):
    data = arms()
    if change == "drift":
        data["after"]["cells"][0]["samples"][0]["decode_s"] = 1.0
        data["after"]["cells"][0]["samples"][1]["decode_s"] = 1.0
    else:
        name = "compact" if change == "candidate_tokens" else "after"
        for sample in data[name]["cells"][0]["samples"]:
            sample["output_sha256_per_replica"] = ["b" * 64]
    row = compare_sweeps(data)[0]
    assert not row["comparison_qualified"] and not row["target_reached"]
    assert row["speedup"] is None


def test_incomplete_full_model_run_is_not_a_measurement(expect_error):
    data = arms()
    data["compact"]["cleanup_completed"] = False
    with expect_error(ValueError, "completed sweep"):
        compare_sweeps(data)
