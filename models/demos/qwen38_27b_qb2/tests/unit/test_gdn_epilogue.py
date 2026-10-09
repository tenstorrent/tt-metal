# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings, predecessor_ready


def terminal():
    return dict(MainPID="0", ActiveState="inactive", LoadState="loaded", Result="success", InvocationID="expected")


def test_live_or_unknown_predecessor_never_releases_hardware():
    receipt = dict(state="completed", cleanup_completed=True)
    assert not predecessor_ready(dict(terminal(), MainPID="1", ActiveState="active"), receipt, "expected")
    assert not predecessor_ready(dict(terminal(), LoadState="error"), receipt, "expected")


def test_wrong_invocation_or_failed_unit_rejected(expect_error):
    receipt = dict(state="completed", cleanup_completed=True)
    with expect_error(ValueError, "invocation changed"):
        predecessor_ready(dict(terminal(), InvocationID="other"), receipt, "expected")
    with expect_error(ValueError, "exited unsuccessfully"):
        predecessor_ready(dict(terminal(), Result="timeout"), receipt, "expected")


@pytest.mark.parametrize("receipt", [None, {}, dict(state="completed"), dict(state="failed", cleanup_completed=True)])
def test_terminal_without_clean_receipt_rejected(receipt, expect_error):
    with expect_error(ValueError, "lacks completed qualification"):
        predecessor_ready(terminal(), receipt, "expected")


def test_terminal_or_collected_unit_with_clean_receipt_accepted():
    receipt = dict(state="completed", cleanup_completed=True)
    assert predecessor_ready(terminal(), receipt, "expected")
    assert predecessor_ready(dict(terminal(), LoadState="not-found", InvocationID=""), receipt, "expected")


def bracket(after=100):
    return [
        dict(variant=variant, traced_call_us=[value] * 5)
        for variant, value in (("native", 100), ("fused", 50), ("native", after))
    ]


def test_timing_requires_stable_control_and_labels_projection():
    result = compare_timings(bracket())
    assert result["qualified_speedup"] == 2
    assert result["projected_48_layer_saving_ms"] == 2.4
    assert result["full_model_speedup_measured"] is False
    result = compare_timings(bracket(104))
    assert result["qualified_speedup"] is None
    assert result["projected_48_layer_saving_ms"] is None


@pytest.mark.parametrize("samples", [[50] * 4, [float("nan")] * 5, [0] * 5, [-1] * 5])
def test_invalid_timing_rejected(samples, expect_error):
    cases = bracket()
    cases[1]["traced_call_us"] = samples
    with expect_error(ValueError, "Invalid or incomplete"):
        compare_timings(cases)


def test_missing_control_rejected(expect_error):
    with expect_error(ValueError, "Require native/fused/native"):
        compare_timings(bracket()[:2])
