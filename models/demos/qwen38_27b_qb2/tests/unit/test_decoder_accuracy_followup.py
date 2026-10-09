# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Do not qualify a precision candidate while its diagnostic still owns hardware."""

import copy

import pytest

from models.demos.qwen38_27b_qb2.demo.run_accuracy_followup import decoder_controls_ready


def stopped():
    return dict(LoadState="loaded", ActiveState="inactive", MainPID="0", Result="success", InvocationID="owned")


def completed():
    return dict(
        state="completed",
        cleanup_completed=True,
        runs=[
            dict(name=name, state="completed", cleanup_completed=True, logit_metrics=[{} for _ in range(8)])
            for name in ("bfp4-hifi2", "bfp8-hifi2")
        ],
    )


def test_terminal_controls_and_missing_collected_service_are_supported():
    assert decoder_controls_ready(stopped(), completed(), "owned")
    assert decoder_controls_ready(dict(stopped(), LoadState="not-found", InvocationID=""), completed(), "owned")


def test_live_control_is_waited_on_even_with_stale_completed_receipt():
    assert not decoder_controls_ready(dict(stopped(), MainPID="123", ActiveState="active"), completed(), "owned")


@pytest.mark.parametrize("failure", ["missing_run", "failed_run", "no_cleanup", "short_run", "duplicate_run"])
def test_incomplete_control_never_authorizes_g0(failure, expect_error):
    report = completed()
    if failure == "missing_run":
        report["runs"].pop()
    elif failure == "failed_run":
        report["runs"][1]["state"] = "failed"
    elif failure == "no_cleanup":
        report["runs"][1]["cleanup_completed"] = False
    elif failure == "short_run":
        report["runs"][1]["logit_metrics"].pop()
    else:
        report["runs"][1] = copy.deepcopy(report["runs"][0])
    with expect_error(ValueError, "Decoder controls"):
        decoder_controls_ready(stopped(), report, "owned")


def test_changed_invocation_and_failed_service_are_rejected(expect_error):
    for updates in (dict(InvocationID="different"), dict(Result="exit-code")):
        with expect_error(ValueError, "Predecessor"):
            decoder_controls_ready(dict(stopped(), **updates), completed(), "owned")
