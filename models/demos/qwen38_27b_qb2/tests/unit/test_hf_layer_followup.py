# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The diagnostic cannot overlap CPU work or silently weaken GPQA qualification."""

from models.demos.qwen38_27b_qb2.demo.run_hf_layer_followup import cpu_predecessor_ready, needs_diagnostic


def stopped():
    return dict(InvocationID="owned", LoadState="loaded", ActiveState="inactive", MainPID="0", Result="success")


def receipt():
    return dict(state="completed", stages=[dict(name="hf-head-reference", state="completed")])


def test_cpu_work_must_stop_even_with_a_complete_receipt():
    assert cpu_predecessor_ready(stopped(), receipt(), "owned")
    for updates in (dict(MainPID="123"), dict(ActiveState="deactivating"), dict(LoadState="error")):
        assert not cpu_predecessor_ready(dict(stopped(), **updates), receipt(), "owned")


def test_replaced_or_failed_cpu_job_is_rejected(expect_error):
    for updates in (dict(InvocationID="different"), dict(Result="timeout")):
        with expect_error(ValueError, "CPU predecessor"):
            cpu_predecessor_ready(dict(stopped(), **updates), receipt(), "owned")


def test_missing_service_requires_completed_cpu_evidence(expect_error):
    properties = dict(stopped(), InvocationID="", LoadState="not-found")
    assert cpu_predecessor_ready(properties, receipt(), "owned")
    for bad in (None, dict(state="running"), dict(state="completed", stages=[])):
        with expect_error(ValueError, "CPU predecessor|CPU HF reference"):
            cpu_predecessor_ready(properties, bad, "owned")


def test_skip_only_for_complete_passing_gpqa(expect_error):
    result = dict(completed_samples=198, dataset_samples=198, correct=176, passed=False)
    assert needs_diagnostic(dict(gpqa_result=result))
    assert not needs_diagnostic(dict(gpqa_result=dict(result, correct=177, passed=True)))
    for updates in (dict(completed_samples=197), dict(passed=True), dict(correct=177)):
        with expect_error(ValueError, "Head control"):
            needs_diagnostic(dict(gpqa_result=dict(result, **updates)))
