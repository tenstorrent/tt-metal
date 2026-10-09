# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A finished receipt cannot allow the next job to overlap a live predecessor."""

import pytest

from models.demos.qwen38_27b_qb2.demo.run_chunked_prefill_followup import predecessor_ready


def stopped():
    return dict(InvocationID="owned", LoadState="loaded", ActiveState="inactive", MainPID="0", Result="success")


def test_wait_for_process_and_deactivation():
    receipt = dict(state="completed", hardware_opened=True, cleanup_completed=True)
    assert predecessor_ready(stopped(), receipt, "owned")
    assert not predecessor_ready(dict(stopped(), MainPID="123"), receipt, "owned")
    assert not predecessor_ready(dict(stopped(), ActiveState="deactivating"), receipt, "owned")


def test_explicit_no_hardware_skip_is_ready():
    assert predecessor_ready(stopped(), dict(state="completed", hardware_opened=False), "owned")


@pytest.mark.parametrize(
    "properties,receipt,pattern",
    [
        (dict(InvocationID="other"), dict(state="completed", hardware_opened=False), "invocation"),
        (dict(Result="timeout"), dict(state="completed", hardware_opened=False), "failed"),
        ({}, None, "completed receipt"),
        ({}, dict(state="running", hardware_opened=False), "completed receipt"),
        ({}, dict(state="completed", hardware_opened=True), "cleanup"),
        ({}, dict(state="completed"), "cleanup"),
    ],
)
def test_reject_unproven_completion(properties, receipt, pattern, expect_error):
    with expect_error(ValueError, pattern):
        predecessor_ready(dict(stopped(), **properties), receipt, "owned")
