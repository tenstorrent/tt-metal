# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Queue only after a terminal process plus clean receipts, never after quiet logs."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import wait_for_sweep

MODULE = "models.demos.qwen38_27b_qb2.demo.run_long_context_capacity"


def receipts(root, *, cleanup=True):
    for variant in ("native", "single-step"):
        directory = root / variant
        directory.mkdir()
        (directory / "sweep.json").write_text(json.dumps(dict(state="completed_with_oom", cleanup_completed=cleanup)))


def test_live_pid_waits_even_with_completed_receipts(tmp_path):
    receipts(tmp_path)
    observations = [
        SimpleNamespace(stdout="LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=456\nResult=success\n"),
        SimpleNamespace(stdout="LoadState=loaded\nActiveState=inactive\nSubState=dead\nMainPID=0\nResult=success\n"),
    ]
    with patch(MODULE + ".subprocess.run", side_effect=observations) as run, patch(MODULE + ".time.sleep") as sleep:
        wait_for_sweep("example.service", tmp_path, {}, tmp_path / "queue.json")
    assert run.call_count == 2
    sleep.assert_called_once_with(20)


def test_garbage_collected_success_requires_clean_receipts(tmp_path, expect_error):
    receipts(tmp_path, cleanup=False)
    missing = SimpleNamespace(stdout="LoadState=not-found\nActiveState=inactive\nSubState=dead\nMainPID=0\n")
    with patch(MODULE + ".subprocess.run", return_value=missing):
        with expect_error(RuntimeError, "no terminal sweep and clean-device receipt"):
            wait_for_sweep("example.service", tmp_path, {}, tmp_path / "queue.json")
        for variant in ("native", "single-step"):
            (tmp_path / variant / "sweep.json").write_text(json.dumps(dict(state="completed", cleanup_completed=True)))
        wait_for_sweep("example.service", tmp_path, {}, tmp_path / "queue.json")


def test_failed_process_does_not_start_hardware_even_with_clean_receipts(tmp_path, expect_error):
    receipts(tmp_path)
    failed = SimpleNamespace(
        stdout="LoadState=loaded\nActiveState=failed\nSubState=failed\nMainPID=0\nResult=timeout\n"
    )
    with patch(MODULE + ".subprocess.run", return_value=failed):
        with expect_error(RuntimeError, "did not exit successfully"):
            wait_for_sweep("example.service", tmp_path, {}, tmp_path / "queue.json")
