# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Protect queued hardware work from stale receipts, live owners and failed jobs."""

import copy
import json
from argparse import Namespace

from models.demos.qwen38_27b_qb2.demo import run_overnight_qualification as queue
from models.demos.qwen38_27b_qb2.demo.run_accuracy_followup import predecessor_ready


def completed():
    return {
        "state": "completed",
        "passed": False,
        "steps": [
            {"name": name, "state": "completed"} for name in ("native-g0-run", "native-control", "delivery-extended")
        ],
        "native-control": {"owned_processes_stopped": True},
    }


def stopped():
    return dict(LoadState="loaded", ActiveState="inactive", MainPID="0", Result="success", InvocationID="original")


def test_completed_accuracy_failure_allows_new_experiment():
    assert predecessor_ready(stopped(), completed(), "original")


def test_live_or_stopping_owner_overrides_completed_receipt():
    for properties in (
        dict(stopped(), ActiveState="active", MainPID="123"),
        dict(stopped(), ActiveState="deactivating", MainPID="0"),
        dict(stopped(), ActiveState="inactive", MainPID="123"),
        dict(stopped(), LoadState="error"),
        dict(stopped(), LoadState="not-found", ActiveState="active", MainPID="123"),
    ):
        assert predecessor_ready(properties, completed(), "original") is False


def test_replaced_invocation_is_rejected(expect_error):
    with expect_error(ValueError, "replaced"):
        predecessor_ready(dict(stopped(), InvocationID="replacement"), completed(), "original")


def test_missing_handle_requires_complete_hardware_receipts(expect_error):
    properties = dict(LoadState="not-found", ActiveState="inactive", MainPID="0", InvocationID="")
    assert predecessor_ready(properties, completed(), "original")
    for receipt in (None, dict(completed(), state="failed")):
        with expect_error(ValueError, "without completing"):
            predecessor_ready(properties, receipt, "original")
    receipt = completed()
    receipt["steps"][-1]["state"] = "failed"
    with expect_error(ValueError, "did not all finish"):
        predecessor_ready(properties, receipt, "original")
    receipt = completed()
    receipt["steps"].append(receipt["steps"][-1])
    with expect_error(ValueError, "did not all finish"):
        predecessor_ready(properties, receipt, "original")


def test_unclean_workers_or_failed_service_block_followup(expect_error):
    receipt = completed()
    receipt["native-control"]["owned_processes_stopped"] = False
    with expect_error(ValueError, "workers"):
        predecessor_ready(stopped(), receipt, "original")
    with expect_error(ValueError, "service failed"):
        predecessor_ready(dict(stopped(), ActiveState="failed", Result="timeout"), completed(), "original")


def test_accuracy_only_reuses_g0_and_full_eval_without_optional_stages(tmp_path, monkeypatch):
    source = tmp_path / "source"
    model = source / "models/demos/qwen38_27b_qb2"
    model.mkdir(parents=True)
    results = tmp_path / "results"
    policy = "precision_accurate_decode_bfp8_head.json"
    calls = []
    monkeypatch.setattr(queue, "environment", lambda *args: {})
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue, "check_g0", lambda *args: None)

    def capture(command, **kwargs):
        calls.append((command, copy.deepcopy(kwargs["env"])))
        if "--exit-after-eval" in command:
            output = results / "native-control/evaluation"
            output.mkdir(parents=True)
            (output / "deployment.json").write_text(
                json.dumps(
                    {
                        "state": "evaluation_completed",
                        "owned_processes_stopped": True,
                        "passed": False,
                        "gpqa": {"completed_samples": 198, "passed": False},
                    }
                )
            )

    monkeypatch.setattr(queue, "run_capture", capture)
    # run() propagates this to its G0 source check through os.environ.
    monkeypatch.setenv("QWEN_PRECISION_CONFIG", "before")
    queue.run(
        Namespace(
            task=tmp_path / "task",
            source=source,
            weights=tmp_path / "weights",
            results=results,
            candidate_g0=None,
            tau_root=None,
            delivery_source=None,
            native_control_only=True,
            accuracy_only=True,
            control_precision=policy,
        )
    )
    assert len(calls) == 2
    for command, env in calls:
        assert env["QWEN_PRECISION_CONFIG"] == str(model / "config" / policy)
        assert "--sweep-before-exit" not in command and "--tau-source" not in command
    assert any(str(model / "tests/test_galaxy_replicas.py") in command for command, _ in calls)
    evaluation = calls[1][0]
    assert evaluation[evaluation.index("--gpqa-max-tokens") + 1] == "65536"
    assert "--retain-raw-responses" in evaluation
    report = json.loads((results / "queue.json").read_text())
    assert report["state"] == "completed" and report["passed"] is False
    assert report["control_precision"] == policy
