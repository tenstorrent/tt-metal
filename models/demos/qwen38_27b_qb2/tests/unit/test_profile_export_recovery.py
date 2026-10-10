# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recovery cannot turn a hardware failure or mismatched run into a pass."""

import copy
import json
from argparse import Namespace

import pytest

from models.demos.qwen38_27b_qb2.demo import recover_full_profile_export as exporter
from models.demos.qwen38_27b_qb2.demo import watch_profile_export as watcher
from models.demos.qwen38_27b_qb2.tests.full_trace_profile import SCOPE
from models.demos.qwen38_27b_qb2.tests.profile_export_recovery import (
    replacement_command,
    require_unstarted_failure,
    terminal,
    validate_pair,
)


def receipt():
    return dict(
        state="completed",
        passed=True,
        cleanup_completed=True,
        scope=SCOPE,
        layer_indices=list(range(64)),
        prefill_calls=0,
        replays=[{}, {}, {}],
        device_ids=[0, 4, 8, 12],
        output_hashes=["logits"] * 5,
        token_hashes=["tokens"] * 5,
        input_tokens=32768,
        batch=16,
        precision={"decode_recurrence": "single_step_compact_gdn"},
        expected_recurrence="single_step_compact_gdn",
        operand_hashes={"state": "a"},
        source_sha256={"model": "b"},
    )


def stopped():
    return dict(LoadState="loaded", InvocationID="original", MainPID="0", ActiveState="failed", Result="exit-code")


def test_pair_rejects_partial_hardware_and_mismatched_inputs(expect_error):
    profile = receipt()
    validate_pair(profile, copy.deepcopy(profile))
    for key, bad in (
        ("state", "running"),
        ("passed", False),
        ("cleanup_completed", False),
        ("layer_indices", [0]),
        ("replays", [{}, {}]),
        ("device_ids", [0, 0, 0, 0]),
        ("output_hashes", ["a", "b", "a", "a", "a"]),
        ("token_hashes", ["a"]),
        ("operand_hashes", {}),
        ("source_sha256", {}),
        ("batch", 32),
        ("input_tokens", 16384),
        ("precision", {"decode_recurrence": "different"}),
    ):
        changed = dict(profile, **{key: bad})
        with expect_error(ValueError, ".*"):
            validate_pair(changed, profile)


def test_legacy_capture_still_requires_matching_effective_recurrence(expect_error):
    profile = receipt()
    del profile["expected_recurrence"]
    validate_pair(profile, copy.deepcopy(profile))
    with expect_error(ValueError, "recurrence mismatch"):
        validate_pair(profile, dict(profile, expected_recurrence="another"))


def test_streaming_capture_hash(tmp_path):
    path = tmp_path / "capture"
    path.write_bytes(b"abc")
    assert exporter.digest(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_replacement_requires_exact_terminal_unstarted_failure(expect_error):
    props = stopped()
    row = dict(state="failed", hardware_started=False, detail="Predecessor exited unsuccessfully")
    require_unstarted_failure(props, row, "original")
    assert not terminal(dict(props, MainPID="99", ActiveState="active"), "original")
    assert not terminal(dict(props, LoadState="not-found", InvocationID=""), "original")
    with expect_error(ValueError, ".*"):
        terminal(dict(props, InvocationID="another"), "original")
    for bad in (dict(row, hardware_started=True), dict(row, detail="Hardware timed out"), dict(row, state="completed")):
        with expect_error(ValueError, ".*"):
            require_unstarted_failure(props, bad, "original")
    with expect_error(ValueError, ".*"):
        require_unstarted_failure(dict(props, Result="success"), row, "original")


def test_launch_preserves_source_resources_and_hardware_policy(expect_error):
    command = [
        "systemd-run",
        "--user",
        "--unit=old",
        "--property=KillMode=control-group",
        "--property=MemoryMax=16G",
        "--property=RuntimeMaxSec=28h",
        "--property=StandardOutput=append:/old/run.log",
        "--property=StandardError=append:/old/run.log",
        "/existing/python",
        "-m",
        "existing.controller",
        "--source",
        "/frozen/source",
        "--weights",
        "/weights",
        "--manifest",
        "/frozen/manifest",
        "--output",
        "/old/result",
        "--after-unit",
        "previous",
        "--after-invocation",
        "previous-id",
        "--after-receipt",
        "/previous/queue.json",
    ]
    kwargs = dict(
        old_unit="old.service",
        new_unit="new.service",
        old_output="/old/result",
        new_output="/new/result",
        old_log="/old/run.log",
        new_log="/new/run.log",
        after_unit="watcher.service",
        after_invocation="watcher-id",
        after_receipt="/recovered/queue.json",
    )
    changed = replacement_command(command, **kwargs)
    assert command[2] == "--unit=old"
    assert changed[2] == "--unit=new"
    assert changed[3:6] == command[3:6]
    assert changed[8:17] == command[8:17]
    assert changed[-6:] == [
        "--after-unit",
        "watcher.service",
        "--after-invocation",
        "watcher-id",
        "--after-receipt",
        "/recovered/queue.json",
    ]
    for bad in (
        command + ["--output", "/extra"],
        [v for v in command if v != "--property=KillMode=control-group"],
        [v.replace("/old/result", "/other") for v in command],
    ):
        with expect_error(ValueError, ".*"):
            replacement_command(bad, **kwargs)


def test_failed_hardware_never_creates_recovery_directory(tmp_path, expect_error):
    for name in ("original", "baseline"):
        root = tmp_path / name
        root.mkdir()
        (root / "profile.json").write_text(json.dumps(receipt()))
        (root / "hardware.xml").write_text('<testsuites><testsuite tests="1" failures="1" /></testsuites>')
    with expect_error(ValueError, "hardware test failed"):
        exporter.run(
            Namespace(
                original=tmp_path / "original",
                baseline=tmp_path / "baseline",
                output=tmp_path / "new",
                python=tmp_path / "python",
                exporter=tmp_path / "exporter",
            )
        )
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("collected", [False, True])
def test_successful_original_does_not_recover_or_relaunch(tmp_path, monkeypatch, collected):
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(dict(state="completed", cleanup_completed=True)))
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(dict(parent_unit="parent.service", parent_invocation="original", parent_receipt=str(parent)))
    )
    monkeypatch.setenv("INVOCATION_ID", "watcher-id")
    monkeypatch.setattr(watcher.signal, "signal", lambda *args: None)
    props = dict(stopped(), ActiveState="inactive", Result="success")
    if collected:
        props.update(LoadState="not-found", InvocationID="")
    monkeypatch.setattr(watcher, "properties", lambda unit: props)
    monkeypatch.setattr(watcher, "recover", lambda args: pytest.fail("Unneeded export"))
    monkeypatch.setattr(watcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unneeded relaunch"))
    output = tmp_path / "result"
    watcher.run(Namespace(output=output, plan=plan))
    row = json.loads((output / "queue.json").read_text())
    assert row["state"] == "completed" and row["cleanup_completed"]
    assert row["recovery_needed"] is False and row["replacements"] == []


def test_qualification_failure_never_releases_followers(tmp_path, monkeypatch, expect_error):
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(dict(state="failed", active_stage="profiled")))
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps(dict(state="failed", **{"native-control": {"owned_processes_stopped": False}})))
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(
                parent_unit="parent.service",
                parent_invocation="original",
                parent_receipt=str(parent),
                qualification=str(qualification),
            )
        )
    )
    monkeypatch.setenv("INVOCATION_ID", "watcher-id")
    monkeypatch.setattr(watcher.signal, "signal", lambda *args: None)
    monkeypatch.setattr(watcher, "properties", lambda unit: stopped())
    monkeypatch.setattr(watcher, "recover", lambda args: pytest.fail("Unqualified recovery"))
    monkeypatch.setattr(watcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unqualified relaunch"))
    output = tmp_path / "result"
    with expect_error(ValueError, "completed qualification"):
        watcher.run(Namespace(output=output, plan=plan))
    row = json.loads((output / "queue.json").read_text())
    assert row["state"] == "failed" and not row["cleanup_completed"]
    assert row["replacements"] == []


def test_recovery_failure_never_relaunches(tmp_path, monkeypatch, expect_error):
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(dict(state="failed", active_stage="profiled")))
    qualification = tmp_path / "qualification.json"
    qualification.write_text(
        json.dumps(dict(state="completed", **{"native-control": {"owned_processes_stopped": True}}))
    )
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps(dict(state="completed", full_trace_reconciliation_passed=True)))
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(
                parent_unit="parent.service",
                parent_invocation="original",
                parent_receipt=str(parent),
                qualification=str(qualification),
                validation_receipt=str(validation),
                profiled="/profiled",
                baseline="/baseline",
                exporter="/exporter",
                python="/python",
            )
        )
    )
    monkeypatch.setenv("INVOCATION_ID", "watcher-id")
    monkeypatch.setattr(watcher.signal, "signal", lambda *args: None)
    monkeypatch.setattr(watcher, "properties", lambda unit: stopped())

    def fail(args):
        raise ValueError("Incomplete capture")

    monkeypatch.setattr(watcher, "recover", fail)
    monkeypatch.setattr(watcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unqualified relaunch"))
    output = tmp_path / "result"
    with expect_error(ValueError, "Incomplete capture"):
        watcher.run(Namespace(output=output, plan=plan))
    row = json.loads((output / "queue.json").read_text())
    assert row["state"] == "failed" and not row["cleanup_completed"] and not row["replacements"]
