# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resume must retain valid measurements without hiding correctness or device failures."""

import copy
import json
from types import SimpleNamespace

import pytest

from models.demos.qwen38_27b_qb2.tests.sweep_recovery import can_restart, is_dram_allocation_error, resume_measurements
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, summarize

OOM = {
    "type": "RuntimeError",
    "message": "Out of Memory: Not enough space to allocate 1342177280 B DRAM buffer across 8 banks",
}


def receipt():
    report = make_plan(batches=(1, 2), input_lengths=(8192,))
    report.update(
        source_sha256={"tt/model.py": "model-hash", "effective_precision_override": "policy-hash"},
        precision={"decode_recurrence": "native"},
        configuration={"environment": {"QWEN_DECODE_BUCKETS": "0", "QWEN_SWEEP_RESULTS": "/old/output"}},
    )
    return report


def completed_cell(report):
    cell = report["cells"][0]
    sample = dict(
        prefill_s=2, decode_s=3, elapsed_s=6, ttft_s=[2.5], trace_captures=0, output_sha256_per_replica=["tokens"]
    )
    cell.update(
        status="completed",
        prompt_sha256="prompt",
        warmup=copy.deepcopy(sample),
        samples=[copy.deepcopy(sample) for _ in range(3)],
    )
    cell["summary"] = summarize(cell["samples"], concurrency=1, input_tokens=8192, output_tokens=128)


def write_receipt(tmp_path, report):
    path = tmp_path / "previous.json"
    path.write_text(json.dumps(report))
    return path


def test_only_allocator_dram_oom_can_restart_after_clean_shutdown():
    report = receipt()
    report.update(state="allocation_failed", cleanup_completed=True, error=OOM)
    report["cells"][0].update(status="oom", error=OOM)
    assert can_restart(report, 1)
    for code in (0, 2, 124, 137):
        assert not can_restart(report, code)
    report["cleanup_completed"] = False
    assert not can_restart(report, 1)
    for error in (MemoryError("Out of Memory"), RuntimeError("dispatch timeout"), AssertionError(OOM["message"])):
        assert not is_dram_allocation_error(error)


def test_resume_preserves_completed_cells_and_retries_legacy_failure(tmp_path):
    old, new = receipt(), receipt()
    completed_cell(old)
    old.update(state="failed", error=OOM)
    old["cells"][1]["status"] = "failed"
    new["configuration"]["environment"]["QWEN_SWEEP_RESULTS"] = "/new/output"
    new["configuration"]["environment"]["QWEN_SWEEP_RESUME_FROM"] = "[]"
    path = write_receipt(tmp_path, old)
    original = path.read_bytes()
    resume_measurements(new, [path])
    assert new["cells"][0]["samples"] == old["cells"][0]["samples"]
    assert new["cells"][0]["carried_from"]["path"] == str(path)
    assert new["cells"][1]["status"] == "queued"
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", {"changed": "hash"}),
        ("precision", {"decode_recurrence": "single_step"}),
        ("output_tokens", 256),
    ],
)
def test_resume_rejects_model_or_workload_change(tmp_path, field, value, expect_error):
    old, new = receipt(), receipt()
    old["state"] = "completed"
    completed_cell(old)
    new[field] = value
    with expect_error(ValueError, "differs"):
        resume_measurements(new, [write_receipt(tmp_path, old)])


def test_resume_rejects_changed_runtime_knob(tmp_path, expect_error):
    old, new = receipt(), receipt()
    old["state"] = "completed"
    new["configuration"]["environment"]["QWEN_DECODE_BUCKETS"] = "1"
    with expect_error(ValueError, "runtime configuration"):
        resume_measurements(new, [write_receipt(tmp_path, old)])


@pytest.mark.parametrize("defect", ["trace", "tokens", "missing_repeat", "summary"])
def test_resume_rechecks_raw_measurements(tmp_path, defect, expect_error):
    old, new = receipt(), receipt()
    old["state"] = "completed"
    completed_cell(old)
    cell = old["cells"][0]
    if defect == "trace":
        cell["samples"][0]["trace_captures"] = 1
    elif defect == "tokens":
        cell["samples"][0]["output_sha256_per_replica"] = ["changed"]
    elif defect == "missing_repeat":
        cell["samples"].pop()
    else:
        cell["summary"]["tokens_per_second_per_user"] *= 2
    message = (
        "warmup is incomplete"
        if defect == "trace"
        else "accounting differs"
        if defect == "summary"
        else "repeatability evidence"
    )
    with expect_error(ValueError, message):
        resume_measurements(new, [write_receipt(tmp_path, old)])


def test_accuracy_failure_cannot_be_resumed_as_capacity(tmp_path, expect_error):
    old, new = receipt(), receipt()
    old.update(state="failed", error={"type": "AssertionError", "message": "Greedy outputs changed"})
    completed_cell(old)
    with expect_error(ValueError, "not a DRAM allocation"):
        resume_measurements(new, [write_receipt(tmp_path, old)])


def test_known_oom_stays_visible_and_does_not_become_a_measurement(tmp_path):
    old, new = receipt(), receipt()
    old.update(state="allocation_failed", error=OOM, cleanup_completed=True)
    old["cells"][0].update(status="oom", error=OOM)
    resume_measurements(new, [write_receipt(tmp_path, old)])
    assert new["cells"][0]["status"] == "oom"
    assert "summary" not in new["cells"][0]
    assert new["cells"][1]["status"] == "queued"


def test_controller_restarts_only_clean_oom_and_preserves_attempts(tmp_path, monkeypatch):
    from models.demos.qwen38_27b_qb2.demo import run_sweep_attempts as controller

    root = tmp_path / "results"
    controller.save_report(receipt(), root)
    calls = []

    def run_command(command, *, env, check):
        assert f"--rootdir={tmp_path}" in command
        directory = controller.Path(env["QWEN_SWEEP_RESULTS"])
        report = json.loads((directory / "sweep.json").read_text())
        calls.append((directory, json.loads(env["QWEN_SWEEP_RESUME_FROM"])))
        if len(calls) == 1:
            report.update(state="allocation_failed", cleanup_completed=True, error=OOM)
            report["cells"][0].update(status="oom", error=OOM)
            report["cells"][1]["status"] = "not_run"
            returncode = 1
        else:
            report.update(state="completed_with_oom", cleanup_completed=True)
            report["cells"][0].update(status="oom", error=OOM)
            report["cells"][1]["status"] = "completed"
            returncode = 0
        controller.save_report(report, directory)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(controller.subprocess, "run", run_command)
    monkeypatch.setattr(controller, "render", lambda *_: None)
    controller.run(SimpleNamespace(results=root, resume=[], timeout=900, task=tmp_path, source=tmp_path))
    assert len(calls) == 2
    assert calls[1][1] == [str(calls[0][0] / "sweep.json")]
    assert json.loads((calls[0][0] / "sweep.json").read_text())["state"] == "allocation_failed"
    assert json.loads((root / "sweep.json").read_text())["state"] == "completed_with_oom"


@pytest.mark.parametrize(
    "code,clean,error",
    [(2, True, OOM), (1, False, OOM), (1, True, {"type": "AssertionError", "message": "output mismatch"})],
)
def test_controller_stops_on_unclean_or_nonallocator_failure(tmp_path, monkeypatch, code, clean, error, expect_error):
    from models.demos.qwen38_27b_qb2.demo import run_sweep_attempts as controller

    root = tmp_path / "results"
    controller.save_report(receipt(), root)

    def run_command(command, *, env, check):
        directory = controller.Path(env["QWEN_SWEEP_RESULTS"])
        report = receipt()
        report.update(state="allocation_failed", cleanup_completed=clean, error=error)
        report["cells"][0].update(status="oom", error=error)
        controller.save_report(report, directory)
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(controller.subprocess, "run", run_command)
    monkeypatch.setattr(controller, "render", lambda *_: None)
    with expect_error(RuntimeError, "refusing automatic recovery"):
        controller.run(SimpleNamespace(results=root, resume=[], timeout=900, task=tmp_path, source=tmp_path))
    assert not (root / "attempt-02").exists()
