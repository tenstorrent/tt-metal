# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Target preparation tests use synthetic commits and exports only."""

import json
from pathlib import Path

import pytest

from tools.generic_op_to_factory import prepare_baseline, prepare_target
from tools.generic_op_to_factory.export_run import ExportError
from tools.generic_op_to_factory.tests.test_export_run import snapshot  # noqa: F401
from tools.generic_op_to_factory.tests.test_prepare_baseline import inputs, commit  # noqa: F401


@pytest.fixture
def target(inputs, tmp_path):
    _, evaluator, runtime = inputs
    (runtime / "eval").symlink_to(evaluator / "eval", target_is_directory=True)
    package = tmp_path / "prepared"
    manifest = prepare_baseline.prepare(*inputs, package)
    return package, runtime, manifest["metal_commit"]


def test_inspect_is_read_only(target):
    package, runtime, revision = target
    before = set(runtime.rglob("*"))
    report = prepare_target.inspect(*target)
    assert not report["migration_ready"]
    assert not report["target_baseline_reproduced"]
    assert not any(row["changed"] for row in report["dependencies"])
    assert set(runtime.rglob("*")) == before


def test_installs_exact_source_without_overwriting(target, tmp_path):
    package, runtime, revision = target
    report = prepare_target.install(*target, tmp_path / "evidence")
    destination = runtime / "ttnn/ttnn/operations/sample_op/planner.py"
    assert destination.read_bytes() == (package / "overlay/ttnn/ttnn/operations/sample_op/planner.py").read_bytes()
    assert report["installed"] and not report["migration_ready"]
    with pytest.raises(ExportError, match="destination exists"):  # allow-pytest.raises: host-only workflow validation
        prepare_target.install(*target, tmp_path / "second")
    assert not (tmp_path / "second").exists()


def test_changed_committed_reference_is_reported_not_overwritten(target, tmp_path):
    package, runtime, _ = target
    relative = "ttnn/ttnn/operations/_op_contract.py"
    revision = commit(runtime, {relative: b"# target contract version\n"})
    report = prepare_target.install(package, runtime, revision, tmp_path / "evidence")
    assert next(row for row in report["dependencies"] if row["path"] == relative)["changed"]
    assert (runtime / relative).read_bytes() == b"# target contract version\n"


def test_uncommitted_reference_is_refused(target, tmp_path):
    package, runtime, _ = target
    (runtime / "ttnn/ttnn/operations/_op_contract.py").write_text("# changed\n")
    with pytest.raises(ExportError, match="uncommitted"):  # allow-pytest.raises: host-only workflow validation
        prepare_target.install(*target, tmp_path / "evidence")
    assert not (tmp_path / "evidence").exists()


def test_changed_golden_is_refused(target):
    package, runtime, _ = target
    (runtime / "eval/metrics.py").write_text("# changed\n")
    with pytest.raises(ExportError, match="golden/harness"):  # allow-pytest.raises: host-only workflow validation
        prepare_target.inspect(*target)


def test_only_exact_flow_runner_update_is_allowed_and_recorded(target):
    _, runtime, _ = target
    runner = runtime / "scripts/run_safe_pytest.sh"
    canonical = Path(prepare_target.__file__).resolve().parents[2] / "scripts/run_safe_pytest.sh"
    runner.write_bytes(canonical.read_bytes())
    report = prepare_target.inspect(*target)
    row = next(row for row in report["dependencies"] if row["path"] == "scripts/run_safe_pytest.sh")
    assert row["flow_runner_update"]
    assert row["target_sha256"] != row["target_commit_sha256"]
    runner.write_bytes(runner.read_bytes() + b"# arbitrary target edit\n")
    with pytest.raises(ExportError, match="uncommitted"):  # allow-pytest.raises: harness update gate
        prepare_target.inspect(*target)


def test_colocated_driver_does_not_approve_arbitrary_runner_changes(target, monkeypatch):
    _, runtime, _ = target
    monkeypatch.setattr(prepare_target, "__file__", str(runtime / "tools/generic_op_to_factory/prepare_target.py"))
    (runtime / "scripts/run_safe_pytest.sh").write_text("# arbitrary local runner\n")
    with pytest.raises(ExportError, match="uncommitted"):  # allow-pytest.raises: colocated admission gate
        prepare_target.inspect(*target)


def test_wrong_revision_is_refused(target):
    package, runtime, _ = target
    commit(runtime, {"other": b"new commit"})
    with pytest.raises(ExportError, match="Target HEAD"):  # allow-pytest.raises: host-only workflow validation
        prepare_target.inspect(*target)
