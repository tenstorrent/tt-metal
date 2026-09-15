# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic approved donor exports; no production identities or device usage."""

import json
from pathlib import Path

import pytest

from tools.generic_op_to_factory import dependency_substitutions as dependencies
from tools.generic_op_to_factory import migration_workflow, prepare_target, validate_port
from tools.generic_op_to_factory.export_run import ExportError, export_snapshot
from tools.generic_op_to_factory.prepare_baseline import GitTree
from tools.generic_op_to_factory.tests.test_export_run import ReadConnection, snapshot  # noqa: F401
from tools.generic_op_to_factory.tests.test_prepare_baseline import inputs  # noqa: F401
from tools.generic_op_to_factory.tests.test_prepare_target import target  # noqa: F401
from tools.generic_op_to_factory.tests.test_migration_workflow import configured  # noqa: F401
from tools.generic_op_to_factory.tests.test_validate_port import port, record_review  # noqa: F401


@pytest.fixture
def donor(snapshot, tmp_path):
    connection, run_id = snapshot
    connection.execute("PRAGMA query_only = OFF")
    connection.execute(
        "INSERT INTO kernels (run_id, filename, source_code) VALUES (?, ?, ?)",
        (run_id, "optional_helper.hpp", "#pragma once\n// exact synthetic donor bytes\n"),
    )
    connection.execute("PRAGMA query_only = ON")
    package = tmp_path / "donor-export"
    export_snapshot(ReadConnection(connection), run_id, package, database={"host": "synthetic"})
    return {
        "export": str(package),
        "source_path": "source/kernels/optional_helper.hpp",
        "destination": "ttnn/cpp/ttnn/kernel_lib/optional_helper.hpp",
        "approval": "Synthetic explicit user authorization",
        "reason": "Missing original dependency",
        "limitations": "Donor provenance does not establish historical dependency identity",
    }


def test_exact_install_provenance_and_no_overwrite(target, donor):
    _, runtime, revision = target
    resolved = dependencies.resolve([donor], GitTree(runtime, revision))
    (entry,) = resolved
    assert entry["table"] == "kernels" and entry["row_id"] > 0
    assert not entry["historical_dependency_identity_verified"]
    dependencies.install(resolved, runtime)
    destination = runtime / donor["destination"]
    assert destination.read_bytes() == (Path(donor["export"]) / donor["source_path"]).read_bytes()
    dependencies.check_runtime(resolved, runtime, installed=True)
    with pytest.raises(ExportError, match="refusing overwrite"):  # allow-pytest.raises: host-only validation
        dependencies.install(resolved, runtime)
    destination.write_text("changed")
    with pytest.raises(ExportError, match="substitution changed"):  # allow-pytest.raises: host-only validation
        dependencies.check_runtime(resolved, runtime, installed=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("approval", ""),
        ("limitations", ""),
        ("destination", "../escape.hpp"),
        ("destination", "ttnn/cpp/ttnn/kernel_lib/../../escape.hpp"),
        ("destination", "scripts/header.hpp"),
        ("destination", "ttnn/cpp/ttnn/kernel_lib/script.py"),
        ("source_path", "records/run.json"),
        ("source_path", "source/kernels/missing.hpp"),
        ("export", "relative/export"),
    ],
)
def test_invalid_substitution_refused(target, donor, field, value):
    donor[field] = value
    with pytest.raises(ExportError):  # allow-pytest.raises: host-only validation
        dependencies.resolve([donor], GitTree(target[1], target[2]))


def test_pinned_file_and_duplicate_targets_refused(target, donor):
    tree = GitTree(target[1], target[2])
    with pytest.raises(ExportError, match="Duplicate"):  # allow-pytest.raises: host-only validation
        dependencies.resolve([donor, donor], tree)
    donor["destination"] = next(path for path in tree.entries if path.startswith("ttnn/cpp/ttnn/kernel_lib/"))
    with pytest.raises(ExportError, match="pinned Git entry"):  # allow-pytest.raises: host-only validation
        dependencies.resolve([donor], tree)


def test_redirected_destination_refused(target, donor, tmp_path):
    _, runtime, revision = target
    donor["destination"] = "ttnn/cpp/ttnn/kernel_lib/redirect/optional_helper.hpp"
    resolved = dependencies.resolve([donor], GitTree(runtime, revision))
    (runtime / "ttnn/cpp/ttnn/kernel_lib/redirect").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ExportError, match="redirected"):  # allow-pytest.raises: host-only validation
        dependencies.install(resolved, runtime)
    assert not (tmp_path / "optional_helper.hpp").exists()


def test_donor_drift_refused_before_write(target, donor):
    _, runtime, revision = target
    resolved = dependencies.resolve([donor], GitTree(runtime, revision))
    (Path(donor["export"]) / donor["source_path"]).write_text("changed donor")
    with pytest.raises(ExportError, match="source changed"):  # allow-pytest.raises: host-only validation
        dependencies.install(resolved, runtime)
    assert not (runtime / donor["destination"]).exists()


def test_workflow_substitution_scope_resume_and_drift(configured, donor):
    configured["dependency_substitutions"] = [donor]
    migration_workflow.initialize(configured)
    workflow = migration_workflow.Workflow(configured["workspace"])
    workflow.run()
    assert "not exact historical" in workflow.state["baseline_scope"]
    receipt = workflow.receipt("install")["result"]
    assert receipt["dependency_substitutions"][0]["donor_snapshot_sha256"]
    comparison = workflow.receipt("compare")["result"]
    assert comparison["outcomes_match"] and "substituted" in comparison["baseline_scope"]
    workflow.run()  # No-op resume verifies donor and installed hashes.
    (workflow.runtime / donor["destination"]).write_text("tampered")
    with pytest.raises(ExportError, match="substitution changed"):  # allow-pytest.raises: host-only validation
        workflow.run()


def test_target_installs_and_revalidates_substitution(target, donor, tmp_path):
    report = prepare_target.install(*target, tmp_path / "evidence", substitutions=[donor])
    assert "substituted" in report["baseline_scope"]
    prepare_target.inspect(*target, allow_installed=True, substitutions=[donor])
    (target[1] / donor["destination"]).unlink()
    with pytest.raises(ExportError, match="substitution changed"):  # allow-pytest.raises: host-only validation
        prepare_target.inspect(*target, allow_installed=True, substitutions=[donor])


def test_target_refuses_undeclared_canonical_header(target):
    header = target[1] / "ttnn/cpp/ttnn/kernel_lib/undeclared.hpp"
    header.write_text("// unrecorded helper\n")
    with pytest.raises(ExportError, match="explicit substitution"):  # allow-pytest.raises: host-only validation
        prepare_target.inspect(*target)


def test_baseline_refuses_undeclared_canonical_header(configured):
    migration_workflow.initialize(configured)
    workflow = migration_workflow.Workflow(configured["workspace"])
    workflow.run("install")
    (workflow.runtime / "ttnn/cpp/ttnn/kernel_lib/undeclared.hpp").write_text("// unrecorded helper\n")
    with pytest.raises(ExportError, match="explicit substitution"):  # allow-pytest.raises: host-only validation
        workflow.run()


def test_port_completion_keeps_substitution_scope(port, donor):
    runtime = Path(port["runtime"])
    resolved = dependencies.resolve([donor], GitTree(runtime, port["target_revision"]))
    dependencies.install(resolved, runtime)
    port["dependency_substitutions"] = [donor]
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run("acceptance")
    record_review(validation)
    validation.run()
    result = json.loads((validation.workspace / "attempts/complete/001/result.json").read_text())
    assert "not exact historical" in result["baseline_scope"]
    assert result["dependency_substitutions"][0]["sha256"] == resolved[0]["sha256"]
    assert not result["production_ready"]
