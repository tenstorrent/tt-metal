# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic target/native orchestration; never opens a real device."""

import json
from pathlib import Path

import pytest

from tools.generic_op_to_factory import migration_workflow, validate_port
from tools.generic_op_to_factory.export_run import ExportError
from tools.generic_op_to_factory.tests.test_export_run import snapshot  # noqa: F401
from tools.generic_op_to_factory.tests.test_prepare_baseline import inputs, commit  # noqa: F401
from tools.generic_op_to_factory.tests.test_migration_workflow import configured  # noqa: F401


@pytest.fixture
def port(configured, tmp_path):
    migration_workflow.initialize(configured)
    baseline = migration_workflow.Workflow(configured["workspace"])
    baseline.run("build")
    revision = commit(
        baseline.runtime,
        {
            "tools/generic_op_to_factory/native_adapter.py": b"# synthetic adapter; fake runner does not import it\n",
            "tests/test_cache.py": b"# fake runner emits a passing cache test\n",
        },
    )
    return {
        "runtime": str(baseline.runtime),
        "target_revision": revision,
        "preparation": str(baseline.prepared),
        "export": configured["export"],
        "phase": "initial",
        "workspace": str(tmp_path / "port-evidence"),
        "source_entry": "ttnn.operations.sample_op:sample_op",
        "native_entry": "ttnn:sample_native",
        "smoke_nodeid": "eval/golden_tests/sample_suite/test_golden.py::test_sample[case]",
        "cache_test": "tests/test_cache.py",
        "build_argv": ["./build_metal.sh"],
        "precompile": False,
        "allow_recorded_failures": False,
    }


def test_port_plan_read_only(port):
    planned = validate_port.plan(port)
    assert planned["recorded_case_count"] == 1
    assert not Path(port["workspace"]).exists()


def test_complete_synthetic_port_and_resume(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run("cache")
    record_review(validation)
    state = validation.run()
    assert all(r["status"] == "complete" for r in state["stages"].values())
    before = (validation.workspace / "state.json").read_bytes()
    validation.run()
    assert (validation.workspace / "state.json").read_bytes() == before


def review_receipt(validation):
    return {
        "plan_sha256": validation.state["plan_sha256"],
        "author": "synthetic-author",
        "reviewer": "synthetic-independent-reviewer",
        "topics": {key: "Synthetic evidence for orchestration test" for key in validate_port.REVIEW_TOPICS},
        "findings": [],
        "performance": {
            "classification": "not_measured",
            "assessment": "No timings measured; no performance claim",
            "measurements": [],
        },
    }


def record_review(validation, receipt=None):
    (validation.workspace / "review.json").write_text(json.dumps(receipt or review_receipt(validation)))


def test_missing_review_blocks_completion_without_rerunning_device_stages(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        ExportError, match="Independent review required"
    ):
        validation.run()
    assert validation.state["stages"]["complete"]["status"] == "pending"
    prior_cache = dict(validation.state["stages"]["cache"])
    record_review(validation)
    validation.run(retry=True)
    assert validation.state["stages"]["cache"] == prior_cache


@pytest.mark.parametrize(
    "defect",
    ["stale", "same_author", "missing_topic", "unresolved", "unsupported_measurement"],
)
def test_invalid_review_is_rejected(port, defect):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    receipt = review_receipt(validation)
    if defect == "stale":
        receipt["plan_sha256"] = "old-plan"
    elif defect == "same_author":
        receipt["reviewer"] = receipt["author"]
    elif defect == "missing_topic":
        del receipt["topics"]["alias_transitions"]
    elif defect == "unresolved":
        receipt["findings"] = [{"finding": "wrong address", "status": "open", "disposition": "pending"}]
    else:
        receipt["performance"]["classification"] = "measured"
    with pytest.raises(ExportError):  # allow-pytest.raises: host-only workflow validation
        validate_port.verify_review(receipt, validation.state["plan_sha256"])


def test_completed_review_drift_is_rejected(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run("cache")
    record_review(validation)
    validation.run()
    (validation.workspace / "review.json").write_text("{}")
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        ExportError, match="evidence/runtime changed"
    ):
        validation.run()


def test_measurement_evidence_is_checked(port, tmp_path):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    receipt = review_receipt(validation)
    measurement = tmp_path / "measurement.json"
    measurement.write_text('{"synthetic": true}')
    receipt["performance"] = {
        "classification": "measured",
        "assessment": "Synthetic measurement receipt only",
        "measurements": [{"path": str(measurement), "sha256": validate_port.file_hash(measurement)}],
    }
    assert str(measurement) in validate_port.verify_review(receipt, validation.state["plan_sha256"])
    measurement.write_text("changed")
    with pytest.raises(ExportError, match="measurement evidence"):  # allow-pytest.raises: host-only workflow validation
        validate_port.verify_review(receipt, validation.state["plan_sha256"])


def test_port_drift_blocks_resume(port):
    validate_port.initialize(port)
    (Path(port["runtime"]) / "tests/test_cache.py").write_text("# changed\n")
    with pytest.raises(ExportError, match="drift"):  # allow-pytest.raises: host-only workflow validation
        validate_port.PortValidation(port["workspace"]).run()


def test_port_completed_evidence_is_verified(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run("build")
    (validation.workspace / "attempts/build/001/build.log").write_text("changed")
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        ExportError, match="evidence/runtime changed"
    ):
        validation.run()


def test_port_evidence_cannot_enter_target_repo(port):
    port["workspace"] = str(Path(port["runtime"]) / "new-evidence")
    with pytest.raises(ExportError, match="outside the target"):  # allow-pytest.raises: host-only workflow validation
        validate_port.initialize(port)


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("precompile", "yes", "explicit boolean"),
        ("allow_recorded_failures", 1, "explicit boolean"),
        ("precompile_workers", 0, "positive integer"),
        ("command_timeout_seconds", -1, "positive integer"),
        ("environment", {"PYTHONPATH": "/another/runtime"}, "Unsupported environment"),
        ("build_argv", ["sh", "build_metal.sh"], "Build must use"),
        ("build_argv", ["./build_metal.sh", "--clean"], "Unsupported build"),
        ("build_argv", ["./build_metal.sh", "--configure-only"], "Unsupported build"),
        ("cache_test", "../test_cache.py", "[Uu]nsafe|[Ii]nvalid|[Pp]ath"),
        ("cache_test", "tools/test_cache.py", "checked-in-style"),
        ("smoke_nodeid", "tests/test_other.py::test_case", "explicit smoke case"),
        (
            "source_entry",
            "ttnn.operations.unrelated:sample_op",
            "frozen operation package",
        ),
        ("native_entry", "ttnn:invalid.symbol", "module.path:symbol"),
        ("native_entry", "ttnn.operations.sample_op:sample_op", "must differ"),
    ],
)
def test_port_rejects_unsafe_or_ambiguous_configuration(port, key, value, message):
    port[key] = value
    with pytest.raises(ExportError, match=message):  # allow-pytest.raises: host-only workflow validation
        validate_port.plan(port)
    assert not Path(port["workspace"]).exists()


def test_port_new_untracked_source_blocks_resume(port):
    validate_port.initialize(port)
    (Path(port["runtime"]) / "new_factory.cpp").write_text("// new source\n")
    with pytest.raises(ExportError, match="drift"):  # allow-pytest.raises: host-only workflow validation
        validate_port.PortValidation(port["workspace"]).run()


def test_port_rejects_missing_cache_test(port):
    port["cache_test"] = "tests/test_missing.py"
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        ExportError, match="Required validation source"
    ):
        validate_port.plan(port)
