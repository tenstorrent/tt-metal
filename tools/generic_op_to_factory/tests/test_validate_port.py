# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic target/native orchestration; never opens a real device."""

import json
import shutil
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
            "tools/generic_op_to_factory/native_adapter.py": Path(validate_port.__file__)
            .with_name("native_adapter.py")
            .read_bytes(),
            "tests/test_cache.py": b"# fake runner emits a passing cache test\n",
            "ttnn/cpp/ttnn/operations/sample/device/sample.hpp": b"// synthetic operation header\n",
            "ttnn/cpp/ttnn/operations/sample/device/sample_program_factory.cpp": b"// synthetic factory\n",
            "fake_contract_compiler.py": b"print('synthetic compile; type rules are tested with a real host compiler separately')\n",
        },
    )
    factory_source = baseline.runtime / "ttnn/cpp/ttnn/operations/sample/device/sample_program_factory.cpp"
    (baseline.runtime / "build_Release/compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": str(baseline.runtime / "build_Release"),
                    "file": str(factory_source),
                    "arguments": [
                        "python3",
                        str(baseline.runtime / "fake_contract_compiler.py"),
                        "-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK",
                        "-c",
                        str(factory_source),
                        "-o",
                        "ignored.o",
                    ],
                }
            ]
        )
    )
    (baseline.runtime / "build_Release/CMakeCache.txt").write_text("ENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK:BOOL=ON\n")
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
        "acceptance_tests": ["tests/test_cache.py"],
        "build_argv": ["./build_metal.sh"],
        "precompile": False,
        "allow_recorded_failures": False,
        "factory_contract": {
            "operation_header": "ttnn/cpp/ttnn/operations/sample/device/sample.hpp",
            "operation_type": "sample::DeviceOperation",
            "factory_source": str(factory_source.relative_to(baseline.runtime)),
        },
    }


def test_port_plan_read_only(port):
    planned = validate_port.plan(port)
    assert planned["recorded_case_count"] == 1
    assert not Path(port["workspace"]).exists()


def test_parity_disabled_blocks_before_device_tests(port):
    cache = Path(port["runtime"]) / "build_Release/CMakeCache.txt"
    cache.write_text("ENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK:BOOL=OFF\n")
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    with pytest.raises(ExportError, match="PARITY_CHECK=ON"):  # allow-pytest.raises: build instrumentation gate
        validation.run("acceptance")
    assert validation.state["stages"]["factory_contract"]["status"] == "blocked"
    assert validation.state["stages"]["source"]["status"] == "pending"
    assert not list(validation.workspace.glob("attempts/*/*/junit.xml"))


def test_parity_configuration_is_recorded_and_cannot_drift(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run("acceptance")
    attempt = validation.workspace / "attempts/factory_contract/001"
    assert json.loads((attempt / "contract.json").read_text())["descriptor_patching_parity_enabled"]
    cache = Path(port["runtime"]) / "build_Release/CMakeCache.txt"
    assert str(cache) in validation.state["stages"]["factory_contract"]["evidence"]
    cache.write_text("ENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK:BOOL=OFF\n")
    with pytest.raises(ExportError, match="evidence/runtime changed"):  # allow-pytest.raises: instrumentation drift
        validation.run()


def test_review_requires_cache_hit_parity_coverage(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    receipt = review_receipt(validation)
    del receipt["topics"]["descriptor_cache_hit_parity"]
    with pytest.raises(ExportError, match="all required topics"):  # allow-pytest.raises: coverage review gate
        validate_port.verify_review(receipt, validation.state["plan_sha256"])


@pytest.mark.parametrize("value", [[], "tests/test_contract.py", [None], ["tests/test_cache.py"] * 2])
def test_acceptance_requires_explicit_nonempty_unique_file_list(port, value):
    port["acceptance_tests"] = value
    with pytest.raises(ExportError, match="acceptance_tests"):  # allow-pytest.raises: acceptance selection gate
        validate_port.plan(port)


def test_old_cache_only_config_is_not_silently_reinterpreted(port):
    port["cache_test"] = port.pop("acceptance_tests")[0]
    with pytest.raises(ExportError, match="config keys"):  # allow-pytest.raises: renamed required input
        validate_port.plan(port)


def test_multiple_acceptance_files_run_once_with_direct_native_route(port):
    runtime = Path(port["runtime"])
    (runtime / "tests/test_contract.py").write_text("# additional API/output tests\n")
    port["acceptance_tests"].append("tests/test_contract.py")
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run("acceptance")
    attempt = validation.workspace / "attempts/acceptance/001"
    command = json.loads((attempt / "acceptance.command.json").read_bytes())
    assert all(test in command["argv"] for test in port["acceptance_tests"])
    assert "--migration-acceptance-route" in command["argv"]
    route = json.loads((attempt / "route.json").read_bytes())
    assert route["mode"] == "native"
    assert route["tests"] == port["acceptance_tests"]
    assert json.loads((attempt / "execution.json").read_bytes())["route"]["calls"] > 0
    assert len(validation.state["stages"]["acceptance"]["attempts"]) == 1
    assert not (validation.workspace / "attempts/source_acceptance").exists()


@pytest.mark.parametrize("status", ["failed", "error", "skipped", "xfail", "xpass"])
def test_acceptance_rejects_every_nonpassing_outcome(port, monkeypatch, status):
    original = validate_port.parse_junit_xml

    def outcomes(path):
        rows = original(path)
        if "acceptance" in Path(path).parts:
            rows[0]["status"] = status
        return rows

    monkeypatch.setattr(validate_port, "parse_junit_xml", outcomes)
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    with pytest.raises(ExportError, match="Every selected native acceptance test"):  # allow-pytest.raises: gate
        validation.run()
    assert validation.state["stages"]["acceptance"]["status"] == "blocked"
    assert validation.state["stages"]["review"]["status"] == "pending"


@pytest.mark.parametrize("omit", [True, False])
def test_final_validation_runs_only_two_full_suites_after_acceptance(port, omit):
    if omit:
        del port["smoke_nodeid"]
    else:
        port["smoke_nodeid"] = None
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run("native_compare")
    for stage in ("source_smoke", "native_smoke"):
        attempt = validation.workspace / f"attempts/{stage}/001"
        assert (attempt / "skipped.json").is_file()
        assert not list(attempt.glob("*.command.json"))
    real = list(validation.workspace.glob("attempts/*/001/junit.xml"))
    assert {path.parent.parent.name for path in real} == {"source", "native", "acceptance"}


def test_stale_target_adapter_is_refused(port):
    (Path(port["runtime"]) / "tools/generic_op_to_factory/native_adapter.py").write_text("# stale")
    with pytest.raises(ExportError, match="must match this flow"):  # allow-pytest.raises: host-only adapter gate
        validate_port.plan(port)


def test_factory_contract_is_required(port):
    del port["factory_contract"]
    with pytest.raises(ExportError, match="config keys"):  # allow-pytest.raises: host-only config validation
        validate_port.plan(port)


def test_compilation_database_stdout_is_separate_from_diagnostics(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    attempt = validation.workspace / "stdout-test"
    attempt.mkdir()
    validation.runner.command(
        ["python3", "-c", "import sys; print('[]'); print('compiler metadata warning', file=sys.stderr)"],
        attempt,
        "compdb",
        stdout_file="database.json",
    )
    assert json.loads((attempt / "database.json").read_text()) == []
    assert (attempt / "compdb.log").read_text() == "compiler metadata warning\n"


@pytest.mark.parametrize("aliases", ["ttnn:alias", [None], ["bad-symbol"], ["ttnn:alias", "ttnn:alias"]])
def test_source_alias_configuration_is_validated(port, aliases):
    port["source_aliases"] = aliases
    with pytest.raises(ExportError):  # allow-pytest.raises: host-only alias config
        validate_port.plan(port)


def test_source_aliases_are_preserved_in_route_evidence(port):
    port["source_aliases"] = ["ttnn:sample_alias"]
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run("source_smoke")
    route = validation.workspace / "attempts/source_smoke/001/route.json"
    assert json.loads(route.read_text())["aliases"] == port["source_aliases"]


def test_native_entry_cannot_be_a_source_alias(port):
    port["source_aliases"] = [port["native_entry"]]
    with pytest.raises(ExportError, match="must differ"):  # allow-pytest.raises: host-only alias config
        validate_port.plan(port)


def test_alias_config_drift_invalidates_resume(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.config["source_aliases"] = ["ttnn:added_alias"]
    with pytest.raises(ExportError, match="identity changed"):  # allow-pytest.raises: host-only evidence guard
        validation.validate()


def test_failed_factory_contract_blocks_device_stages(port):
    runtime = Path(port["runtime"])
    (runtime / "fake_contract_compiler.py").write_text("raise SystemExit(1)\n")
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    with pytest.raises(  # allow-pytest.raises: synthetic compiler failure
        ExportError, match="factory-contract exited 1"
    ):
        validation.run()
    assert validation.state["stages"]["factory_contract"]["status"] == "blocked"
    assert validation.state["stages"]["source_smoke"]["status"] == "pending"


def test_factory_contract_evidence_is_preserved(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run("factory_contract")
    attempt = validation.workspace / "attempts/factory_contract/001"
    assert json.loads((attempt / "contract.json").read_text())["kind"] == "ProgramDescriptor"
    assert "ProgramDescriptorFactoryConcept" in (attempt / "factory_contract.cpp").read_text()
    assert (attempt / "factory-contract.command.json").is_file()
    command = json.loads((attempt / "factory-contract.command.json").read_text())
    assert command["cwd"] == str(Path(port["runtime"]) / "build_Release")
    database = Path(port["runtime"]) / "build_Release/compile_commands.json"
    database.write_text("[]")
    with pytest.raises(ExportError, match="evidence/runtime changed"):  # allow-pytest.raises: evidence drift validation
        validation.run()


def test_ninja_factory_gate_keeps_unity_configuration_and_fingerprints_metadata(port):
    if shutil.which("ninja") is None:
        pytest.skip("Ninja unavailable")
    runtime = Path(port["runtime"])
    build = runtime / "build_Release"
    factory = runtime / port["factory_contract"]["factory_source"]
    unity = build / "unity_0_cxx.cxx"
    unity.write_text(f'#include "{factory}"\n')
    ninja = build / "build.ninja"
    ninja.write_text(
        "rule compile\n"
        f"  command = python3 {runtime / 'fake_contract_compiler.py'} -DTT_DESCRIPTOR_PATCHING_PARITY_CHECK -c $in -o $out\n"
        f"build ignored.o: compile {unity}\n"
    )
    # A partial CMake database must not force a rebuild or hide the Ninja entry.
    (build / "compile_commands.json").write_text("[]")
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run("factory_contract")
    attempt = validation.workspace / "attempts/factory_contract/001"
    extraction = json.loads((attempt / "compile-database.command.json").read_text())
    assert extraction["argv"] == ["ninja", "-C", str(build), "-t", "compdb"]
    evidence = validation.state["stages"]["factory_contract"]["evidence"]
    assert str(ninja) in evidence
    assert str(unity) in evidence
    ninja.write_text(ninja.read_text() + "# changed metadata\n")
    with pytest.raises(ExportError, match="evidence/runtime changed"):  # allow-pytest.raises: evidence drift validation
        validation.run()


def test_complete_synthetic_port_and_resume(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run("acceptance")
    record_review(validation)
    state = validation.run("complete")
    assert all(r["status"] == "complete" for r in state["stages"].values())
    before = (validation.workspace / "state.json").read_bytes()
    validation.run("complete")
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
        validation.run("complete")
    assert validation.state["stages"]["complete"]["status"] == "pending"
    prior_cache = dict(validation.state["stages"]["acceptance"])
    record_review(validation)
    validation.run("complete", retry=True)
    assert validation.state["stages"]["acceptance"] == prior_cache


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
    validation.run("acceptance")
    record_review(validation)
    validation.run("complete")
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


def test_acceptance_contract_cannot_change_during_factory_iteration(port):
    validate_port.initialize(port)
    (Path(port["runtime"]) / "tests/test_cache.py").write_text("# changed\n")
    with pytest.raises(ExportError, match="acceptance tests changed"):  # allow-pytest.raises: fixed test contract
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
        ("acceptance_tests", ["../test_cache.py"], "[Uu]nsafe|[Ii]nvalid|[Pp]ath"),
        ("acceptance_tests", ["tools/test_cache.py"], "checked-in-style"),
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


def test_port_new_untracked_source_starts_iteration(port):
    validate_port.initialize(port)
    (Path(port["runtime"]) / "new_factory.cpp").write_text("// new source\n")
    validation = validate_port.PortValidation(port["workspace"])
    validation.run()
    assert "new_factory.cpp" in validation.planned["untracked_files"]
    assert validation.state["stages"]["acceptance"]["status"] == "complete"


def test_default_iteration_stops_at_acceptance_before_goldens(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run()
    assert list(validate_port.STAGES[:3]) == ["build", "factory_contract", "acceptance"]
    assert validation.state["stages"]["acceptance"]["status"] == "complete"
    assert all(validation.state["stages"][stage]["status"] == "pending" for stage in validate_port.STAGES[3:])
    assert not (validation.workspace / "attempts/source").exists()


def test_factory_edit_supersedes_completed_results_in_same_workspace(port):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run()
    record_review(validation)
    validation.run("complete")
    previous_sha = validation.state["plan_sha256"]
    old_log = validation.workspace / "attempts/acceptance/001/acceptance.log"
    old_bytes = old_log.read_bytes()
    (Path(port["runtime"]) / port["factory_contract"]["factory_source"]).write_text("// corrected factory\n")
    # A new process, same command and workspace; no re-init or --retry needed.
    validation = validate_port.PortValidation(port["workspace"])
    validation.run()
    assert validation.state["plan_sha256"] != previous_sha
    assert old_log.read_bytes() == old_bytes
    for stage in ("build", "factory_contract", "acceptance"):
        assert len(validation.state["stages"][stage]["attempts"]) == 2
        assert validation.state["stages"][stage]["status"] == "complete"
    for stage in validate_port.STAGES[3:]:
        assert validation.state["stages"][stage]["status"] == "pending"
    with pytest.raises(ExportError, match="review must match"):  # allow-pytest.raises: stale review
        validation.run("complete")
    record_review(validation)
    validation.run("complete", retry=True)
    assert validation.state["stages"]["complete"]["status"] == "complete"


def test_failed_acceptance_fix_rebuilds_and_retests_in_place(port, monkeypatch):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    original = validation.execute

    def fail_acceptance(stage, attempt):
        if stage == "acceptance":
            raise ExportError("Synthetic factory defect")
        return original(stage, attempt)

    monkeypatch.setattr(validation, "execute", fail_acceptance)
    with pytest.raises(ExportError, match="factory defect"):  # allow-pytest.raises: synthetic failure
        validation.run()
    (Path(port["runtime"]) / port["factory_contract"]["factory_source"]).write_text("// fix\n")
    monkeypatch.setattr(validation, "execute", original)
    validation.run()
    assert validation.state["stages"]["acceptance"]["status"] == "complete"
    assert len(validation.state["stages"]["build"]["attempts"]) == 2
    assert not (validation.workspace / "attempts/source").exists()


def test_edit_during_stage_does_not_receive_a_pass(port, monkeypatch):
    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    original = validation.execute

    def edit_while_testing(stage, attempt):
        result = original(stage, attempt)
        if stage == "acceptance":
            (Path(port["runtime"]) / port["factory_contract"]["factory_source"]).write_text("// concurrent edit\n")
        return result

    monkeypatch.setattr(validation, "execute", edit_while_testing)
    with pytest.raises(ExportError, match="Source changed during validation"):  # allow-pytest.raises: freshness
        validation.run()
    assert validation.state["stages"]["acceptance"]["status"] == "blocked"


def test_factory_edit_cannot_bypass_live_command_guard(port):
    import os

    validate_port.initialize(port)
    validation = validate_port.PortValidation(port["workspace"])
    validation.run()
    (validation.workspace / "attempts/acceptance/001/live.command.json").write_text(json.dumps({"pid": os.getpid()}))
    (Path(port["runtime"]) / port["factory_contract"]["factory_source"]).write_text("// fix\n")
    with pytest.raises(ExportError, match="may still be alive"):  # allow-pytest.raises: process guard
        validation.run()
    assert len(validation.state["stages"]["build"]["attempts"]) == 1


def test_port_rejects_missing_acceptance_tests(port):
    port["acceptance_tests"] = ["tests/test_missing.py"]
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        ExportError, match="Required validation source"
    ):
        validate_port.plan(port)
