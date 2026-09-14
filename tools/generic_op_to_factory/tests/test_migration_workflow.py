# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic workflow integration: temporary Git repos, fake builds, no device."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from tools.generic_op_to_factory import migration_workflow as flow
from tools.generic_op_to_factory.export_run import ExportError, export_snapshot
from tools.generic_op_to_factory.tests.test_export_run import ReadConnection, snapshot  # noqa: F401
from tools.generic_op_to_factory.tests.test_prepare_baseline import commit, inputs  # noqa: F401


@pytest.fixture
def configured(inputs, snapshot, tmp_path, request, monkeypatch):
    _, evaluator, metal = inputs
    status = getattr(request, "param", "passed")
    scripts = {
        "create_venv.sh": "#!/bin/sh\nexec python3 -m venv --without-pip python_env\n",
        "build_metal.sh": "#!/bin/sh\nexec python3 fake_build.py\n",
        "scripts/run_safe_pytest.sh": '#!/bin/sh\n# SAFE_PYTEST_RAW_EXIT_CODE= and SAFE_PYTEST_WARMUP_JUNIT= supported by fake runner\nexec python3 fake_pytest.py "$@"\n',
        "fake_build.py": """from pathlib import Path
p = Path('build_Release/lib')
p.mkdir(parents=True, exist_ok=True)
(p / '_ttnn.so').write_bytes(b'synthetic library; never dlopened')
print('synthetic build completed')
""",
        "ttnn/__init__.py": """from pathlib import Path
from types import SimpleNamespace
_ttnn = SimpleNamespace(__file__=str(Path(__file__).parent.parent / 'build_Release/lib/_ttnn.so'))
""",
        "fake_pytest.py": """import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
if '--collect-only' in sys.argv:
    print('eval/golden_tests/sample_suite/test_golden.py::test_sample[case]')
    print('1 test collected')
else:
    output = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--junitxml='))
    root = ET.Element('testsuite')
    case = ET.SubElement(root, 'testcase', classname='eval.golden_tests.sample_suite.test_golden', name='test_sample[case]')
    failed = STATUS == 'failed'
    if failed:
        ET.SubElement(case, 'failure', message='synthetic failure')
    ET.ElementTree(root).write(output, encoding='utf-8')
    if '--migration-route' in sys.argv:
        route = json.loads(Path(sys.argv[sys.argv.index('--migration-route') + 1]).read_text())
        print('MIGRATION_ROUTE=' + json.dumps({**route, 'calls': 1}))
    print('SAFE_PYTEST_RAW_EXIT_CODE=' + str(1 if failed else 0))
    print('SAFE_PYTEST_RESULT: FAIL' if failed else 'SAFE_PYTEST_RESULT: PASS')
    sys.exit(1 if failed else 0)
""".replace(
            "STATUS", repr(status)
        ),
        ".gitignore": "python_env/\nbuild_Release/\n__pycache__/\n",
    }
    # This synthetic runtime intentionally ships a fake safe runner, not the
    # hardware wrapper. Scope its approved hash to this fixture only.
    monkeypatch.setattr(
        flow.test_evidence,
        "RUNNER_SHA256",
        flow.hashlib.sha256(scripts["scripts/run_safe_pytest.sh"].encode()).hexdigest(),
    )
    metal_revision = commit(metal, {name: content.encode() for name, content in scripts.items()})
    for name in ("create_venv.sh", "build_metal.sh", "scripts/run_safe_pytest.sh"):
        (metal / name).chmod(0o755)
    metal_revision = commit(metal, {"fixture-marker": b"executable scripts"})
    eval_revision = subprocess.check_output(["git", "-C", str(evaluator), "rev-parse", "HEAD"], text=True).strip()
    connection, run_id = snapshot
    connection.execute("PRAGMA query_only = OFF")
    connection.execute(
        "UPDATE runs SET starting_commit=?, eval_commit=? WHERE id=?",
        (metal_revision, eval_revision, run_id),
    )
    connection.execute(
        "UPDATE test_results SET test_file=?, status=?",
        ("eval.golden_tests.sample_suite.test_golden", status),
    )
    connection.execute("PRAGMA query_only = ON")
    exported = tmp_path / "workflow-export"
    export_snapshot(ReadConnection(connection), run_id, exported, database={"host": "synthetic"})
    return {
        "export": str(exported),
        "metal_repository": str(metal),
        "eval_repository": str(evaluator),
        "workspace": str(tmp_path / "workspace"),
        "target_revision": metal_revision,
        "phase": "initial",
        "precompile": False,
        "capture_metrics": False,
        "build_argv": ["./build_metal.sh", "--enable-ccache"],
    }


def test_plan_is_read_only(configured):
    planned = flow.plan(configured)
    assert not Path(configured["workspace"]).exists()
    assert planned["recorded_case_count"] == 1
    assert planned["config"]["phase"] == "initial"
    assert planned["future_gates"] == list(flow.FUTURE_GATES)
    assert not planned["migration_ready"]


@pytest.mark.parametrize("configured", ["passed", "failed"], indirect=True)
def test_complete_workflow_and_noop_resume(configured):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    state = workflow.run()
    assert all(record["status"] == "complete" for record in state["stages"].values())
    comparison = workflow.receipt("compare")["result"]
    assert comparison["outcomes_match"]
    assert not comparison["migration_ready"]
    assert state["future_gates"]["cpp_translation"] == "not_implemented"
    before = (workflow.workspace / "state.json").read_bytes()
    with mock.patch.object(workflow, "execute", side_effect=AssertionError("must not rerun")):
        workflow.run()
    assert (workflow.workspace / "state.json").read_bytes() == before
    command_path = workflow.workspace / state["stages"]["baseline"]["attempts"][0] / "baseline.command.json"
    command = json.loads(command_path.read_bytes())
    assert "./scripts/run_safe_pytest.sh" in command["argv"]
    assert "--run-all" in command["argv"] and "--no-precompile" in command["argv"]
    assert "EVAL_DATABASE_URL" not in command["environment_overrides"]


def test_checkpoint_resume_and_source_drift(configured):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    workflow.run("install")
    assert workflow.state["stages"]["build"]["status"] == "pending"
    installed = workflow.runtime / "ttnn/ttnn/operations/sample_op/planner.py"
    installed.write_text("# changed\n")
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        ExportError, match="source/dependency changed"
    ):
        flow.Workflow(workflow.workspace).run()
    assert not (workflow.runtime / "python_env").exists()


def test_changed_export_blocks_resume(configured):
    flow.initialize(configured)
    (Path(configured["export"]) / "source/planner.py").write_text("modified")
    with pytest.raises(ExportError, match="checksum"):  # allow-pytest.raises: host-only workflow validation
        flow.Workflow(configured["workspace"]).run()


def test_changed_implementation_blocks_resume(configured, monkeypatch):
    flow.initialize(configured)
    monkeypatch.setattr(flow, "implementation", lambda: {"changed": "implementation"})
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        ExportError, match="implementation changed"
    ):
        flow.Workflow(configured["workspace"]).run()


def test_changed_evidence_blocks_resume(configured):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    workflow.run("checkout")
    (workflow.workspace / "attempts/checkout/001/submodules.log").write_text("modified evidence")
    with pytest.raises(ExportError, match="Changed evidence"):  # allow-pytest.raises: host-only workflow validation
        flow.Workflow(workflow.workspace).run()


def test_changed_binary_blocks_resume(configured):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    workflow.run("build")
    (workflow.runtime / "build_Release/lib/_ttnn.so").write_bytes(b"changed binary")
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        ExportError, match="Changed evidence/runtime output"
    ):
        workflow.run()
    assert workflow.state["stages"]["collect"]["status"] == "pending"


def test_changed_python_environment_blocks_resume(configured):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    workflow.run("build")
    packages = next((workflow.runtime / "python_env/lib").glob("python*/site-packages"))
    metadata = packages / "synthetic_added-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: synthetic-added\nVersion: 1.0\n")
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        ExportError, match="Python package environment changed"
    ):
        workflow.run()


@pytest.mark.parametrize("target", ["runtime", "attempts"])
def test_redirected_workflow_directories_block(configured, tmp_path, target):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    outside = tmp_path / "outside"
    outside.mkdir()
    (workflow.workspace / target).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ExportError, match="redirected"):  # allow-pytest.raises: host-only workflow validation
        workflow.run()
    assert not list(outside.iterdir())


def test_mismatch_keeps_comparison_and_blocks_migration(configured):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    command = workflow.command

    def different_outcome(argv, attempt, label, **kwargs):
        code = command(argv, attempt, label, **kwargs)
        if label == "baseline":
            junit = attempt / "junit.xml"
            junit.write_text(junit.read_text().replace("test_sample[case]", "test_sample[other]"))
        return code

    with mock.patch.object(workflow, "command", side_effect=different_outcome):
        with pytest.raises(  # allow-pytest.raises: host-only workflow validation
            ExportError, match="differs from selected historical phase"
        ):
            workflow.run()
    comparison = json.loads((workflow.workspace / "attempts/compare/001/comparison.json").read_bytes())
    assert not comparison["outcomes_match"]
    assert workflow.state["stages"]["compare"]["status"] == "blocked"
    assert not workflow.state["migration_ready"]


@pytest.mark.parametrize("failure", ["missing_xml", "hang"])
def test_failed_execution_cannot_reach_comparison(configured, failure):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    command = workflow.command

    def broken_execution(argv, attempt, label, **kwargs):
        code = command(argv, attempt, label, **kwargs)
        if label == "baseline":
            if failure == "missing_xml":
                (attempt / "junit.xml").unlink()
            else:
                with (attempt / "baseline.log").open("a") as stream:
                    stream.write("SAFE_PYTEST_RESULT: HANG\n")
        return code

    with mock.patch.object(workflow, "command", side_effect=broken_execution):
        with pytest.raises((ExportError, FileNotFoundError)):  # allow-pytest.raises: host-only workflow validation
            workflow.run()
    assert workflow.state["stages"]["baseline"]["status"] == "blocked"
    assert workflow.state["stages"]["compare"]["status"] == "pending"


def test_explicit_smoke_precompile_and_metrics(configured, monkeypatch):
    # This fixture simulates command results, not warmup. Real runner admission
    # and crash/warmup separation have dedicated test_test_evidence regressions.
    monkeypatch.setattr(flow.test_evidence, "check_runner", lambda *args, **kwargs: None)
    configured.update(
        precompile=True,
        capture_metrics=True,
        precompile_workers=3,
        smoke_nodeid="eval/golden_tests/sample_suite/test_golden.py::test_sample[case]",
    )
    monkeypatch.setenv("EVAL_DATABASE_URL", "do-not-forward")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k unintended_filter")
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    state = workflow.run()
    assert not workflow.receipt("smoke")["result"].get("skipped")
    for stage in ("collect", "smoke", "baseline"):
        attempt = workflow.workspace / state["stages"][stage]["attempts"][0]
        command = json.loads((attempt / f"{stage}.command.json").read_bytes())
        argv = command["argv"]
        if stage == "baseline":
            assert argv[argv.index("--precompile-workers") + 1] == "3"
            assert "--precompile" in argv and "--no-precompile" not in argv
        else:
            assert "--no-precompile" in argv
        env, _ = workflow.environment(attempt, device=stage != "collect")
        assert "EVAL_DATABASE_URL" not in env
        assert env["PYTEST_ADDOPTS"] == ""
        assert ("TT_METAL_DEVICE_PROFILER" in env) == (stage != "collect")


def test_exclusive_workspace_and_lock(configured):
    flow.initialize(configured)
    with pytest.raises(FileExistsError):  # allow-pytest.raises: host-only workflow validation
        flow.initialize(configured)
    with flow.locked(Path(configured["workspace"])):
        with pytest.raises(ExportError, match="Another driver"):  # allow-pytest.raises: host-only workflow validation
            flow.Workflow(configured["workspace"]).run()


def test_retry_requires_explicit_action_and_retains_attempts(configured):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    workflow.run("install")
    original = workflow.execute

    def fail(stage, attempt):
        if stage == "environment":
            raise ExportError("synthetic setup failure")
        return original(stage, attempt)

    with mock.patch.object(workflow, "execute", side_effect=fail):
        with pytest.raises(ExportError, match="synthetic"):  # allow-pytest.raises: host-only workflow validation
            workflow.run("environment")
    with pytest.raises(ExportError, match="explicitly --retry"):  # allow-pytest.raises: host-only workflow validation
        workflow.run("environment")
    workflow.run("environment", retry=True)
    assert len(workflow.state["stages"]["environment"]["attempts"]) == 2
    assert (workflow.workspace / "attempts/environment/001").is_dir()


def test_interrupt_marks_checkpoint_and_cannot_succeed(configured):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    with mock.patch.object(workflow, "execute", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):  # allow-pytest.raises: host-only workflow validation
            workflow.run("prepare")
    assert flow.Workflow(workflow.workspace).state["stages"]["prepare"]["status"] == "interrupted"
    workflow.run("prepare", retry=True)


def test_do_not_retry_live_unfinished_command(configured):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    attempt = workflow.workspace / "attempts/prepare/001"
    attempt.mkdir(parents=True)
    flow.write_json(attempt / "child.command.json", {"pid": os.getpid()})
    workflow.state["stages"]["prepare"].update(status="running", attempts=["attempts/prepare/001"])
    workflow.save()
    with pytest.raises(ExportError, match="may still be alive"):  # allow-pytest.raises: host-only workflow validation
        workflow.run(retry=True)


def test_subprocess_interrupt_signals_only_owned_group(configured):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    attempt = workflow.workspace / "mock-attempt"
    attempt.mkdir()
    child = mock.Mock(pid=123456)
    child.wait.side_effect = [KeyboardInterrupt, -15]
    child.poll.return_value = -15
    with mock.patch.object(flow.subprocess, "Popen", return_value=child), mock.patch.object(flow.os, "killpg") as kill:
        with pytest.raises(KeyboardInterrupt):  # allow-pytest.raises: host-only workflow validation
            workflow.command(["not-executed"], attempt, "interrupted")
    kill.assert_called_once_with(child.pid, flow.signal.SIGTERM)
    assert json.loads((attempt / "interrupted.command.json").read_bytes())["interrupted"]


def test_unknown_unfinished_process_blocks_retry(configured):
    flow.initialize(configured)
    workflow = flow.Workflow(configured["workspace"])
    attempt = workflow.workspace / "attempts/prepare/001"
    attempt.mkdir(parents=True)
    flow.write_json(attempt / "child.command.json", {"started_at": flow.now()})
    workflow.state["stages"]["prepare"].update(status="running", attempts=["attempts/prepare/001"])
    workflow.save()
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        ExportError, match="unknown process identity"
    ):
        workflow.run(retry=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "absent"),
        ("precompile", "yes"),
        ("precompile_workers", 0),
        ("build_argv", ["git", "reset", "--hard"]),
        ("build_argv", ["./build_metal.sh", "--clean"]),
        ("build_argv", ["./build_metal.sh", "--configure-only"]),
        ("environment", {"EVAL_DATABASE_URL": "must-not-be-stored"}),
        ("evaluator_path", "../other"),
        ("configure_argv", ["cmake", "-E", "remove_directory", "/"]),
        ("smoke_nodeid", "eval/golden_tests/unrelated/test_golden.py::test_sample"),
    ],
)
def test_invalid_configuration_rejected_before_mutation(configured, field, value):
    configured[field] = value
    with pytest.raises(ExportError):  # allow-pytest.raises: host-only workflow validation
        flow.initialize(configured)
    assert not Path(configured["workspace"]).exists()


def test_workspace_cannot_overlap_source(configured):
    configured["workspace"] = str(Path(configured["metal_repository"]) / "workflow")
    with pytest.raises(ExportError, match="disjoint"):  # allow-pytest.raises: host-only workflow validation
        flow.initialize(configured)


def test_status_cli_does_not_mutate_workspace(configured):
    flow.initialize(configured)
    workspace = Path(configured["workspace"])
    before = {path.name: path.read_bytes() for path in workspace.iterdir()}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.generic_op_to_factory.migration_workflow",
            "status",
            "--workspace",
            str(workspace),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["stages"]["prepare"]["status"] == "pending"
    assert {path.name: path.read_bytes() for path in workspace.iterdir()} == before
