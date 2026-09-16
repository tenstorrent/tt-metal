# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Evaluated-branch tests use temporary repositories and synthetic names only."""

import json
import subprocess
from pathlib import Path

import pytest

from tools.generic_op_to_factory import prepare_branch as branch, validate_port
from tools.generic_op_to_factory.export_run import ExportError
from tools.generic_op_to_factory.tests.test_export_run import snapshot  # noqa: F401
from tools.generic_op_to_factory.tests.test_prepare_baseline import commit, git, inputs  # noqa: F401
from tools.generic_op_to_factory.tests.test_migration_workflow import configured  # noqa: F401
from tools.generic_op_to_factory.tests.test_validate_port import record_review


@pytest.fixture
def evaluated(configured, tmp_path, monkeypatch):
    repository = Path(configured["metal_repository"])
    evaluator = Path(configured["eval_repository"])
    # No network, no real eval source or card. Exercise actual recursive gitlinks.
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    git(repository, "submodule", "add", str(evaluator), "vendor/evaluator")
    (repository / "eval").symlink_to("vendor/evaluator/eval", target_is_directory=True)
    revision = commit(
        repository,
        {
            "ttnn/ttnn/operations/sample_op/__init__.py": b"# complete evaluated operation\n",
            "ttnn/ttnn/operations/sample_op/planner.py": b"# evaluated planner, not DB export\n",
            "ttnn/cpp/ttnn-nanobind/tensor.cpp": b"// supporting API added during evaluation\n",
            "ttnn/cpp/ttnn/kernel_lib/run_helper.hpp": b"// supporting kernel header added during evaluation\n",
        },
    )
    git(repository, "branch", "evaluated-candidate")
    return {
        "repository": repository,
        "branch": "evaluated-candidate",
        "runtime": tmp_path / "native-runtime",
        "operation": "sample_op",
        "golden_suite": "sample_suite",
        "output": tmp_path / "native-runtime" / branch.ARTIFACT_DIR / "inputs",
        "revision": revision,
    }


def prepare(evaluated):
    return branch.prepare(**{key: value for key, value in evaluated.items() if key != "revision"})


def test_complete_final_tree_and_own_evaluator_are_preserved(evaluated):
    result = prepare(evaluated)
    runtime = evaluated["runtime"]
    assert result["source_revision"] == evaluated["revision"]
    assert result["source_branch"] == "refs/heads/evaluated-candidate"
    assert (
        runtime / "ttnn/cpp/ttnn-nanobind/tensor.cpp"
    ).read_bytes() == b"// supporting API added during evaluation\n"
    assert (runtime / "ttnn/cpp/ttnn/kernel_lib/run_helper.hpp").is_file()
    assert (
        runtime / "ttnn/ttnn/operations/sample_op/planner.py"
    ).read_bytes() == b"# evaluated planner, not DB export\n"
    assert result["submodules"]["vendor/evaluator"] == git(runtime / "vendor/evaluator", "rev-parse", "HEAD")
    assert git(runtime, "status", "--porcelain") == ""
    commands = json.loads((evaluated["output"] / "commands.json").read_bytes())
    assert commands[1]["argv"][-4:] == ["update", "--init", "--recursive", "--checkout"]
    assert commands[0]["argv"][-1] == evaluated["revision"]


@pytest.mark.parametrize("name", ["missing-branch", "HEAD", "--bad", ""])
def test_missing_branch_never_falls_back_to_starting_commit(evaluated, name):
    evaluated["branch"] = name
    with pytest.raises(ExportError, match="branch|Branch"):  # allow-pytest.raises: host-only branch input gate
        prepare(evaluated)
    assert not evaluated["runtime"].exists()


@pytest.mark.parametrize(
    "relative", ["uncommitted.hpp", "ttnn/cpp/ttnn-nanobind/tensor.cpp", "vendor/evaluator/eval/metrics.py"]
)
def test_dirty_evaluated_tree_is_not_silently_lost(evaluated, relative):
    path = evaluated["repository"] / relative
    path.write_text("uncheckpointed run change\n")
    with pytest.raises(ExportError, match="dirty|checkpoint"):  # allow-pytest.raises: source preservation
        prepare(evaluated)
    assert path.read_text() == "uncheckpointed run change\n"
    assert not evaluated["runtime"].exists()


def test_missing_operation_is_not_installed_from_export(evaluated):
    evaluated["operation"] = "missing_op"
    with pytest.raises(ExportError, match="Operation is absent"):  # allow-pytest.raises: branch completeness
        prepare(evaluated)


def test_existing_runtime_is_never_overwritten(evaluated):
    evaluated["runtime"].mkdir()
    marker = evaluated["runtime"] / "keep"
    marker.write_text("user file")
    with pytest.raises(ExportError, match="must be new"):  # allow-pytest.raises: exclusive target reservation
        prepare(evaluated)
    assert marker.read_text() == "user file"


def test_branch_can_move_after_snapshot_without_retargeting_migration(evaluated):
    result = prepare(evaluated)
    newer = commit(evaluated["repository"], {"later.txt": b"later evaluation"})
    git(evaluated["repository"], "update-ref", "refs/heads/evaluated-candidate", newer)
    assert branch.inspect(evaluated["output"], evaluated["runtime"])["source_revision"] == result["source_revision"]


@pytest.mark.parametrize(
    "relative",
    [
        "ttnn/ttnn/operations/sample_op/planner.py",
        "eval/golden_tests/sample_suite/test_golden.py",
        "vendor/evaluator/eval/metrics.py",
    ],
)
def test_original_source_and_tests_cannot_be_allowlisted(evaluated, relative):
    prepare(evaluated)
    with pytest.raises(ExportError, match="cannot replace"):  # allow-pytest.raises: frozen original source
        branch.inspect(evaluated["output"], evaluated["runtime"], [relative])


def test_unlisted_supporting_change_is_rejected(evaluated):
    prepare(evaluated)
    (evaluated["runtime"] / "ttnn/cpp/ttnn-nanobind/tensor.cpp").write_text("changed API")
    with pytest.raises(ExportError, match="Undeclared change"):  # allow-pytest.raises: source drift
        branch.inspect(evaluated["output"], evaluated["runtime"])


def test_submodule_revision_drift_is_rejected(evaluated):
    prepare(evaluated)
    commit(evaluated["runtime"] / "vendor/evaluator", {"eval/metrics.py": b"changed evaluator"})
    with pytest.raises(  # allow-pytest.raises: evaluator drift
        ExportError, match="differs from evaluated branch gitlink"
    ):
        branch.inspect(evaluated["output"], evaluated["runtime"])


def test_snapshot_tampering_is_rejected(evaluated):
    prepare(evaluated)
    path = evaluated["output"] / "branch.json"
    data = json.loads(path.read_text())
    data["source_revision"] = "0" * 40
    path.write_text(json.dumps(data))
    with pytest.raises(ExportError, match="checksum"):  # allow-pytest.raises: evidence integrity
        branch.inspect(evaluated["output"], evaluated["runtime"])


def test_reviewed_tools_are_explicit_changes_not_source_reconstruction(evaluated, monkeypatch):
    prepare(evaluated)
    original = (evaluated["repository"] / "scripts/run_safe_pytest.sh").read_bytes()
    root = Path(branch.__file__).resolve().parents[2]
    monkeypatch.setattr(
        branch.test_evidence, "RUNNER_SHA256", branch._hash_file(root / "scripts/run_safe_pytest.sh")[0]
    )
    branch.install_tools(evaluated["runtime"])
    result = branch.inspect(evaluated["output"], evaluated["runtime"])
    assert set(result["tooling_updates"]) == set(branch.TOOL_FILES)
    assert (evaluated["repository"] / "scripts/run_safe_pytest.sh").read_bytes() == original


def test_submodules_marked_update_none_still_use_evaluated_gitlink(evaluated):
    repository = evaluated["repository"]
    git(repository, "config", "-f", ".gitmodules", "submodule.vendor/evaluator.update", "none")
    revision = commit(repository, {"checkpoint-note": b"evaluated submodule policy"})
    git(repository, "update-ref", "refs/heads/evaluated-candidate", revision)
    result = prepare(evaluated)
    assert result["submodules"]["vendor/evaluator"] == git(
        evaluated["runtime"] / "vendor/evaluator", "rev-parse", "HEAD"
    )


def test_remote_tracking_branch_is_resolved_explicitly(evaluated):
    git(evaluated["repository"], "update-ref", "refs/remotes/origin/evaluated-copy", evaluated["revision"])
    evaluated["branch"] = "origin/evaluated-copy"
    assert prepare(evaluated)["source_branch"] == "refs/remotes/origin/evaluated-copy"


def test_missing_golden_retains_failed_attempt_without_borrowing_tests(evaluated):
    evaluated["golden_suite"] = "absent_suite"
    with pytest.raises(ExportError, match="Golden suite is missing"):  # allow-pytest.raises: no evaluator substitution
        prepare(evaluated)
    assert evaluated["runtime"].is_dir()
    assert (evaluated["output"] / "commands.json").is_file()
    assert not (evaluated["output"] / "branch.json").exists()


@pytest.fixture
def branch_port(evaluated):
    prepare(evaluated)
    runtime = evaluated["runtime"]
    files = {
        "tools/generic_op_to_factory/native_adapter.py": Path(validate_port.__file__)
        .with_name("native_adapter.py")
        .read_bytes(),
        "tests/test_cache.py": b"# synthetic cache test\n",
        "ttnn/cpp/ttnn/operations/sample/device/sample.hpp": b"// synthetic operation header\n",
        "ttnn/cpp/ttnn/operations/sample/device/sample_program_factory.cpp": b"// synthetic factory\n",
        "fake_contract_compiler.py": b"print('synthetic compiler')\n",
    }
    commit(runtime, files)
    # Runtime bootstrap is explicit and uses this checkout's own environment.
    subprocess.run(["./create_venv.sh"], cwd=runtime, check=True)
    build = runtime / "build_Release"
    build.mkdir()
    (build / "CMakeCache.txt").write_text("ENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK:BOOL=ON\n")
    factory = "ttnn/cpp/ttnn/operations/sample/device/sample_program_factory.cpp"
    (build / "compile_commands.json").write_text(
        json.dumps(
            [
                {
                    "directory": str(build),
                    "file": str(runtime / factory),
                    "arguments": [
                        "python3",
                        str(runtime / "fake_contract_compiler.py"),
                        "-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK",
                        "-c",
                        str(runtime / factory),
                        "-o",
                        "unused.o",
                    ],
                }
            ]
        )
    )
    return {
        "runtime": str(runtime),
        "evaluated_branch": str(evaluated["output"]),
        "migration_paths": sorted(files),
        "workspace": str(runtime / branch.ARTIFACT_DIR / "validation"),
        "source_entry": "ttnn.operations.sample_op:sample_op",
        "native_entry": "ttnn:sample_native",
        "acceptance_tests": ["tests/test_cache.py"],
        "build_argv": ["./build_metal.sh"],
        "precompile": False,
        "factory_contract": {
            "operation_header": "ttnn/cpp/ttnn/operations/sample/device/sample.hpp",
            "operation_type": "sample::DeviceOperation",
            "factory_source": factory,
        },
    }


def test_branch_validation_never_reads_db_or_installs_export(branch_port, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("branch mode must not reconstruct DB source")

    monkeypatch.setattr(validate_port, "verify_export", forbidden)
    monkeypatch.setattr(validate_port, "verify_preparation", forbidden)
    monkeypatch.setattr(validate_port.prepare_target, "inspect", forbidden)
    validate_port.initialize(branch_port)
    validation = validate_port.PortValidation(branch_port["workspace"])
    validation.run("native_compare")
    suites = list(validation.workspace.glob("attempts/*/001/junit.xml"))
    assert {path.parent.parent.name for path in suites} == {"source", "native", "acceptance"}
    comparison = json.loads((validation.workspace / "attempts/source_compare/001/comparison.json").read_bytes())
    assert comparison["recorded_results_compared"] is False
    assert comparison["observed_failures"] == 0
    assert validation.planned["historical_failures"] is None
    record_review(validation)
    validation.run("complete")
    result = json.loads((validation.workspace / "attempts/complete/001/result.json").read_bytes())
    assert result["input_mode"] == "evaluated_branch"
    assert result["historical_failures"] is None
    assert result["source_failures"] == 0
    before = (validation.workspace / "state.json").read_bytes()
    validation.run()
    assert (validation.workspace / "state.json").read_bytes() == before


def test_branch_validation_rejects_mixed_historical_config(branch_port):
    branch_port["export"] = "/unused-export"
    with pytest.raises(ExportError, match="config keys"):  # allow-pytest.raises: mutually exclusive source modes
        validate_port.plan(branch_port)


def test_branch_acceptance_tests_are_protected_on_resume(branch_port):
    validate_port.initialize(branch_port)
    (Path(branch_port["runtime"]) / "tests/test_cache.py").write_text("changed test")
    with pytest.raises(ExportError, match="acceptance tests changed"):  # allow-pytest.raises: fixed test contract
        validate_port.PortValidation(branch_port["workspace"]).run()


def test_branch_factory_is_improved_in_place(branch_port):
    validate_port.initialize(branch_port)
    validation = validate_port.PortValidation(branch_port["workspace"])
    validation.run()
    factory = Path(branch_port["runtime"]) / branch_port["factory_contract"]["factory_source"]
    factory.write_text("// corrected descriptor factory\n")
    validation.run()
    assert len(validation.state["stages"]["acceptance"]["attempts"]) == 2
    assert validation.state["stages"]["source"]["status"] == "pending"
    source = Path(branch_port["runtime"]) / "ttnn/ttnn/operations/sample_op/planner.py"
    source.write_text("# not a factory fix\n")
    with pytest.raises(ExportError, match="Undeclared change"):  # allow-pytest.raises: original remains reference
        validation.run()


@pytest.mark.parametrize("configured", ["failed"], indirect=True)
def test_source_failures_remain_visible_and_native_comparison_is_mandatory(branch_port):
    validate_port.initialize(branch_port)
    validation = validate_port.PortValidation(branch_port["workspace"])
    validation.run("native_compare")
    comparison = json.loads((validation.workspace / "attempts/native_compare/001/comparison.json").read_bytes())
    assert comparison["outcomes_match"]
    assert comparison["observed_failures"] == 1
    assert validation.state["stages"]["native"]["status"] == "complete"
    evidence = json.loads((validation.workspace / "attempts/source_compare/001/comparison.json").read_bytes())
    assert evidence["observed_failures"] == 1


@pytest.mark.parametrize("configured", ["failed"], indirect=True)
@pytest.mark.parametrize("change", ["status", "missing", "added"])
def test_failing_source_never_disables_native_outcome_or_case_checks(branch_port, monkeypatch, change):
    original = validate_port.parse_junit_xml

    def changed_native(path):
        rows = original(path)
        if "native" in Path(path).parts:
            if change == "status":
                rows[0]["status"] = "passed"
            elif change == "missing":
                rows[0]["test_name"] = "different_case"
            else:
                rows.append({**rows[0], "test_name": "additional_case"})
        return rows

    monkeypatch.setattr(validate_port, "parse_junit_xml", changed_native)
    validate_port.initialize(branch_port)
    validation = validate_port.PortValidation(branch_port["workspace"])
    with pytest.raises(ExportError, match="Case outcomes differ"):  # allow-pytest.raises: mandatory parity gate
        validation.run("native_compare")
    assert validation.state["stages"]["native_compare"]["status"] == "blocked"
    assert validation.state["stages"]["acceptance"]["status"] == "complete"


def test_obsolete_source_failure_switch_is_not_silently_accepted(branch_port):
    branch_port["allow_source_failures"] = False
    with pytest.raises(ExportError, match="config keys"):  # allow-pytest.raises: no source-green policy
        validate_port.plan(branch_port)


def test_preparation_refuses_parent_directory_outputs(evaluated):
    evaluated["output"] = evaluated["runtime"].parent / "loose-evidence"
    with pytest.raises(ExportError, match="artifacts must be inside"):  # allow-pytest.raises: worktree ownership
        prepare(evaluated)
    assert not evaluated["output"].exists()
    assert not evaluated["runtime"].exists()


@pytest.mark.parametrize("kind", ["parent", "source", "input_child", "input_parent", "root"])
def test_validation_requires_separate_worktree_artifact_directory(branch_port, kind):
    runtime = Path(branch_port["runtime"])
    paths = {
        "parent": runtime.parent / "loose-evidence",
        "source": runtime / "ttnn" / "evidence",
        "input_child": Path(branch_port["evaluated_branch"]) / "evidence",
        "input_parent": runtime / branch.ARTIFACT_DIR / "nested",
        "root": runtime / branch.ARTIFACT_DIR,
    }
    branch_port["workspace"] = str(paths[kind])
    if kind == "input_parent":
        branch_port["evaluated_branch"] = str(paths[kind] / "inputs")
    with pytest.raises(ExportError, match="artifacts must be inside|separate artifact"):  # allow-pytest.raises: paths
        validate_port.plan(branch_port)
    assert not paths[kind].exists() or kind == "root"


def test_generated_artifacts_do_not_contaminate_source_snapshot(branch_port):
    initial = validate_port.initialize(branch_port)
    validation = validate_port.PortValidation(branch_port["workspace"])
    cache = validation.workspace / "device-cache" / "kernel"
    cache.mkdir(parents=True)
    (cache / "binary.elf").write_bytes(b"generated")
    (validation.workspace / "diagnostic.log").write_text("generated evidence")
    validation.validate()
    assert validate_port.digest(validate_port.plan(branch_port)) == initial["plan_sha256"]
    assert not any(branch.ARTIFACT_DIR in path for path in validation.planned["untracked_files"])
    assert git(Path(branch_port["runtime"]), "status", "--porcelain") == ""


@pytest.mark.parametrize("mutation", ["ignore", "tracked", "allowlist", "symlink"])
def test_artifact_namespace_cannot_hide_source_or_be_redirected(branch_port, tmp_path, mutation):
    runtime = Path(branch_port["runtime"])
    root = runtime / branch.ARTIFACT_DIR
    if mutation == "ignore":
        (root / ".gitignore").write_text("different ignore policy\n")
    elif mutation == "tracked":
        (root / "source.cpp").write_text("// wrongly tracked source")
        git(runtime, "add", "-f", str(root / "source.cpp"))
    elif mutation == "allowlist":
        branch_port["migration_paths"].append(branch.ARTIFACT_DIR + "/source.cpp")
    else:
        (root / "redirected").symlink_to(tmp_path, target_is_directory=True)
        branch_port["workspace"] = str(root / "redirected" / "validation")
    with pytest.raises(  # allow-pytest.raises: artifact/source boundary
        ExportError, match="ignore marker|tracked source|cannot replace|unredirected"
    ):
        validate_port.plan(branch_port)


def test_evaluated_tree_cannot_occupy_reserved_artifact_namespace(evaluated):
    revision = commit(evaluated["repository"], {branch.ARTIFACT_DIR + "/source.cpp": b"source"})
    git(evaluated["repository"], "update-ref", "refs/heads/evaluated-candidate", revision)
    with pytest.raises(ExportError, match="reserved"):  # allow-pytest.raises: checkpoint namespace collision
        prepare(evaluated)
    assert not evaluated["runtime"].exists()
