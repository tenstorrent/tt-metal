# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generic source-preparation tests; no production runs, network or device."""

import hashlib
import json
import subprocess

import pytest

from tools.generic_op_to_factory import prepare_baseline as preparation
from tools.generic_op_to_factory.export_run import ExportError, export_snapshot, json_bytes
from tools.generic_op_to_factory.tests.test_export_run import ReadConnection, snapshot  # noqa: F401


def git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args], stderr=subprocess.DEVNULL).decode().strip()


def commit(repository, files):
    repository.mkdir(exist_ok=True)
    if not (repository / ".git").exists():
        git(repository, "init")
    for name, content in files.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    git(repository, "add", ".")
    git(
        repository,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "fixture",
    )
    return git(repository, "rev-parse", "HEAD")


@pytest.fixture
def inputs(tmp_path, snapshot):
    connection, run_id = snapshot
    evaluator, metal = tmp_path / "evaluator", tmp_path / "metal"
    eval_revision = commit(
        evaluator,
        {
            "eval/__init__.py": b"",
            "eval/golden_tests/__init__.py": b"",
            "eval/golden_tests/conftest.py": b"import pytest\n",
            "eval/golden_tests/sample_suite/__init__.py": b"",
            "eval/golden_tests/sample_suite/test_golden.py": b"from .helpers import check\n",
            "eval/golden_tests/sample_suite/helpers.py": b"from eval.metrics import check\nimport torch\n",
            "eval/golden_tests/sample_suite/data.bin": b"\x00\xff",
            "eval/metrics.py": b"def check(): pass\n",
            "eval/hang_plugin.py": b"",
            "eval/metrics_plugin.py": b"",
            "eval/axes_plugin.py": b"",
            "eval/run_eval.py": b"# orchestration evidence\n",
            "eval/eval_test_runner.sh": b"# runner evidence\n",
            "eval/unrelated.py": b"raise Exception('must not import')\n",
        },
    )
    metal_revision = commit(
        metal,
        {
            "ttnn/cpp/ttnn/kernel_lib/helper.hpp": b"// canonical\r\n",
            "ttnn/ttnn/operations/_op_contract.py": b"class SupportRefusal(Exception): pass\n",
            "conftest.py": b"# runtime fixture\n",
            "pytest.ini": b"[pytest]\n",
            "scripts/run_safe_pytest.sh": b"# safe runner\n",
            ".gitmodules": b"",
        },
    )
    connection.execute("PRAGMA query_only = OFF")
    connection.execute(
        "UPDATE runs SET starting_commit=?, eval_commit=? WHERE id=?",
        (metal_revision, eval_revision, run_id),
    )
    connection.execute("PRAGMA query_only = ON")
    exported = tmp_path / "exported"
    export_snapshot(ReadConnection(connection), run_id, exported, database={"host": "fixture"})
    return exported, evaluator, metal


def test_pinned_bytes_and_determinism(inputs, tmp_path):
    exported, evaluator, metal = inputs
    # Dirty/current checkout files must not influence recorded revision exports.
    (evaluator / "eval/metrics.py").write_bytes(b"raise Exception('dirty')\n")
    first, second = tmp_path / "first", tmp_path / "second"
    result = preparation.prepare(*inputs, first)
    assert result == preparation.prepare(*inputs, second)
    assert not result["baseline_reproduced"] and not result["migration_ready"]
    assert "torch" in result["external_python_imports"]
    assert (first / "overlay/eval/metrics.py").read_bytes() == b"def check(): pass\n"
    assert not (first / "overlay/eval/unrelated.py").exists()
    assert (first / "overlay/eval/golden_tests/sample_suite/data.bin").read_bytes() == b"\x00\xff"
    assert (first / "overlay/ttnn/ttnn/operations/sample_op/planner.py").read_bytes() == (
        exported / "source/planner.py"
    ).read_bytes()
    for entry in result["files"]:
        raw = (first / entry["path"]).read_bytes()
        assert raw == (second / entry["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert (
        result["preparation_sha256"]
        == hashlib.sha256(
            json_bytes({key: value for key, value in result.items() if key != "preparation_sha256"})
        ).hexdigest()
    )


def test_does_not_overwrite(inputs, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ExportError, match="already exists"):  # allow-pytest.raises: host-only workflow validation
        preparation.prepare(*inputs, output)
    assert list(output.iterdir()) == []


def test_rejects_corrupt_export(inputs, tmp_path):
    (inputs[0] / "source/planner.py").write_bytes(b"modified")
    with pytest.raises(ExportError, match="checksum"):  # allow-pytest.raises: host-only workflow validation
        preparation.prepare(*inputs, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("revision", [None, "HEAD", "main", "../bad", "f" * 40])
def test_requires_available_full_commit(inputs, revision):
    with pytest.raises(ExportError):  # allow-pytest.raises: host-only workflow validation
        preparation.GitTree(inputs[1], revision)


def test_missing_import_fails(inputs):
    evaluator = inputs[1]
    revision = commit(
        evaluator,
        {"eval/golden_tests/sample_suite/helpers.py": b"import eval.missing\n"},
    )
    with pytest.raises(  # allow-pytest.raises: host-only workflow validation
        ExportError, match="Unresolved local eval import"
    ):
        preparation.eval_closure(preparation.GitTree(evaluator, revision), "sample_suite")


def test_dynamic_import_is_flagged(inputs):
    evaluator = inputs[1]
    revision = commit(
        evaluator,
        {"eval/metrics.py": b"def check():\n    return __import__('unknown')\n"},
    )
    _, _, dynamic = preparation.eval_closure(preparation.GitTree(evaluator, revision), "sample_suite")
    assert dynamic == [{"path": "eval/metrics.py", "line": 2, "call": "__import__"}]


def test_symlink_is_not_followed(inputs):
    evaluator = inputs[1]
    link = evaluator / "eval/golden_tests/sample_suite/link"
    link.symlink_to("/etc/passwd")
    revision = commit(evaluator, {"marker": b"new"})
    with pytest.raises(ExportError, match="regular file"):  # allow-pytest.raises: host-only workflow validation
        preparation.eval_closure(preparation.GitTree(evaluator, revision), "sample_suite")


def test_preserves_recorded_status(inputs, tmp_path):
    # The existing export tests cover terminal failure snapshots. Preparation
    # must copy their recorded status, not turn export success into eval success.
    manifest = preparation.prepare(*inputs, tmp_path / "output")
    run = json.loads((inputs[0] / "records/run.json").read_bytes())
    assert manifest["historical_status"]["status"] == run["status"]


@pytest.mark.parametrize("change", ["content", "missing", "extra", "symlink", "manifest"])
def test_offline_verification_rejects_drift(inputs, tmp_path, change):
    output = tmp_path / "output"
    preparation.prepare(*inputs, output)
    preparation.verify_preparation(output)
    path = output / "overlay/eval/metrics.py"
    if change == "content":
        path.write_bytes(b"drift")
    elif change == "missing":
        path.unlink()
    elif change == "extra":
        (output / "extra").write_bytes(b"extra")
    elif change == "symlink":
        (output / "extra").symlink_to(path)
    else:
        manifest = json.loads((output / "baseline.json").read_bytes())
        manifest["migration_ready"] = True
        (output / "baseline.json").write_bytes(json_bytes(manifest))
    with pytest.raises(ExportError):  # allow-pytest.raises: host-only workflow validation
        preparation.verify_preparation(output)


def test_install_only_new_operation(inputs, tmp_path):
    _, evaluator, runtime = inputs
    (runtime / "eval").symlink_to(evaluator / "eval", target_is_directory=True)
    output = tmp_path / "output"
    preparation.prepare(*inputs, output)
    result = preparation.install(output, runtime)
    assert not result["baseline_reproduced"]
    assert (runtime / "ttnn/ttnn/operations/sample_op/planner.py").read_bytes() == (
        inputs[0] / "source/planner.py"
    ).read_bytes()
    with pytest.raises(ExportError, match="already exists"):  # allow-pytest.raises: host-only workflow validation
        preparation.install(output, runtime)


def test_install_rejects_dirty_dependency(inputs, tmp_path):
    _, evaluator, runtime = inputs
    (runtime / "eval").symlink_to(evaluator / "eval", target_is_directory=True)
    output = tmp_path / "output"
    preparation.prepare(*inputs, output)
    (evaluator / "eval/metrics.py").write_bytes(b"modified")
    with pytest.raises(ExportError, match="dependency differs"):  # allow-pytest.raises: host-only workflow validation
        preparation.install(output, runtime)
    assert not (runtime / "ttnn/ttnn/operations/sample_op").exists()


def test_install_rejects_wrong_runtime_revision(inputs, tmp_path):
    output = tmp_path / "output"
    preparation.prepare(*inputs, output)
    commit(inputs[2], {"new": b"new revision"})
    with pytest.raises(ExportError, match="HEAD differs"):  # allow-pytest.raises: host-only workflow validation
        preparation.install(output, inputs[2])
