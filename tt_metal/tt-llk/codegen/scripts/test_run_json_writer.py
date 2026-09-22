# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for run_json_writer.py — dashboard-compatibility schema."""

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "run_json_writer.py"
STATE = Path(__file__).parent / "state.py"
SETUP_WORKTREE = Path(__file__).parent / "setup_worktree.sh"
ORCHESTRATOR_STEPS = Path(__file__).parent / "issue_solver" / "orchestrator_steps.sh"
RUN_UTILS = Path(__file__).parent / "issue_solver_run_utils.py"
RUN_TEST = Path(__file__).parents[2] / ".claude" / "scripts" / "run_test.sh"
LLK_CONFTEST = Path(__file__).parents[2] / "tests" / "python_tests" / "conftest.py"


def _run(log_dir, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--log-dir", str(log_dir)],
        check=True,
        capture_output=True,
        text=True,
    )


def _required_manifest(
    tmp_path,
    analysis,
    plan,
    *extra,
    check=True,
    git_changes: dict[str, str] | None = None,
):
    worktree = tmp_path / "worktree"
    llk_tests = worktree / "tt_metal" / "tt-llk" / "tests" / "python_tests"
    llk_tests.mkdir(parents=True, exist_ok=True)
    (llk_tests / "test_reduce.py").write_text("def test_reduce(): pass\n")
    (llk_tests / "perf_reduce.py").write_text("def test_reduce(): pass\n")
    metal = worktree / "tests" / "tt_metal" / "tt_metal" / "llk"
    metal.mkdir(parents=True, exist_ok=True)
    (metal / "test_reduce.cpp").write_text("// test\n")
    ttnn = worktree / "tests" / "ttnn" / "unit_tests" / "operations"
    ttnn.mkdir(parents=True, exist_ok=True)
    (ttnn / "test_reduce.py").write_text("def test_reduce(): pass\n")
    expected_base = "a" * 40
    if git_changes is not None:
        subprocess.run(["git", "init", "-q", str(worktree)], check=True)
        subprocess.run(
            ["git", "-C", str(worktree), "config", "user.name", "test"], check=True
        )
        subprocess.run(
            ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(["git", "-C", str(worktree), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(worktree), "commit", "-qm", "base"], check=True
        )
        expected_base = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        for relative, contents in git_changes.items():
            changed = worktree / relative
            changed.parent.mkdir(parents=True, exist_ok=True)
            changed.write_text(contents)
    analysis_path = tmp_path / "analysis.md"
    plan_path = tmp_path / "plan.md"
    analysis_path.write_text(analysis)
    plan_path.write_text(plan)
    output = tmp_path / "required_verification_manifest.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "required-verification",
            "--log-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--analysis",
            str(analysis_path),
            "--plan",
            str(plan_path),
            "--worktree",
            str(worktree),
            "--run-id",
            "run-1",
            "--expected-base-sha",
            expected_base,
            "--architectures-json",
            '["blackhole"]',
            "--backend",
            "local",
            *extra,
        ],
        check=check,
        capture_output=True,
        text=True,
    )
    return proc, output


def test_required_verification_seals_independent_suites_and_perf_measurement(
    tmp_path,
):
    analysis = """\
## Scope
arch_scope:
  blackhole: in_scope
## Verification
fix_layer: mixed
verification_required: yes
verifiable_in_llk_suite: partial
llk_coverage: existing
metal_verification:
  target: unit_tests_llk
  coverage: added
  test_file: tests/tt_metal/tt_metal/llk/test_reduce.cpp
  gtest_filter: 'LLKFixture.Reduce'
  dispatch: fast
"""
    plan = """\
## Test Strategy
reproduction_tests:
- arch: blackhole
  test: tests/python_tests/test_reduce.py::test_reduce
regression_tests:
- arch: blackhole
  test: perf_reduce.py
  coverage: existing
  reason: determinism needs N>=3 independent reloads
"""
    proc, output = _required_manifest(tmp_path, analysis, plan)
    manifest = json.loads(output.read_text())
    assert proc.returncode == 0
    assert manifest["revision"] == 1
    assert manifest["attempt_id"] == "attempt-001"
    assert manifest["waivers"] == []
    assert [item["requirement_id"] for item in manifest["requirements"]] == [
        "blackhole:llk:1",
        "blackhole:metal:1",
        "blackhole:perf:1",
    ]
    llk, metal, perf = manifest["requirements"]
    assert llk["selector"] == {
        "test": "test_reduce.py",
        "test_id": "test_reduce.py::test_reduce",
        "k": None,
    }
    assert metal["selector"]["test"] == "LLKFixture.Reduce"
    assert {llk["backend"], metal["backend"], perf["backend"]} == {"silicon"}
    assert perf["minimum_executed"] == 3
    assert perf["required_measurements"] == ["cycle_comparison", "repeatability"]
    expected = hashlib.sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "manifest_id"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert manifest["manifest_id"] == expected


@pytest.mark.parametrize("k_filter", [None, "bf4b and HiFi3"])
def test_required_verification_seals_ttnn_as_an_independent_suite(tmp_path, k_filter):
    analysis = """\
## Scope
arch_scope:
  blackhole: in_scope
## Verification
fix_layer: ttnn
verification_required: yes
verifiable_in_llk_suite: partial
llk_coverage: existing
metal_verification:
  target: none
  coverage: not_applicable
  test_file: none
  gtest_filter: none
  dispatch: none
ttnn_verification:
  target: ttnn
  coverage: added
  test: tests/ttnn/unit_tests/operations/test_reduce.py::test_reduce
  dispatch: fast
"""
    plan = """\
## Test Strategy
reproduction_tests:
- arch: blackhole
  test: tests/python_tests/test_reduce.py::test_reduce
"""
    if k_filter:
        analysis = analysis.replace(
            "::test_reduce\n", f"::test_reduce -k '{k_filter}'\n"
        )
    proc, output = _required_manifest(tmp_path, analysis, plan)
    requirements = json.loads(output.read_text())["requirements"]
    assert proc.returncode == 0
    assert [item["suite"] for item in requirements] == ["llk", "ttnn"]
    assert requirements[1]["requirement_id"] == "blackhole:ttnn:1"
    assert requirements[1]["selector"] == {
        "test": "tests/ttnn/unit_tests/operations/test_reduce.py",
        "test_id": "tests/ttnn/unit_tests/operations/test_reduce.py::test_reduce",
        "k": k_filter,
    }
    assert "route=llk+ttnn" in proc.stdout

    # The same selector must survive reading, result reduction, and resealing.
    manifest = json.loads(output.read_text())
    results = tmp_path / "verification-results"
    results.mkdir()
    for requirement in requirements:
        result = _sealed_result(manifest, requirement)
        (results / f"{requirement['requirement_id']}.json").write_text(
            json.dumps(result)
        )
    reduced = _reduce(tmp_path, output)
    assert reduced.returncode == 0, reduced.stderr
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "success"
    _required_manifest(tmp_path, analysis, plan, "--supersedes-reason", "retry")
    revised = json.loads(output.read_text())
    assert revised["revision"] == 2
    assert revised["parent_manifest_id"] == manifest["manifest_id"]
    assert revised["requirements"] == requirements


def test_required_verification_rejects_missing_ttnn_coverage(tmp_path):
    analysis = """\
## Scope
arch_scope:
  blackhole: in_scope
## Verification
fix_layer: ttnn
verification_required: yes
verifiable_in_llk_suite: no
llk_coverage: not_applicable
ttnn_verification:
  target: ttnn
  coverage: add_required
  test: tests/ttnn/unit_tests/operations/test_reduce.py::test_reduce
  dispatch: fast
"""
    proc, output = _required_manifest(
        tmp_path, analysis, "## Test Strategy\ncompile_checks:\n- none\n", check=False
    )
    assert proc.returncode != 0
    assert "TTNN verification target/coverage is not executable" in proc.stderr
    assert not output.exists()


def test_required_verification_rejects_ttnn_layer_without_ttnn_suite(tmp_path):
    analysis = """\
## Scope
arch_scope:
  blackhole: in_scope
## Verification
fix_layer: ttnn
verification_required: yes
verifiable_in_llk_suite: no
llk_coverage: not_applicable
metal_verification:
  target: unit_tests_llk
  coverage: existing
  test_file: tests/tt_metal/tt_metal/llk/test_reduce.cpp
  gtest_filter: LLKFixture.Reduce
  dispatch: fast
ttnn_verification:
  target: none
  coverage: not_applicable
  test: none
  dispatch: none
"""
    proc, output = _required_manifest(
        tmp_path, analysis, "## Test Strategy\ncompile_checks:\n- none\n", check=False
    )
    assert proc.returncode != 0
    assert "TTNN-layer fix requires executable TTNN verification" in proc.stderr
    assert not output.exists()


def test_required_verification_rejects_mixed_diff_that_omits_ttnn_suite(tmp_path):
    analysis = """\
## Scope
arch_scope:
  blackhole: in_scope
## Verification
fix_layer: mixed
verification_required: yes
verifiable_in_llk_suite: no
llk_coverage: not_applicable
metal_verification:
  target: unit_tests_llk
  coverage: existing
  test_file: tests/tt_metal/tt_metal/llk/test_reduce.cpp
  gtest_filter: LLKFixture.Reduce
  dispatch: fast
ttnn_verification:
  target: none
  coverage: not_applicable
  test: none
  dispatch: none
"""
    proc, output = _required_manifest(
        tmp_path,
        analysis,
        "## Test Strategy\ncompile_checks:\n- none\n",
        check=False,
        git_changes={"ttnn/cpp/ttnn/new_operation.cpp": "// candidate\n"},
    )
    assert proc.returncode != 0
    assert (
        "TTNN source/test changes require executable TTNN verification" in proc.stderr
    )
    assert not output.exists()


def test_required_verification_revisions_are_immutable_and_linked(tmp_path):
    analysis = """\
## Scope
arch_scope:
  blackhole: in_scope
## Verification
verification_required: yes
verifiable_in_llk_suite: yes
llk_coverage: existing
"""
    plan = """\
## Test Strategy
reproduction_tests:
- arch: all
  test: test_reduce.py -k reduce
"""
    _, output = _required_manifest(tmp_path, analysis, plan)
    revision_one = (
        tmp_path / "required_verification_manifests" / "revision-001.json"
    ).read_bytes()
    failed, _ = _required_manifest(tmp_path, analysis, plan, check=False)
    assert failed.returncode != 0
    assert "superseding manifest requires --supersedes-reason" in failed.stderr
    _required_manifest(
        tmp_path,
        analysis,
        plan,
        "--supersedes-reason",
        "functional retry after candidate failure",
    )
    current = json.loads(output.read_text())
    first = json.loads(revision_one)
    assert current["revision"] == 2
    assert current["parent_manifest_id"] == first["manifest_id"]
    assert current["supersedes_reason"] == "functional retry after candidate failure"
    assert (
        tmp_path / "required_verification_manifests" / "revision-001.json"
    ).read_bytes() == revision_one


def test_required_verification_preserves_old_schema_llk_without_metal(tmp_path):
    analysis = """\
## Scope
in_scope: true
## Verification
verifiable_in_llk_suite: yes
## Test Candidates
- test: tests/python_tests/test_reduce.py::test_reduce
  arch: blackhole
"""
    plan = "## Test Strategy\ncompile_checks:\n- none\n"
    _, output = _required_manifest(tmp_path, analysis, plan)
    manifest = json.loads(output.read_text())
    assert [(r["suite"], r["selector"]["test"]) for r in manifest["requirements"]] == [
        ("llk", "test_reduce.py")
    ]


def test_required_verification_keeps_bh_fixture_off_wh(tmp_path):
    analysis = """\
## Scope
arch_scope:
  blackhole: in_scope
  wormhole: in_scope
## Verification
verification_required: yes
verifiable_in_llk_suite: partial
llk_coverage: existing
metal_verification:
  architectures: ["blackhole"]
  target: unit_tests_llk
  coverage: existing
  test_file: tests/tt_metal/tt_metal/llk/test_reduce.cpp
  gtest_filter: 'BlackholeFixture.Reduce'
  dispatch: fast
"""
    plan = (
        "## Test Strategy\nreproduction_tests:\n- arch: all\n  test: test_reduce.py\n"
    )
    _, output = _required_manifest(
        tmp_path, analysis, plan, "--architectures-json", '["blackhole", "wormhole"]'
    )
    requirements = json.loads(output.read_text())["requirements"]
    assert {(r["architecture"], r["suite"]) for r in requirements} == {
        ("blackhole", "llk"),
        ("wormhole", "llk"),
        ("blackhole", "metal"),
    }

    missing_coverage = analysis.replace(
        "verifiable_in_llk_suite: partial", "verifiable_in_llk_suite: no"
    )
    proc, _ = _required_manifest(
        tmp_path / "missing",
        missing_coverage,
        "## Test Strategy\n",
        "--architectures-json",
        '["blackhole", "wormhole"]',
        check=False,
    )
    assert proc.returncode != 0
    assert "no executable requirement for: wormhole" in proc.stderr


@pytest.mark.parametrize("suite", ["metal", "ttnn"])
@pytest.mark.parametrize(
    "architectures,reason",
    [
        ("blackhole", "must be a JSON list"),
        ("<JSON list of architectures>", "must be a JSON list"),
        ("42", "must be a nonempty subset"),
        ("[]", "must be a nonempty subset"),
        ('["blackhole", "blackhole"]', "must be a nonempty subset"),
        ('["quasar"]', "must be a nonempty subset"),
    ],
)
def test_required_verification_names_invalid_suite_architectures(
    tmp_path, suite, architectures, reason
):
    target = "unit_tests_llk" if suite == "metal" else "ttnn"
    analysis = f"""\
## Scope
in_scope: true
## Verification
verification_required: yes
verifiable_in_llk_suite: no
{suite}_verification:
  architectures: {architectures}
  target: {target}
  coverage: existing
  test_file: tests/tt_metal/tt_metal/llk/test_reduce.cpp
  gtest_filter: 'Reduce.Test'
  test: tests/ttnn/unit_tests/operations/test_reduce.py
  dispatch: fast
"""
    proc, _ = _required_manifest(tmp_path, analysis, "## Test Strategy\n", check=False)
    assert proc.returncode != 0
    assert f"{suite}_verification architectures {reason}" in proc.stderr


def test_required_verification_infers_llk_from_old_plan_without_verification_section(
    tmp_path,
):
    analysis = "## Scope\nin_scope: true\n"
    plan = """\
## Test Strategy
reproduction_tests:
- arch: blackhole
  test: test_reduce.py::test_reduce
"""
    _, output = _required_manifest(tmp_path, analysis, plan)
    requirement = json.loads(output.read_text())["requirements"][0]
    assert requirement["requirement_id"] == "blackhole:llk:1"
    assert requirement["selector"]["test_id"] == "test_reduce.py::test_reduce"


@pytest.mark.parametrize("arches", [["blackhole"], ["blackhole", "quasar"]])
def test_required_verification_retains_perf_when_hypothesis_is_refuted(
    tmp_path, arches
):
    analysis = """\
## Scope
in_scope: true
## Verification
verification_required: yes
verifiable_in_llk_suite: yes
llk_coverage: add_required
"""
    plan = """\
## Primary Hypothesis
status: refuted
## Test Strategy
regression_tests:
- arch: blackhole
  test: perf_reduce.py
  coverage: existing
"""
    _, output = _required_manifest(
        tmp_path,
        analysis,
        plan,
        "--performance-only",
        "--architectures-json",
        json.dumps(arches),
    )
    requirements = json.loads(output.read_text())["requirements"]
    assert len(requirements) == 1
    assert requirements[0]["suite"] == "perf"
    assert requirements[0]["required_measurements"] == ["cycle_comparison"]


@pytest.mark.parametrize(
    ("analysis", "plan", "reason"),
    [
        (
            "## Scope\nin_scope: true\n## Verification\nverification_required: yes\n"
            "verifiable_in_llk_suite: yes\nllk_coverage: add_required\n",
            "## Test Strategy\nreproduction_tests:\n- arch: blackhole\n"
            "  test: test_reduce.py\n",
            "coverage must be existing|added",
        ),
        (
            "## Scope\nin_scope: true\n## Verification\nverification_required: yes\n"
            "verifiable_in_llk_suite: yes\nllk_coverage: existing\n",
            "## Test Strategy\nreproduction_tests:\n- arch: blackhole\n"
            "  test: test_missing.py\n",
            "names a missing file",
        ),
    ],
)
def test_required_verification_rejects_unexecutable_coverage(
    tmp_path, analysis, plan, reason
):
    proc, output = _required_manifest(tmp_path, analysis, plan, check=False)
    assert proc.returncode != 0
    assert reason in proc.stderr
    assert not output.exists()


def _write_verification_inputs(
    tmp_path, *, selected=1, junit_tests=1, log="", collection_returncode=None
):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "kernel.elf").write_bytes(b"elf-v1")
    manifest = tmp_path / "artifact-manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "artifact-manifest",
            "--output",
            str(manifest),
            "--artifact-root",
            str(artifact_root),
            "--owner-id",
            "owner-1",
            "--build-input-digest",
            "1" * 64,
            "--source-tree-sha256",
            "2" * 64,
            "--compiler-sha256",
            "3" * 64,
        ],
        check=True,
    )
    collection = tmp_path / "collection.json"
    collection.write_text(
        json.dumps(
            {
                "schema": "tt.issue-solver.pytest-collection",
                "version": 1,
                "selected": selected,
                "collected": selected,
                "errors": 0,
                "returncode": (
                    collection_returncode
                    if collection_returncode is not None
                    else (0 if selected else 5)
                ),
            }
        )
    )
    junit = tmp_path / "consumer.junit.xml"
    cases = "".join(f'<testcase name="t{i}"/>' for i in range(junit_tests))
    junit.write_text(
        f'<testsuites><testsuite tests="{junit_tests}" failures="0" errors="0" '
        f'skipped="0">{cases}</testsuite></testsuites>'
    )
    output_log = tmp_path / "consumer.log"
    output_log.write_text(log)
    return artifact_root, manifest, collection, junit, output_log


def _write_verification_result(tmp_path, inputs, *extra):
    artifact_root, manifest, collection, junit, output_log = inputs
    output = tmp_path / "verification-result.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verification-result",
            "--output",
            str(output),
            "--collection-json",
            str(collection),
            "--junit",
            str(junit),
            "--output-log",
            str(output_log),
            "--artifact-manifest",
            str(manifest),
            "--artifact-root",
            str(artifact_root),
            "--requirement-id",
            "blackhole:llk:1",
            "--run-id",
            "run-1",
            "--attempt-id",
            "attempt-1",
            "--job-id",
            "local-1",
            "--architecture",
            "blackhole",
            "--suite",
            "llk",
            "--backend",
            "local",
            "--test",
            "test.py",
            "--expected-base-sha",
            "4" * 40,
            "--actual-base-sha",
            "4" * 40,
            "--patch-sha256",
            "5" * 64,
            "--returncode",
            "0",
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc, output


def test_verification_result_is_strict_content_addressed_success(tmp_path):
    proc, output = _write_verification_result(
        tmp_path, _write_verification_inputs(tmp_path)
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(output.read_text())
    assert result["classification"] == "success"
    assert result["collection"] == {
        "selected": 1,
        "collected": 1,
        "errors": 0,
        "returncode": 0,
    }
    assert result["execution"]["passed"] == 1
    expected_id = hashlib.sha256(
        json.dumps(
            {key: value for key, value in result.items() if key != "result_id"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert result["result_id"] == expected_id


def test_verification_result_rejects_zero_coverage_even_with_exit_zero(tmp_path):
    inputs = _write_verification_inputs(
        tmp_path, selected=0, junit_tests=0, collection_returncode=0
    )
    proc, output = _write_verification_result(tmp_path, inputs)
    assert proc.returncode == 1
    result = json.loads(output.read_text())
    assert result["classification"] == "coverage_error"
    assert result["reason_codes"] == ["zero_selected"]
    assert result["execution"]["ran"] is False


def test_cardless_collection_guard_precedes_device_initialization():
    source = LLK_CONFTEST.read_text(encoding="utf-8")
    start = source.index("def pytest_configure(config):")
    end = source.index("def pytest_ignore_collect", start)
    configure = source[start:end]
    guard = configure.index("if config.option.collectonly:\n        return")
    for operation in (
        "override_gprs_used_by_tensix_dump()",
        "tt_exalens_init.init_ttexalens(",
        "ExalensServer(",
    ):
        assert configure.index(operation) > guard


@pytest.mark.parametrize("returncode", [2, 4])
def test_verification_result_rejects_nonzero_collection(tmp_path, returncode):
    inputs = _write_verification_inputs(tmp_path, collection_returncode=returncode)
    proc, output = _write_verification_result(tmp_path, inputs)
    assert proc.returncode == 3
    result = json.loads(output.read_text())
    assert result["classification"] == "infra_error"
    assert result["reason_codes"] == ["collection_nonzero_exit"]


def test_verification_result_fatal_marker_and_artifact_mutation_are_infra(tmp_path):
    inputs = _write_verification_inputs(tmp_path, log="TT_FATAL during device init")
    (inputs[0] / "kernel.elf").write_bytes(b"mutated")
    proc, output = _write_verification_result(tmp_path, inputs)
    assert proc.returncode == 3
    result = json.loads(output.read_text())
    assert result["classification"] == "infra_error"
    assert result["execution"]["infrastructure_markers"] == [
        "tt_fatal",
        "artifact_mutated_during_execution",
    ]
    assert (
        result["provenance"]["executed_artifact_sha256"]
        != result["provenance"]["artifact_set_sha256"]
    )


def _content_id(document, omitted):
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in document.items() if key not in omitted},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _reducer_manifest(log_dir, requirements):
    document = {
        "schema": "tt.issue-solver.required-verification",
        "version": 1,
        "manifest_id": "0" * 64,
        "run_id": "run-reducer",
        "attempt_id": "attempt-001",
        "expected_base_sha": "a" * 40,
        "revision": 1,
        "parent_manifest_id": None,
        "supersedes_reason": None,
        "requirements": requirements,
        "waivers": [],
    }
    document["manifest_id"] = _content_id(document, {"manifest_id"})
    path = log_dir / "required_verification_manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return document, path


def _requirement(arch="blackhole", suite="llk", index=1, **overrides):
    document = {
        "requirement_id": f"{arch}:{suite}:{index}",
        "architecture": arch,
        "suite": suite,
        "backend": "silicon",
        "selector": {
            "test": "test_reduce.py" if suite != "metal" else "LLK.Reduce",
            "test_id": None,
            "k": None,
        },
        "minimum_selected": 1,
        "minimum_executed": 1,
        "required_measurements": [],
    }
    document.update(overrides)
    return document


def _sealed_result(
    manifest,
    requirement,
    *,
    selected=1,
    executed=1,
    passed=1,
    failed=0,
    skipped=0,
    xfailed=0,
    collection_errors=0,
    collection_returncode=0,
    returncode=0,
    timed_out=False,
    markers=None,
    patch_sha256="b" * 64,
    artifact_sha256="c" * 64,
    executed_artifact_sha256=None,
    attempt_id=None,
    job_id="job-1",
):
    markers = markers or []
    execution = {
        "ran": executed > 0,
        "executed": executed,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "xfailed": xfailed,
        "xpassed": 0,
        "returncode": returncode,
        "signal": None,
        "timed_out": timed_out,
        "infrastructure_markers": markers,
    }
    collection = {
        "selected": selected,
        "collected": selected,
        "errors": collection_errors,
        "returncode": collection_returncode,
    }
    if timed_out:
        # Mirrors _classify_verification (xpassed is always 0 in this fixture).
        classification, reasons = "timed_out", [
            "execution_timed_out",
            "failures_observed" if failed else "no_failures_observed",
        ]
    elif collection_returncode or collection_errors or markers:
        classification = "infra_error"
        reasons = []
        if collection_returncode:
            reasons.append("collection_nonzero_exit")
        if collection_errors:
            reasons.append("collection_error")
        reasons.extend(markers)
    elif selected == 0:
        classification, reasons = "coverage_error", ["zero_selected"]
    elif executed == 0:
        classification, reasons = "coverage_error", ["zero_executed"]
    elif returncode == 0 and failed == 0 and passed == executed:
        classification, reasons = "success", []
    elif returncode == 1 and failed:
        classification, reasons = "candidate_failure", ["test_failure"]
    elif returncode == 0:
        classification, reasons = "candidate_failure", ["outcome_count_mismatch"]
    else:
        classification, reasons = "infra_error", ["execution_nonzero_exit"]
    result = {
        "schema": "tt.issue-solver.verification-result",
        "version": 2,
        "result_id": "0" * 64,
        "requirement_id": requirement["requirement_id"],
        "run_id": manifest["run_id"],
        "attempt_id": attempt_id or manifest["attempt_id"],
        "job_id": job_id,
        "architecture": requirement["architecture"],
        "suite": requirement["suite"],
        "backend": requirement["backend"],
        "selector": requirement["selector"],
        "provenance": {
            "expected_base_sha": manifest["expected_base_sha"],
            "actual_base_sha": manifest["expected_base_sha"],
            "patch_sha256": patch_sha256,
            "manifest_id": "d" * 64,
            "artifact_set_sha256": artifact_sha256,
            "executed_artifact_sha256": (executed_artifact_sha256 or artifact_sha256),
        },
        "collection": collection,
        "execution": execution,
        "classification": classification,
        "reason_codes": list(dict.fromkeys(reasons)),
    }
    result["result_id"] = _content_id(result, {"result_id"})
    return result


def _reduce(log_dir, manifest_path, scope="all", perf_result=None, worktree=None):
    args = [
        "reduce-verification",
        "--manifest",
        str(manifest_path),
        "--results-dir",
        str(log_dir / "verification-results"),
        "--scope",
        scope,
        "--output",
        str(log_dir / "verification_reduction.json"),
    ]
    if perf_result:
        args.extend(["--perf-result", str(perf_result)])
    if worktree:
        args.extend(["--worktree", str(worktree)])
    return _run(log_dir, *args)


def test_verification_reducer_derives_multi_arch_totals_and_success_token(tmp_path):
    requirements = [
        _requirement(),
        _requirement(
            "wormhole",
            "metal",
            selector={"test": "LLK.Reduce", "test_id": None, "k": None},
        ),
    ]
    manifest, manifest_path = _reducer_manifest(tmp_path, requirements)
    results = tmp_path / "verification-results"
    results.mkdir()
    first = _sealed_result(manifest, requirements[0], selected=2, executed=2, passed=2)
    second = _sealed_result(
        manifest,
        requirements[1],
        selected=3,
        executed=3,
        passed=3,
        job_id="job-2",
    )
    (results / "first.json").write_text(json.dumps(first), encoding="utf-8")
    (results / "second.json").write_text(json.dumps(second), encoding="utf-8")
    (tmp_path / "run.json").write_text(
        json.dumps({"run_id": manifest["run_id"]}), encoding="utf-8"
    )

    _reduce(tmp_path, manifest_path)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    run = json.loads((tmp_path / "run.json").read_text())
    assert reduction["classification"] == "success"
    assert reduction["tests_total"] == reduction["tests_passed"] == 5
    assert reduction["success_token"]
    assert run["arch_results"]["blackhole"]["verdict"] == "SUCCESS"
    assert run["arch_results"]["wormhole"]["tests_total"] == 3


def test_verification_reducer_cannot_hide_one_unexecuted_architecture(tmp_path):
    requirements = [_requirement(), _requirement("wormhole")]
    manifest, manifest_path = _reducer_manifest(tmp_path, requirements)
    results = tmp_path / "verification-results"
    results.mkdir()
    blackhole = _sealed_result(manifest, requirements[0])
    wormhole = _sealed_result(
        manifest, requirements[1], executed=0, passed=0, job_id="job-wormhole"
    )
    (results / "blackhole.json").write_text(json.dumps(blackhole), encoding="utf-8")
    (results / "wormhole.json").write_text(json.dumps(wormhole), encoding="utf-8")

    _reduce(tmp_path, manifest_path)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "coverage_error"
    assert reduction["success_token"] is None
    assert reduction["architecture_results"]["blackhole"]["verdict"] == "SUCCESS"
    assert reduction["architecture_results"]["wormhole"]["verdict"] != "SUCCESS"
    assert any("zero_executed" in reason for reason in reduction["reason_codes"])


def test_verification_reducer_retains_explicit_incomplete_outcome_coverage(tmp_path):
    requirement = _requirement()
    manifest, manifest_path = _reducer_manifest(tmp_path, [requirement])
    results = tmp_path / "verification-results"
    results.mkdir()
    result = _sealed_result(
        manifest,
        requirement,
        selected=3,
        executed=2,
        passed=2,
    )
    (results / "result.json").write_text(json.dumps(result), encoding="utf-8")

    _reduce(tmp_path, manifest_path)

    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "coverage_error"
    assert reduction["leaves"][0]["selected"] == 3
    assert reduction["leaves"][0]["executed"] == 2
    assert (
        "execution_outcome_count_incomplete" in reduction["leaves"][0]["reason_codes"]
    )
    assert reduction["success_token"] is None


@pytest.mark.parametrize(
    ("result_kwargs", "classification", "reason"),
    [
        (
            {"selected": 0, "executed": 0, "passed": 0},
            "coverage_error",
            "zero_selected",
        ),
        ({"executed": 0, "passed": 0}, "coverage_error", "zero_executed"),
        (
            {"collection_errors": 1, "collection_returncode": 2},
            "infra_error",
            "collection_error",
        ),
        ({"returncode": 2}, "infra_error", "execution_nonzero_exit"),
        ({"returncode": 5, "timed_out": True}, "infra_error", "execution_timed_out"),
        ({"markers": ["tt_fatal"]}, "infra_error", "tt_fatal"),
        (
            {"executed_artifact_sha256": "e" * 64},
            "infra_error",
            "identity_mismatch:executed_artifact_sha256",
        ),
    ],
)
def test_verification_reducer_rejects_false_green_evidence(
    tmp_path, result_kwargs, classification, reason
):
    requirement = _requirement()
    manifest, manifest_path = _reducer_manifest(tmp_path, [requirement])
    results = tmp_path / "verification-results"
    results.mkdir()
    result = _sealed_result(manifest, requirement, **result_kwargs)
    (results / "result.json").write_text(json.dumps(result), encoding="utf-8")

    _reduce(tmp_path, manifest_path)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == classification
    assert any(reason in value for value in reduction["reason_codes"])
    assert reduction["success_token"] is None


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("architecture", "wormhole", "identity_mismatch:architecture"),
        ("backend", "ttsim", "identity_mismatch:backend"),
        (
            "selector",
            {"test": "other.py", "test_id": None, "k": None},
            "identity_mismatch:selector",
        ),
        ("actual_base_sha", "f" * 40, "identity_mismatch:actual_base_sha"),
    ],
)
def test_verification_reducer_requires_exact_sealed_identity(
    tmp_path, field, value, reason
):
    requirement = _requirement()
    manifest, manifest_path = _reducer_manifest(tmp_path, [requirement])
    result = _sealed_result(manifest, requirement)
    if field == "actual_base_sha":
        result["provenance"][field] = value
    else:
        result[field] = value
    result["result_id"] = _content_id(result, {"result_id"})
    results = tmp_path / "verification-results"
    results.mkdir()
    (results / "result.json").write_text(json.dumps(result), encoding="utf-8")

    _reduce(tmp_path, manifest_path)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "infra_error"
    assert any(reason in value for value in reduction["reason_codes"])
    assert reduction["success_token"] is None


def test_verification_reducer_rejects_missing_and_mixed_patch_results(tmp_path):
    requirements = [_requirement(), _requirement("wormhole")]
    manifest, manifest_path = _reducer_manifest(tmp_path, requirements)
    results = tmp_path / "verification-results"
    results.mkdir()
    first = _sealed_result(manifest, requirements[0], patch_sha256="b" * 64)
    (results / "first.json").write_text(json.dumps(first), encoding="utf-8")
    _reduce(tmp_path, manifest_path)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "partial"
    assert any("result_missing" in value for value in reduction["reason_codes"])

    second = _sealed_result(
        manifest, requirements[1], patch_sha256="e" * 64, job_id="job-2"
    )
    (results / "second.json").write_text(json.dumps(second), encoding="utf-8")
    _reduce(tmp_path, manifest_path)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "infra_error"
    assert "patch_digest_mismatch" in reduction["reason_codes"]
    assert all(
        result["verdict"] == "ENV_ERROR"
        for result in reduction["architecture_results"].values()
    )


def test_verification_reducer_requires_explicit_performance_measurements(tmp_path):
    requirement = _requirement(
        suite="perf",
        minimum_selected=3,
        minimum_executed=3,
        selector={"test": "perf_reduce.py", "test_id": None, "k": None},
        required_measurements=["cycle_comparison", "repeatability"],
    )
    manifest, manifest_path = _reducer_manifest(tmp_path, [requirement])
    results = tmp_path / "verification-results"
    results.mkdir()
    result = _sealed_result(manifest, requirement, selected=3, executed=3, passed=3)
    (results / "perf.json").write_text(json.dumps(result), encoding="utf-8")

    _reduce(tmp_path, manifest_path)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "coverage_error"
    assert any(
        "required_measurement_result_missing_or_invalid" in value
        for value in reduction["reason_codes"]
    )

    perf_result = tmp_path / "perf_result.json"
    perf_result.write_text(
        json.dumps(
            {
                "outcome": "PERF_OK",
                "measured": True,
                "arch": "blackhole",
                "test": "perf_reduce.py",
                "base_commit": manifest["expected_base_sha"],
                "run_id": manifest["run_id"],
                "attempt_id": manifest["attempt_id"],
                "requirement_id": requirement["requirement_id"],
                "patch_sha256": result["provenance"]["patch_sha256"],
                "measurements": {
                    "cycle_comparison": {"measured": True},
                    "repeatability": {"measured": True, "executions": 3},
                },
            }
        ),
        encoding="utf-8",
    )
    _reduce(tmp_path, manifest_path, perf_result=perf_result)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "success"
    assert reduction["success_token"]


def test_verification_reducer_retains_foreign_attempt_and_rejects_duplicates(tmp_path):
    requirement = _requirement()
    manifest, manifest_path = _reducer_manifest(tmp_path, [requirement])
    results = tmp_path / "verification-results"
    results.mkdir()
    old = _sealed_result(
        manifest, requirement, attempt_id="attempt-000", job_id="old-job"
    )
    current = _sealed_result(manifest, requirement, job_id="current-job")
    (results / "old.json").write_text(json.dumps(old), encoding="utf-8")
    (results / "current.json").write_text(json.dumps(current), encoding="utf-8")
    _reduce(tmp_path, manifest_path)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "success"
    assert reduction["excluded_results"][0]["reason"] == "superseded_or_foreign_attempt"

    duplicate = _sealed_result(manifest, requirement, job_id="second-current-job")
    (results / "duplicate.json").write_text(json.dumps(duplicate), encoding="utf-8")
    _reduce(tmp_path, manifest_path)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "infra_error"
    assert "duplicate_current_result" in reduction["reason_codes"][0]
    assert reduction["success_token"] is None


def test_predeclared_xfail_requires_tracked_policy_and_replacement_coverage(tmp_path):
    worktree = tmp_path / "worktree"
    tests = worktree / "tt_metal" / "tt-llk" / "tests" / "python_tests"
    tests.mkdir(parents=True)
    (tests / "test_known_skip.py").write_text(
        "def test_known_limitation(): pass\n", encoding="utf-8"
    )
    (tests / "test_replacement.py").write_text(
        "def test_replacement_coverage(): pass\n", encoding="utf-8"
    )
    # _required_manifest supplies this shared selector fixture. Keep it in the
    # base commit so this test's candidate diff contains only the waiver edit.
    ttnn_test = (
        worktree / "tests" / "ttnn" / "unit_tests" / "operations" / "test_reduce.py"
    )
    ttnn_test.parent.mkdir(parents=True)
    ttnn_test.write_text("def test_reduce(): pass\n", encoding="utf-8")
    selector = {
        "test": "test_known_skip.py",
        "test_id": "test_known_skip.py::test_known_limitation",
        "k": None,
    }
    replacement_selector = {
        "test": "test_replacement.py",
        "test_id": "test_replacement.py::test_replacement_coverage",
        "k": None,
    }
    policy_path = worktree / "verification_waivers.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema": "tt.issue-solver.verification-waiver-policy",
                "version": 1,
                "policies": [
                    {
                        "policy_id": "known-architecture-xfail",
                        "approver": "llk-verification-owners",
                        "reason": "Known architecture limitation covered by an equivalent selector.",
                        "scope": {
                            "architecture": "blackhole",
                            "suite": "llk",
                            "backend": "silicon",
                            "selector": selector,
                        },
                        "replacement": {
                            "architecture": "blackhole",
                            "suite": "llk",
                            "backend": "silicon",
                            "selector": replacement_selector,
                            "minimum_selected": 1,
                            "minimum_executed": 1,
                            "required_measurements": [],
                        },
                        "allowed_outcomes": ["xfailed", "skipped"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(worktree), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "commit", "-q", "-m", "base policy"],
        check=True,
    )
    base = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Candidate-side policy changes are ignored; authority comes from base.
    policy_path.write_text("candidate-controlled invalid policy\n", encoding="utf-8")
    analysis = """\
## Scope
arch_scope:
  blackhole: in_scope
## Verification
verification_required: yes
verifiable_in_llk_suite: yes
llk_coverage: existing
"""
    plan = """\
## Test Strategy
reproduction_tests:
- arch: blackhole
  test: test_known_skip.py::test_known_limitation
"""
    proc, manifest_path = _required_manifest(
        tmp_path,
        analysis,
        plan,
        "--expected-base-sha",
        base,
        "--waiver-policy",
        "verification_waivers.json",
    )
    assert proc.returncode == 0
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["requirements"]) == 2
    assert manifest["waivers"][0]["policy_id"] == "known-architecture-xfail"
    assert manifest["waivers"][0]["policy_path"] == "verification_waivers.json"

    scope, replacement = manifest["requirements"]
    results = tmp_path / "verification-results"
    results.mkdir()
    xfailed = _sealed_result(
        manifest, scope, executed=0, passed=0, xfailed=1, job_id="job-xfail"
    )
    replacement_result = _sealed_result(manifest, replacement, job_id="job-replacement")
    (results / "xfailed.json").write_text(json.dumps(xfailed), encoding="utf-8")
    (results / "replacement.json").write_text(
        json.dumps(replacement_result), encoding="utf-8"
    )
    _reduce(tmp_path, manifest_path, worktree=worktree)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "success"
    assert reduction["success_token"]
    assert reduction["tests_total"] == 2
    assert reduction["tests_passed"] == 1
    waived = next(
        leaf
        for leaf in reduction["leaves"]
        if leaf["requirement_id"] == scope["requirement_id"]
    )
    assert waived["waived"] is True and waived["xfailed"] == 1
    assert waived["passed"] == 0
    assert waived["reason_codes"] == []
    assert reduction["reason_codes"] == []

    forged = json.loads(json.dumps(manifest))
    forged["waivers"][0]["approver"] = "candidate-agent"
    forged["waivers"][0]["waiver_id"] = _content_id(forged["waivers"][0], {"waiver_id"})
    forged["manifest_id"] = _content_id(forged, {"manifest_id"})
    forged_path = tmp_path / "forged_manifest.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    forged_reduction = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "reduce-verification",
            "--log-dir",
            str(tmp_path),
            "--manifest",
            str(forged_path),
            "--results-dir",
            str(results),
            "--scope",
            "all",
            "--output",
            str(tmp_path / "forged_reduction.json"),
            "--worktree",
            str(worktree),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert forged_reduction.returncode != 0
    assert "waiver is not policy-authorized" in forged_reduction.stderr

    (results / "replacement.json").unlink()
    _reduce(tmp_path, manifest_path, worktree=worktree)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] != "success"
    assert reduction["success_token"] is None

    retry, retry_manifest_path = _required_manifest(
        tmp_path,
        analysis,
        plan,
        "--expected-base-sha",
        base,
        "--supersedes-reason",
        "retry infrastructure failure",
    )
    retry_manifest = json.loads(retry_manifest_path.read_text())
    assert retry.returncode == 0 and retry_manifest["revision"] == 2
    assert retry_manifest["waivers"][0]["policy_id"] == "known-architecture-xfail"
    assert len(retry_manifest["requirements"]) == 2

    late = tmp_path / "late-waiver"
    revisions = late / "required_verification_manifests"
    revisions.mkdir(parents=True)
    unwaived = {
        **manifest,
        "manifest_id": "0" * 64,
        "requirements": [scope],
        "waivers": [],
    }
    unwaived["manifest_id"] = _content_id(unwaived, {"manifest_id"})
    (revisions / "revision-001.json").write_text(json.dumps(unwaived), encoding="utf-8")
    late_output = late / "required_verification_manifest.json"
    late_output.write_text(json.dumps(unwaived), encoding="utf-8")
    late_attempt = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "required-verification",
            "--log-dir",
            str(late),
            "--output",
            str(late_output),
            "--analysis",
            str(tmp_path / "analysis.md"),
            "--plan",
            str(tmp_path / "plan.md"),
            "--worktree",
            str(worktree),
            "--run-id",
            manifest["run_id"],
            "--expected-base-sha",
            base,
            "--architectures-json",
            '["blackhole"]',
            "--backend",
            "local",
            "--supersedes-reason",
            "observed an xfail",
            "--waiver-policy",
            "verification_waivers.json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert late_attempt.returncode != 0
    assert (
        "cannot introduce a verification waiver after revision 1" in late_attempt.stderr
    )


@pytest.mark.parametrize("seed", [20260804, 20260805, 20260806])
def test_verification_reducer_is_repeatable_for_fixed_random_attempt_trees(
    tmp_path, seed
):
    requirements = [
        _requirement("blackhole", "llk", index=1),
        _requirement("blackhole", "metal", index=1),
        _requirement("wormhole", "llk", index=1),
        _requirement("wormhole", "metal", index=1),
        _requirement("quasar", "llk", index=1),
    ]
    manifest, manifest_path = _reducer_manifest(tmp_path, requirements)
    results = tmp_path / "verification-results"
    results.mkdir()
    cases = [
        {},
        {"executed": 0, "passed": 0},
        {"passed": 0, "failed": 1, "returncode": 1},
        {"markers": ["tt_fatal"]},
        {"collection_errors": 1, "collection_returncode": 2},
    ]
    rng = random.Random(seed)
    choices = [rng.choice(cases) for _ in requirements]
    for index, (requirement, result_case) in enumerate(zip(requirements, choices)):
        result = _sealed_result(
            manifest, requirement, job_id=f"job-{index}", **result_case
        )
        (results / f"result-{index}.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

    _reduce(tmp_path, manifest_path)
    first = (tmp_path / "verification_reduction.json").read_bytes()
    _reduce(tmp_path, manifest_path)
    second = (tmp_path / "verification_reduction.json").read_bytes()
    assert first == second

    replay = random.Random(seed)
    assert choices == [replay.choice(cases) for _ in requirements]


@pytest.fixture
def audit_finalize_candidate(tmp_path):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.name", "Test"], check=True
    )
    source = worktree / "source.txt"
    source.write_text("base\n")
    subprocess.run(["git", "-C", str(worktree), "add", "source.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "commit", "-q", "-m", "base"], check=True
    )
    base = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source.write_text("verified\n")
    subprocess.run(["git", "-C", str(worktree), "commit", "-qam", "fix"], check=True)
    patch_sha256 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "candidate-patch-digest",
            "--worktree",
            str(worktree),
            "--expected-base-sha",
            base,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    requirement = _requirement()
    manifest, manifest_path = _reducer_manifest(tmp_path, [requirement])
    manifest["expected_base_sha"] = base
    manifest["manifest_id"] = _content_id(manifest, {"manifest_id"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    results = tmp_path / "verification-results"
    results.mkdir()
    result = _sealed_result(
        manifest, requirement, patch_sha256=patch_sha256, job_id="verified-job"
    )
    (results / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "runner_pool": "audit",
                "status": "running",
                "step_history": [],
            }
        ),
        encoding="utf-8",
    )
    _reduce(tmp_path, manifest_path)

    finalize_args = [
        sys.executable,
        str(SCRIPT),
        "finalize",
        "--log-dir",
        str(tmp_path),
        "--status",
        "success",
        "--final-result",
        "success",
        "--worktree",
        str(worktree),
    ]
    return finalize_args, worktree, source


def test_audit_finalize_accepts_current_and_rejects_changed_patch(
    tmp_path, audit_finalize_candidate
):
    finalize_args, worktree, source = audit_finalize_candidate
    finalized = subprocess.run(
        finalize_args, check=False, capture_output=True, text=True
    )
    assert finalized.returncode == 0, finalized.stderr
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "success"

    source.write_text("changed after verification\n")
    subprocess.run(["git", "-C", str(worktree), "commit", "-qam", "later"], check=True)
    finalized_bytes = (tmp_path / "run.json").read_bytes()
    finalized = subprocess.run(
        finalize_args, check=False, capture_output=True, text=True
    )
    assert finalized.returncode != 0
    assert "candidate patch differs from verified patch" in finalized.stderr
    assert (tmp_path / "run.json").read_bytes() == finalized_bytes


@pytest.mark.parametrize(
    "patch",
    [
        {"run_id": "different-run"},
        {".run_id.": "different-run"},
        {"attempt_id": "different-attempt"},
        {"runner_pool": "prod"},
        {"required_verification.manifest_id": "f" * 64},
        {"verification_reduction": {"reduction_id": "f" * 64}},
        {"review": {"requirements_complete": True}},
    ],
)
def test_audit_finalize_rejects_identity_and_evidence_patch(
    tmp_path, audit_finalize_candidate, patch
):
    finalize_args, _, _ = audit_finalize_candidate
    original = (tmp_path / "run.json").read_bytes()
    result = subprocess.run(
        [*finalize_args, "--patch-json", json.dumps(patch)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "finalize patch cannot change" in result.stderr
    assert (tmp_path / "run.json").read_bytes() == original


def test_audit_finalize_accepts_packaging_metrics(tmp_path, audit_finalize_candidate):
    finalize_args, worktree, _ = audit_finalize_candidate
    patch = {
        "base_commit": json.loads(
            (tmp_path / "required_verification_manifest.json").read_text()
        )["expected_base_sha"],
        "artifact_patch": "generated.patch",
        "worktree_dir": str(worktree),
        "debug_cycles": 2,
        "arch_results": {"blackhole": {"perf": {"measured": False}}},
    }
    result = subprocess.run(
        [*finalize_args, "--patch-json", json.dumps(patch)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["status"] == "success"
    assert run["artifact_patch"] == "generated.patch"
    assert run["debug_cycles"] == 2
    assert run["arch_results"]["blackhole"]["perf"] == {"measured": False}


@pytest.mark.parametrize("dotted", [False, True])
def test_finalize_typed_outcome_cannot_be_promoted_by_patch(tmp_path, dotted):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "audit-promotion",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "writer",
        "--first-message",
        "working",
        "--start-time",
        "2026-09-21T10:00:00Z",
        "--patch-json",
        '{"runner_pool":"audit"}',
    )
    values = {
        "status": "success",
        "final_result": "success",
        "final_message": "passed",
        "end_time": "2099-01-01T00:00:00Z",
        "duration_seconds": 99999,
        "solver_state": "working",
    }
    patch = {
        key + (".override" if dotted else ""): value for key, value in values.items()
    }
    _run(
        tmp_path,
        "finalize",
        "--status",
        "failed",
        "--final-result",
        "test_failure",
        "--final-message",
        "compile failed",
        "--solver-state",
        "not_working",
        "--end-time",
        "2026-09-21T10:01:00Z",
        "--patch-json",
        json.dumps(patch),
    )
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["status"] == "failed"
    assert run["final_result"] == "test_failure"
    assert run["final_message"] == "compile failed"
    assert run["solver_state"] == "not_working"
    assert run["end_time"] == "2026-09-21T10:01:00Z"
    assert run["duration_seconds"] == 60
    assert run["step_history"][-1]["result"] == "test_failure"
    assert not (tmp_path / "required_verification_manifest.json").exists()


@pytest.mark.parametrize("patch", [[], None, "wrong-shape"])
def test_finalize_rejects_non_object_patch_without_mutation(tmp_path, patch):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "writer",
        "--first-message",
        "working",
    )
    original = (tmp_path / "run.json").read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finalize",
            "--log-dir",
            str(tmp_path),
            "--status",
            "failed",
            "--final-result",
            "test_failure",
            "--patch-json",
            json.dumps(patch),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "finalize patch must be a JSON object" in result.stderr
    assert (tmp_path / "run.json").read_bytes() == original


def test_candidate_patch_digest_is_identical_from_llk_subdir_and_repo_root(tmp_path):
    worktree = tmp_path / "worktree"
    llk = worktree / "tt_metal" / "tt-llk"
    llk.mkdir(parents=True)
    (llk / "llk.txt").write_text("base llk\n")
    (worktree / "metal.txt").write_text("base metal\n")
    (worktree / "setup-owned.txt").write_text("original infrastructure\n")
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.name", "test"], check=True
    )
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(worktree), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "base"], check=True)
    base = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (llk / "llk.txt").write_bytes(b"changed\x00llk\n")
    (worktree / "metal.txt").write_text("changed metal\n")
    (worktree / "metal.txt").chmod(0o755)
    (worktree / "new-untracked.txt").write_text("new candidate input\n")
    subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "update-index",
            "--skip-worktree",
            "setup-owned.txt",
        ],
        check=True,
    )
    (worktree / "setup-owned.txt").write_text("local infrastructure overlay\n")
    index = (worktree / ".git/index").read_bytes()
    transport_index = tmp_path / "transport-index"
    transport_index.write_bytes(index)
    transport_env = {**os.environ, "GIT_INDEX_FILE": str(transport_index)}
    subprocess.run(
        ["git", "-C", str(worktree), "add", "-A", "--", "."],
        env=transport_env,
        check=True,
    )
    # Exact hardware-transport Git serialization, independently of the helper.
    transport_patch = subprocess.check_output(
        [
            "git",
            "-C",
            str(worktree),
            "diff",
            "--cached",
            "--binary",
            "--full-index",
            base,
            "--",
        ],
        env=transport_env,
    )
    assert b"GIT binary patch" in transport_patch
    assert b"new-untracked.txt" in transport_patch
    assert b"new mode 100755" in transport_patch
    assert b"setup-owned.txt" not in transport_patch
    expected = hashlib.sha256(transport_patch).hexdigest()

    def digest(path):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "candidate-patch-digest",
                "--worktree",
                str(path),
                "--expected-base-sha",
                base,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    for abbrev in ("7", "12"):
        subprocess.run(
            ["git", "-C", str(worktree), "config", "core.abbrev", abbrev], check=True
        )
        assert digest(llk) == digest(worktree) == expected
        assert (worktree / ".git/index").read_bytes() == index
        assert (worktree / "new-untracked.txt").is_file()
        assert not list((worktree / ".git").glob(".candidate-index-*"))
    run_test_source = RUN_TEST.read_text(encoding="utf-8")
    assert "candidate-patch-digest" in run_test_source
    assert "tt-llk-local-patch-v1" not in run_test_source


def test_candidate_patch_digest_rehashes_racy_same_size_binary(
    tmp_path, reviewed_candidate, monkeypatch
):
    import importlib.util

    wt, logs, git, review, result = reviewed_candidate
    git("config", "core.trustctime", "false")
    fixed_ns = 1_700_000_000_000_000_000
    binary = wt / "binary.dat"
    binary.write_bytes(b"\x00old")
    os.utime(binary, ns=(fixed_ns, fixed_ns))
    git("add", "-A")
    git("commit", "-qm", "binary base")
    base = git("rev-parse", "HEAD")
    index = wt / ".git/index"
    os.utime(index, ns=(fixed_ns, fixed_ns))
    original_index = index.read_bytes()
    binary.write_bytes(b"\x00new")
    os.utime(binary, ns=(fixed_ns, fixed_ns))

    # Deterministic control: newer copied-index mtime defeats Git's racy-clean
    # protection and reuses the old blob despite the changed binary bytes.
    naive_index = tmp_path / "naive-index"
    naive_index.write_bytes(original_index)
    os.utime(naive_index, ns=(fixed_ns + 10**10, fixed_ns + 10**10))
    env = {**os.environ, "GIT_INDEX_FILE": str(naive_index)}
    subprocess.run(["git", "-C", str(wt), "add", "-A"], env=env, check=True)
    assert (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(wt),
                "diff",
                "--cached",
                "--binary",
                "--full-index",
                base,
            ],
            env=env,
        )
        == b""
    )

    spec = importlib.util.spec_from_file_location("racy_index_writer", SCRIPT)
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    real_run = subprocess.run
    captured = []

    def capture_diff(*args, **kwargs):
        process = real_run(*args, **kwargs)
        if "--binary" in args[0]:
            captured.append(process.stdout)
        return process

    monkeypatch.setattr(writer.subprocess, "run", capture_diff)
    digest = writer._candidate_patch_digest(wt, base)
    assert len(captured) == 1
    patch = captured[0]
    assert b"GIT binary patch" in patch
    assert digest == hashlib.sha256(patch).hexdigest()
    assert index.read_bytes() == original_index
    assert index.stat().st_mtime_ns == fixed_ns
    assert not list((wt / ".git").glob(".candidate-index-*"))

    # Check actual reconstructed bytes, not just agreement between two hashes.
    replay = tmp_path / "replay"
    git("worktree", "add", "--detach", "-q", str(replay), base)
    subprocess.run(["git", "-C", str(replay), "apply", "-"], input=patch, check=True)
    assert (replay / "binary.dat").read_bytes() == b"\x00new"
    assert binary.read_bytes() == b"\x00new"


def test_run_test_isolates_artifacts_by_owner_and_full_source_content(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "artifact-roots.txt"
    pytest_bin = fake_bin / "pytest"
    pytest_bin.write_text(
        "#!/bin/bash\n"
        'collection=""; junit=""; producer=false; collect=false\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  case "$1" in\n'
        "    --codegen-collection-json)\n"
        '      [[ "${REJECT_COLLECTION_FLAG:-}" != 1 ]] || exit 4\n'
        '      collection="$2"; shift 2 ;;\n'
        "    --collect-only) collect=true; shift ;;\n"
        '    --junitxml) junit="$2"; shift 2 ;;\n'
        "    --compile-producer) producer=true; shift ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        'if [[ -n "$collection" ]]; then\n'
        '  mkdir -p "$(dirname "$collection")"\n'
        '  printf \'{"schema":"tt.issue-solver.pytest-collection","version":1,"selected":1,"collected":1,"errors":0,"returncode":0}\\n\' > "$collection"\n'
        "fi\n"
        'if [[ -n "$junit" ]]; then\n'
        '  mkdir -p "$(dirname "$junit")"\n'
        '  printf \'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="fake"/></testsuite></testsuites>\\n\' > "$junit"\n'
        "fi\n"
        'if [[ "$collect" == true && -z "$collection" ]]; then\n'
        "  printf 'test_fake.py::test_fake\\n'\n"
        "fi\n"
        'if [[ "$producer" == true ]]; then\n'
        '  artifact_root="$TT_LLK_ARTEFACTS_DIR"\n'
        '  [[ "${HISTORICAL_ARTIFACTS:-}" != 1 ]] || artifact_root="$RUNNER_TEMP/tt-llk-build"\n'
        '  printf \'%s\\n\' "$artifact_root" >> "$CAPTURE"\n'
        '  mkdir -p "$artifact_root"\n'
        '  touch "$artifact_root/fake-output"\n'
        "fi\n"
    )
    pytest_bin.chmod(0o755)

    def make_worktree(name, *, structured_collection=True):
        worktree = tmp_path / name
        test_dir = worktree / "tests" / "python_tests"
        helper_dir = test_dir / "helpers"
        compiler_dir = worktree / "tests" / "sfpi" / "compiler" / "bin"
        source_dir = worktree / "tt_llk_blackhole"
        writer_dir = worktree / "codegen" / "scripts"
        test_dir.mkdir(parents=True)
        helper_dir.mkdir()
        compiler_dir.mkdir(parents=True)
        source_dir.mkdir()
        writer_dir.mkdir(parents=True)
        (writer_dir / "run_json_writer.py").symlink_to(SCRIPT)
        (test_dir / "test_fake.py").write_text("def test_fake(): pass\n")
        (test_dir / "conftest.py").write_text(
            'COLLECTION_OPTION = "--codegen-collection-json"\n'
            if structured_collection
            else "# historical harness without structured collection option\n"
        )
        (helper_dir / "test_config.py").write_text(
            'ARTEFACTS_ENV = "TT_LLK_ARTEFACTS_DIR"\n'
            if structured_collection
            else "# historical harness only honors RUNNER_TEMP\n"
        )
        compiler = compiler_dir / "riscv-tt-elf-g++"
        compiler.write_bytes(b"compiler-v1")
        compiler.chmod(0o755)
        header = source_dir / "kernel.hpp"
        header.write_text("#define VALUE_A 1\n")
        subprocess.run(["git", "init", "-q", str(worktree)], check=True)
        subprocess.run(
            ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(worktree), "config", "user.name", "Test"],
            check=True,
        )
        subprocess.run(["git", "-C", str(worktree), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(worktree), "commit", "-q", "-m", "fixture"],
            check=True,
        )
        return worktree, header

    first_worktree, header = make_worktree("attempt-one")
    second_worktree, _ = make_worktree("attempt-two")
    managed_root = tmp_path / "managed-artifacts"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "TT_LLK_LOCAL_ARTIFACT_ROOT": str(managed_root),
    }

    def compile_in(worktree, log_dir):
        return subprocess.run(
            [
                "bash",
                str(RUN_TEST),
                "compile",
                "--worktree",
                str(worktree),
                "--arch",
                "blackhole",
                "--test",
                "test_fake.py",
                "--log-dir",
                str(log_dir),
            ],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    compile_in(first_worktree, tmp_path / "logs-one")
    compile_in(second_worktree, tmp_path / "logs-two")
    roots = capture.read_text().splitlines()
    assert len(roots) == 2
    assert roots[0] != roots[1]
    assert all(Path(root).is_relative_to(managed_root / "v2") for root in roots)

    historical_worktree, _ = make_worktree(
        "historical-attempt", structured_collection=False
    )
    historical = subprocess.run(
        [
            "bash",
            str(RUN_TEST),
            "compile",
            "--worktree",
            str(historical_worktree),
            "--arch",
            "blackhole",
            "--test",
            "test_fake.py",
            "--log-dir",
            str(tmp_path / "logs-historical"),
        ],
        env={
            **env,
            "REJECT_COLLECTION_FLAG": "1",
            "HISTORICAL_ARTIFACTS": "1",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert historical.returncode == 0, historical.stderr
    assert "test_fake.py::test_fake" in historical.stderr
    historical_root = Path(capture.read_text().splitlines()[-1])
    assert historical_root.name == "tt-llk-build"
    assert historical_root.is_relative_to(managed_root / "v2")

    original = header.stat()
    header.write_text("#define VALUE_B 1\n")  # same size; only content differs
    os.utime(header, ns=(original.st_atime_ns, original.st_mtime_ns))
    compile_in(first_worktree, tmp_path / "logs-one")
    changed_root = capture.read_text().splitlines()[-1]
    assert changed_root != roots[0]

    compile_in(first_worktree, tmp_path / "logs-one")
    assert capture.read_text().splitlines()[-1] == changed_root

    bound_log = tmp_path / "logs-one"
    required_path = bound_log / "required_verification_manifest.json"
    required = {
        "schema": "tt.issue-solver.required-verification",
        "version": 1,
        "manifest_id": "0" * 64,
        "run_id": "run-bound",
        "attempt_id": "attempt-004",
        "expected_base_sha": subprocess.run(
            ["git", "-C", str(first_worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "revision": 4,
        "parent_manifest_id": "1" * 64,
        "supersedes_reason": "fixture retry",
        "requirements": [
            {
                "requirement_id": "blackhole:llk:2",
                "architecture": "blackhole",
                "suite": "llk",
                "backend": "silicon",
                "selector": {"test": "test_fake.py", "test_id": None, "k": None},
                "minimum_selected": 1,
                "minimum_executed": 1,
                "required_measurements": [],
            }
        ],
        "waivers": [],
    }
    required["manifest_id"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in required.items() if key != "manifest_id"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    required_path.write_text(json.dumps(required))
    (bound_log / "state.json").write_text(
        json.dumps({"REQUIRED_VERIFICATION_MANIFEST": str(required_path)})
    )
    result_path = bound_log / "verification-result.json"
    completed = subprocess.run(
        [
            "bash",
            str(RUN_TEST),
            "run",
            "--worktree",
            str(first_worktree),
            "--arch",
            "blackhole",
            "--test",
            "test_fake.py",
            "--log-dir",
            str(tmp_path / "logs-one"),
            "--result-json-out",
            str(result_path),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    local_result = json.loads(result_path.read_text())
    assert local_result["schema"] == "tt.issue-solver.verification-result"
    assert local_result["classification"] == "success"
    assert local_result["run_id"] == "run-bound"
    assert local_result["attempt_id"] == "attempt-004"
    assert local_result["requirement_id"] == "blackhole:llk:2"
    assert local_result["execution"]["executed"] == 1
    assert local_result["provenance"]["artifact_set_sha256"] == (
        local_result["provenance"]["executed_artifact_sha256"]
    )

    prior_manifest_id = required["manifest_id"]
    required["attempt_id"] = "attempt-005"
    required["revision"] = 5
    required["parent_manifest_id"] = prior_manifest_id
    required["supersedes_reason"] = "performance verification fixture"
    required["requirements"][0]["requirement_id"] = "blackhole:perf:1"
    required["requirements"][0]["suite"] = "perf"
    required["manifest_id"] = _content_id(required, {"manifest_id"})
    required_path.write_text(json.dumps(required))
    perf_result_path = bound_log / "performance-verification-result.json"
    perf_completed = subprocess.run(
        [
            "bash",
            str(RUN_TEST),
            "run",
            "--worktree",
            str(first_worktree),
            "--arch",
            "blackhole",
            "--test",
            "test_fake.py",
            "--log-dir",
            str(bound_log),
            "--result-json-out",
            str(perf_result_path),
        ],
        env={**env, "CODEGEN_VERIFICATION_SUITE": "perf"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert perf_completed.returncode == 0, perf_completed.stderr
    perf_result = json.loads(perf_result_path.read_text())
    assert perf_result["suite"] == "perf"
    assert perf_result["attempt_id"] == "attempt-005"
    assert perf_result["requirement_id"] == "blackhole:perf:1"


def test_init_emits_dashboard_fields(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "test_2026-04-17_issue_1_abcd1234",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "Analyzing",
        "--git-branch",
        "llk_code_gen/issue-1-v1",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["git_branch"] == "llk_code_gen/issue-1-v1"
    assert "num_turns" in doc
    assert doc["num_turns"] == 0
    assert doc["tokens"] == {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_creation": 0,
        "total": 0,
        "cost_usd": 0,
    }
    assert doc.get("solver_state") is None  # only set by finalize
    # --version omitted -> null (backward compatible; Quasar codegen omits it).
    assert doc.get("version") is None


def test_progress_sequence_and_heartbeat_advance_monotonically(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "progress-1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "Analyzing",
    )
    first = json.loads((tmp_path / "run.json").read_text())
    _run(tmp_path, "message", "--message", "Still analyzing")
    second = json.loads((tmp_path / "run.json").read_text())
    _run(
        tmp_path,
        "advance",
        "--new-step",
        "writer",
        "--new-message",
        "Writing",
        "--prev-result",
        "success",
    )
    third = json.loads((tmp_path / "run.json").read_text())

    assert [
        first["progress_sequence"],
        second["progress_sequence"],
        third["progress_sequence"],
    ] == [1, 2, 3]
    assert (
        first["last_heartbeat"] <= second["last_heartbeat"] <= third["last_heartbeat"]
    )
    assert third["supervisor_phase"] == "active_compute"

    writers = [
        subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "message",
                "--message",
                f"parallel-{i}",
                "--log-dir",
                str(tmp_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in range(8)
    ]
    for writer in writers:
        stdout, stderr = writer.communicate(timeout=10)
        assert writer.returncode == 0, stderr or stdout
    final = json.loads((tmp_path / "run.json").read_text())
    assert final["progress_sequence"] == 11


def test_concurrent_run_json_patches_preserve_every_writer(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "concurrent-run",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "Analyzing",
    )
    writers = [
        subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "metric",
                "--patch-json",
                json.dumps({f"concurrent.field_{index}": index}),
                "--log-dir",
                str(tmp_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(24)
    ]
    for writer in writers:
        stdout, stderr = writer.communicate(timeout=10)
        assert writer.returncode == 0, stderr or stdout

    final = json.loads((tmp_path / "run.json").read_text())
    assert final["concurrent"] == {f"field_{index}": index for index in range(24)}
    assert final["progress_sequence"] == 25


def test_state_updates_are_locked_and_phase_sequence_tracks_transitions(tmp_path):
    state_path = tmp_path / "state.json"
    writers = [
        subprocess.Popen(
            [
                sys.executable,
                str(STATE),
                "--file",
                str(state_path),
                "set",
                f"FIELD_{index}",
                str(index),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(24)
    ]
    for writer in writers:
        stdout, stderr = writer.communicate(timeout=10)
        assert writer.returncode == 0, stderr or stdout
    for phase in ("active_compute", "hardware_queue_wait", "finalization"):
        subprocess.run(
            [
                sys.executable,
                str(STATE),
                "--file",
                str(state_path),
                "set",
                "SUPERVISOR_PHASE",
                phase,
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(STATE),
            "--file",
            str(state_path),
            "set",
            "SUPERVISOR_PHASE",
            "finalization",
        ],
        check=True,
    )

    state = json.loads(state_path.read_text())
    assert {state[f"FIELD_{index}"] for index in range(24)} == {
        str(index) for index in range(24)
    }
    assert state["SUPERVISOR_PHASE"] == "finalization"
    assert state["SUPERVISOR_PHASE_SEQUENCE"] == 3
    assert state["SUPERVISOR_PHASE_CHANGED_AT"]


def test_runs_jsonl_upserts_are_locked_across_concurrent_finalizers(tmp_path):
    runs_jsonl = tmp_path / "runs.jsonl"
    logs = []
    for run_id in ("run-a", "run-b"):
        log_dir = tmp_path / run_id
        log_dir.mkdir()
        (log_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "end_time": "now",
                }
            )
        )
        logs.append(log_dir)
    writers = [
        subprocess.Popen(
            [
                sys.executable,
                str(RUN_UTILS),
                "upsert-runs-jsonl",
                "--log-dir",
                str(log_dir),
                "--runs-jsonl",
                str(runs_jsonl),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for log_dir in logs
    ]
    for writer in writers:
        stdout, stderr = writer.communicate(timeout=10)
        assert writer.returncode == 0, stderr or stdout

    rows = [json.loads(line) for line in runs_jsonl.read_text().splitlines()]
    assert {row["run_id"] for row in rows} == {"run-a", "run-b"}


def test_init_records_version_when_passed(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "test_2026-04-17_issue_2_abcd1234",
        "--kernel",
        "issue_2",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "Analyzing",
        "--version",
        "1.2.3",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["version"] == "1.2.3"


def test_init_records_audit_lane_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEGEN_RUNNER_POOL", "audit")
    monkeypatch.setenv("CODEGEN_BASE_COMMIT", "a" * 40)
    monkeypatch.setenv("CODEGEN_CAMPAIGN_ID", "infra-audit")
    monkeypatch.setenv("CODEGEN_ATTEMPT_ID", "try-1")
    monkeypatch.setenv("CODEGEN_RESUME_RUN_ID", "source-run")
    monkeypatch.setenv("CODEGEN_RESUME_ATTEMPT_ID", "source-attempt")
    monkeypatch.setenv("CODEGEN_RESUME_CHECKPOINT_DIGEST", "c" * 64)
    monkeypatch.setenv("CODEGEN_RESUME_PATCH_SHA256", "d" * 64)
    monkeypatch.setenv("CODEGEN_RESUME_VERIFICATION_REUSE", "invalidated")
    monkeypatch.setenv("CODEGEN_RESUME_INVALIDATION_REASON", "attempt_identity_changed")
    _run(
        tmp_path,
        "init",
        "--run-id",
        "audit-1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "Analyzing",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["runner_pool"] == "audit"
    assert doc["base_commit"] == "a" * 40
    assert doc["campaign_id"] == "infra-audit"
    assert doc["attempt_id"] == "try-1"
    assert doc["resumed_from_run_id"] == "source-run"
    assert doc["resumed_from_attempt_id"] == "source-attempt"
    assert doc["resume_checkpoint_digest"] == "c" * 64
    assert doc["resume_patch_sha256"] == "d" * 64
    assert doc["resume_verification"] == {
        "outcome": "invalidated",
        "reason_code": "attempt_identity_changed",
    }


@pytest.mark.parametrize(
    "queue_env",
    [
        {"CODEGEN_ATTEMPT_ID": "attempt-1"},
        {"CODEGEN_CAMPAIGN_ID": "campaign-1"},
        {"CODEGEN_RUNNER_POOL": "prod"},
    ],
)
def test_queued_worktree_setup_rejects_missing_exact_base(tmp_path, queue_env):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    command = r"""
source "$1"
REPO_ROOT="$2"
unset CODEGEN_BASE_COMMIT
resolve_worktree_base
"""

    result = subprocess.run(
        ["bash", "-c", command, "bash", str(SETUP_WORKTREE), str(repo)],
        env={**os.environ, **queue_env},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "queued launch requires an exact CODEGEN_BASE_COMMIT" in result.stderr


@pytest.mark.parametrize(
    "artifact", ["generated.patch", "supervisor-checkpoint.patch", "../escaped.patch"]
)
@pytest.mark.parametrize("timeout_classification", ["outer_timeout", "wall_timeout"])
@pytest.mark.parametrize("legacy_abbreviated_patch", [False, True])
def test_setup_worktree_records_exact_base_before_bootstrap(
    tmp_path, timeout_classification, legacy_abbreviated_patch, artifact
):
    repo = tmp_path / "repo"
    llk_tests = repo / "tt_metal" / "tt-llk" / "tests"
    llk_tests.mkdir(parents=True)
    setup_env = llk_tests / "setup_testing_env.sh"
    setup_env.write_text("#!/bin/bash\nexit 0\n")
    setup_env.chmod(0o755)
    # Versioned bootstrap reads the pinned worktree SFPI metadata and output.
    sfpi_info = llk_tests / "sfpi-info.sh"
    sfpi_info.write_text("#!/bin/bash\necho sfpi_version=fixture\n")
    sfpi_info.chmod(0o755)
    (llk_tests / "sfpi").mkdir()
    (llk_tests / "sfpi/sfpi.version").write_text("fixture\n")
    (repo / "tt_metal" / "tt-llk" / ".gitignore").write_text("*.pyc\n")
    (repo / "source.txt").write_text("base\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (repo / "source.txt").write_text("resumed candidate\n")
    subprocess.run(["git", "-C", str(repo), "config", "core.abbrev", "7"], check=True)
    patch = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--binary",
            *([] if legacy_abbreviated_patch else ["--full-index"]),
            base,
            "--",
            "source.txt",
        ],
        check=True,
        capture_output=True,
    ).stdout
    (repo / "source.txt").write_text("base\n")
    source_run_id = "2026-08-07_issue_5_source"
    source_attempt = "source-attempt"
    source_dir = tmp_path / source_run_id
    source_dir.mkdir()
    (source_dir / artifact).write_bytes(patch)
    checkpoint = {
        "run_id": source_run_id,
        "attempt_id": source_attempt,
        "base_commit": base,
        "artifact_patch": artifact,
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "completed_results": {"tests_total": 1, "tests_passed": 1},
    }
    checkpoint["checkpoint_digest"] = hashlib.sha256(
        json.dumps(
            checkpoint,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    (source_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": source_run_id,
                "attempt_id": source_attempt,
                "issue": {"number": 5},
                "status": "failed",
                "end_time": "2026-08-07T01:00:00Z",
                "timeout_classification": timeout_classification,
                "base_commit": base,
                "last_checkpoint": checkpoint,
            }
        )
    )
    resume_env = {
        **os.environ,
        "CODEGEN_BASE_COMMIT": base,
        "CODEGEN_ATTEMPT_ID": "new-attempt",
        "CODEGEN_RESUME_RUN_DIR": str(source_dir),
        "CODEGEN_RESUME_RUN_ID": source_run_id,
        "CODEGEN_RESUME_ATTEMPT_ID": source_attempt,
        "CODEGEN_RESUME_CHECKPOINT_DIGEST": checkpoint["checkpoint_digest"],
        "CODEGEN_RESUME_PATCH_SHA256": checkpoint["patch_sha256"],
        "CODEGEN_RESUME_VERIFICATION_REUSE": "invalidated",
        "CODEGEN_RESUME_INVALIDATION_REASON": "attempt_identity_changed",
    }
    worktrees = tmp_path / "worktrees"
    command = r"""
source "$1"
REPO_ROOT="$2"
LLK_REL="tt_metal/tt-llk"
CODEGEN_GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-dir)"
CODEGEN_SETUP_LOCK="${CODEGEN_GIT_DIR}/codegen-worktree-setup.lock"
CODEGEN_WORKTREE_ROOT="$3"
CODEGEN_BASE_COMMIT="$4"
setup_worktree "${5:-issue-5}"
"""
    proc = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(SETUP_WORKTREE),
            str(repo),
            str(worktrees),
            base,
        ],
        env=resume_env,
        check=False,
        capture_output=True,
        text=True,
    )
    worktree = worktrees / "issue-5-v1"
    if artifact == "../escaped.patch":
        assert proc.returncode != 0
        assert "checkpoint patch identity is missing" in proc.stderr
        assert not worktree.exists()
        return
    if legacy_abbreviated_patch:
        assert proc.returncode != 0
        assert (
            "imported candidate differs from the retained checkpoint patch"
            in proc.stderr
        )
        assert not worktree.exists()
        return
    assert proc.returncode == 0, proc.stderr

    state = json.loads(
        (worktree / "tt_metal" / "tt-llk" / ".codegen_run_state.json").read_text()
    )
    actual = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert state["EXPECTED_BASE_COMMIT"] == base
    assert state["SETUP_BASE_COMMIT"] == base
    assert state["BASE_COMMIT_WAS_PINNED"] is True
    assert state["QUEUE_LAUNCH"] is True
    assert state["QUEUE_ATTEMPT_ID"] == "new-attempt"
    assert state["RESUMED_FROM_RUN_ID"] == source_run_id
    assert state["RESUMED_FROM_ATTEMPT_ID"] == source_attempt
    assert state["RESUME_CHECKPOINT_DIGEST"] == checkpoint["checkpoint_digest"]
    assert state["RESUME_PATCH_SHA256"] == checkpoint["patch_sha256"]
    assert state["RESUME_VERIFICATION_REUSE"] == "invalidated"
    assert state["RESUME_INVALIDATION_REASON"] == "attempt_identity_changed"
    assert actual == base
    assert (worktree / "source.txt").read_text() == "resumed candidate\n"

    bad_digest = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(SETUP_WORKTREE),
            str(repo),
            str(worktrees),
            base,
        ],
        env={**resume_env, "CODEGEN_RESUME_CHECKPOINT_DIGEST": "f" * 64},
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad_digest.returncode != 0
    assert "checkpoint digest mismatch" in bad_digest.stderr
    assert not (worktrees / "issue-5-v2").exists()

    (source_dir / artifact).write_bytes(patch + b"\nmutation\n")
    bad_patch = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(SETUP_WORKTREE),
            str(repo),
            str(worktrees),
            base,
        ],
        env=resume_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad_patch.returncode != 0
    assert "patch digest mismatch" in bad_patch.stderr
    assert not (worktrees / "issue-5-v2").exists()
    (source_dir / artifact).write_bytes(patch)

    subprocess.run(
        ["git", "-C", str(repo), "commit", "--allow-empty", "-qm", "later base"],
        check=True,
    )
    later_base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    bad_base = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(SETUP_WORKTREE),
            str(repo),
            str(worktrees),
            later_base,
        ],
        env={**resume_env, "CODEGEN_BASE_COMMIT": later_base},
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad_base.returncode != 0
    assert "source base mismatch" in bad_base.stderr
    assert not (worktrees / "issue-5-v2").exists()

    normal_env = {
        key: value
        for key, value in resume_env.items()
        if not key.startswith("CODEGEN_RESUME_")
    }
    normal_env.update(
        {
            "CODEGEN_BASE_COMMIT": later_base,
            "CODEGEN_ATTEMPT_ID": "ordinary-attempt",
        }
    )
    normal = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(SETUP_WORKTREE),
            str(repo),
            str(worktrees),
            later_base,
            "issue-6",
        ],
        env=normal_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert normal.returncode == 0, normal.stderr
    normal_state = json.loads(
        (
            worktrees / "issue-6-v1" / "tt_metal" / "tt-llk" / ".codegen_run_state.json"
        ).read_text()
    )
    assert "RESUMED_FROM_RUN_ID" not in normal_state


def test_validate_input_rejects_unset_changed_or_drifted_base(tmp_path):
    worktree = tmp_path / "worktree"
    llk = worktree / "tt_metal" / "tt-llk"
    llk.mkdir(parents=True)
    (llk / ".gitignore").write_text(".codegen_run_state.json\n")
    (llk / "source.txt").write_text("base\n")
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.name", "test"], check=True
    )
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(worktree), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "base"], check=True)
    base = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = {
        "RUN_MODE": "single",
        "ISSUE_NUMBER": "5",
        "ISSUE_TITLE": "Pinned base",
        "WORKTREE_BRANCH": "test/base",
        "TEST_BACKEND": "local",
        "CREATE_LOCAL_BRANCH": "yes",
        "CREATE_PR": "no",
        "TARGET_ARCH": "blackhole",
        "TARGET_ARCHES": "",
        "EXPECTED_BASE_COMMIT": base,
        "SETUP_BASE_COMMIT": base,
        "BASE_COMMIT_WAS_PINNED": True,
    }
    (llk / ".codegen_run_state.json").write_text(json.dumps(state))
    command = 'source "$1"; execute_step_validate_input "$2"'

    accepted = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR_STEPS), str(worktree)],
        env={**os.environ, "CODEGEN_BASE_COMMIT": base},
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert f"BASE={base}" in accepted.stdout

    state["QUEUE_LAUNCH"] = True
    state["BASE_COMMIT_WAS_PINNED"] = False
    (llk / ".codegen_run_state.json").write_text(json.dumps(state))
    rejected_queue_fallback = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR_STEPS), str(worktree)],
        env={**os.environ, "CODEGEN_BASE_COMMIT": base},
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_queue_fallback.returncode == 1
    assert "queued launch was not created from an exact pinned base" in (
        rejected_queue_fallback.stdout
    )
    state["BASE_COMMIT_WAS_PINNED"] = True
    (llk / ".codegen_run_state.json").write_text(json.dumps(state))

    unset = {
        key: value for key, value in os.environ.items() if key != "CODEGEN_BASE_COMMIT"
    }
    rejected_unset = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR_STEPS), str(worktree)],
        env=unset,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_unset.returncode == 1
    assert "queued launch lost CODEGEN_BASE_COMMIT after setup" in rejected_unset.stdout

    rejected_changed = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR_STEPS), str(worktree)],
        env={**os.environ, "CODEGEN_BASE_COMMIT": "f" * 40},
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_changed.returncode == 1
    assert "changed after setup" in rejected_changed.stdout

    (llk / "source.txt").write_text("resumed candidate\n")
    candidate_digest = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "candidate-patch-digest",
            "--worktree",
            str(worktree),
            "--expected-base-sha",
            base,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state.update(
        {
            "QUEUE_ATTEMPT_ID": "new-attempt",
            "RESUMED_FROM_RUN_ID": "source-run",
            "RESUMED_FROM_ATTEMPT_ID": "source-attempt",
            "RESUME_CHECKPOINT_DIGEST": "c" * 64,
            "RESUME_PATCH_SHA256": candidate_digest,
            "RESUME_VERIFICATION_REUSE": "invalidated",
            "RESUME_INVALIDATION_REASON": "attempt_identity_changed",
        }
    )
    (llk / ".codegen_run_state.json").write_text(json.dumps(state))
    resume_env = {
        **os.environ,
        "CODEGEN_BASE_COMMIT": base,
        "CODEGEN_ATTEMPT_ID": "new-attempt",
        "CODEGEN_RESUME_RUN_ID": "source-run",
        "CODEGEN_RESUME_ATTEMPT_ID": "source-attempt",
        "CODEGEN_RESUME_CHECKPOINT_DIGEST": "c" * 64,
        "CODEGEN_RESUME_PATCH_SHA256": candidate_digest,
        "CODEGEN_RESUME_VERIFICATION_REUSE": "invalidated",
        "CODEGEN_RESUME_INVALIDATION_REASON": "attempt_identity_changed",
    }
    accepted_resume = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR_STEPS), str(worktree)],
        env=resume_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted_resume.returncode == 0, (
        accepted_resume.stdout + accepted_resume.stderr
    )

    (llk / "source.txt").write_text("mutated after resume setup\n")
    rejected_resume_mutation = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR_STEPS), str(worktree)],
        env=resume_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_resume_mutation.returncode == 1
    assert (
        "resumed candidate patch changed after setup" in rejected_resume_mutation.stdout
    )

    (llk / "source.txt").write_text("drift\n")
    subprocess.run(["git", "-C", str(worktree), "commit", "-qam", "drift"], check=True)
    rejected_drift = subprocess.run(
        ["bash", "-c", command, "bash", str(ORCHESTRATOR_STEPS), str(worktree)],
        env=resume_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_drift.returncode == 1
    assert "base drift before agent execution" in rejected_drift.stdout


def test_issue_url_preserved(tmp_path):
    issue = {
        "number": 1148,
        "title": "Foo",
        "url": "https://github.com/x/y/issues/1148",
        "labels": [],
    }
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1148",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
        "--issue",
        json.dumps(issue),
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["issue"]["url"] == "https://github.com/x/y/issues/1148"


def test_finalize_sets_solver_state(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "start",
    )
    _run(
        tmp_path,
        "finalize",
        "--status",
        "success",
        "--final-result",
        "success",
        "--final-message",
        "done",
        "--solver-state",
        "working",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["solver_state"] == "working"
    assert doc["status"] == "success"
    assert doc["final_result"] == "success"
    assert doc["final_message"] == "done"


def test_finalize_rejects_bad_solver_state(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "start",
    )
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "finalize",
            "--log-dir",
            str(tmp_path),
            "--status",
            "success",
            "--final-result",
            "success",
            "--final-message",
            "x",
            "--solver-state",
            "bogus",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
    assert "bogus" in (r.stderr + r.stdout)


def test_finalize_without_solver_state_preserves_none(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "start",
    )
    _run(
        tmp_path,
        "finalize",
        "--status",
        "success",
        "--final-result",
        "success",
        "--final-message",
        "done",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["solver_state"] is None


def test_finalize_solver_state_wins_over_patch_json(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "start",
    )
    _run(
        tmp_path,
        "finalize",
        "--status",
        "success",
        "--final-result",
        "success",
        "--final-message",
        "done",
        "--solver-state",
        "working",
        "--patch-json",
        '{"solver_state": "bogus_via_patch"}',
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["solver_state"] == "working"


def test_finalize_computes_duration_seconds(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r1",
        "--kernel",
        "issue_1",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "start",
        "--start-time",
        "2026-04-17T12:00:00Z",
    )
    _run(
        tmp_path,
        "finalize",
        "--status",
        "success",
        "--final-result",
        "success",
        "--final-message",
        "done",
        "--end-time",
        "2026-04-17T12:03:45Z",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["duration_seconds"] == 225  # 3m45s


# --------------------------------------------------------------------------
# Legacy multi-arch grouping — issue_run_id + sibling_runs
#
# Older multi-arch issue-solver runs produced N per-arch runs, each with its
# own run.json. They are grouped via an `issue_run_id` and a `sibling_runs`
# array. New multi-arch issue-solver runs use one run.json with arch="multi",
# target_arches, and arch_results; these fields stay optional for backwards
# compatibility.
# --------------------------------------------------------------------------


def test_init_without_multi_arch_fields_defaults(tmp_path):
    """Single-arch (today's default) runs get issue_run_id=None, sibling_runs=[]."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_solo",
        "--kernel",
        "issue_42",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["issue_run_id"] is None
    assert doc["sibling_runs"] == []


def test_init_accepts_issue_run_id(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_bh",
        "--kernel",
        "issue_1089",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
        "--issue-run-id",
        "issue-1089-multi-abc",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["issue_run_id"] == "issue-1089-multi-abc"


def test_init_accepts_sibling_runs(tmp_path):
    siblings = [{"arch": "wormhole", "run_id": "r_wh"}]
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_bh",
        "--kernel",
        "issue_1089",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
        "--sibling-runs",
        json.dumps(siblings),
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["sibling_runs"] == siblings


def test_init_accepts_single_run_multi_arch_patch(tmp_path):
    """init with a multi-arch patch-json stores target_arches, arch_results, and multi_arch_run."""
    arch_results = {
        "wormhole": {"status": "pending"},
        "blackhole": {"status": "pending"},
    }
    _run(
        tmp_path,
        "init",
        "--run-id",
        "issue_11384_multi",
        "--kernel",
        "issue_11384",
        "--arch",
        "multi",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
        "--patch-json",
        json.dumps(
            {
                "target_arches": ["wormhole", "blackhole"],
                "arch_results": arch_results,
                "multi_arch_run": True,
            }
        ),
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["arch"] == "multi"
    assert doc["target_arches"] == ["wormhole", "blackhole"]
    assert doc["arch_results"] == arch_results
    assert doc["multi_arch_run"] is True
    assert doc["issue_run_id"] is None
    assert doc["sibling_runs"] == []


def test_metric_merges_nested_arch_results(tmp_path):
    """metric deep-merges per-arch entries so updating one arch does not wipe the others."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "issue_11384_multi",
        "--kernel",
        "issue_11384",
        "--arch",
        "multi",
        "--first-step",
        "tester",
        "--first-message",
        "go",
        "--patch-json",
        json.dumps(
            {
                "arch_results": {
                    "wormhole": {"status": "pending", "tests_total": 0},
                    "blackhole": {"status": "pending", "tests_total": 0},
                }
            }
        ),
    )
    _run(
        tmp_path,
        "metric",
        "--patch-json",
        json.dumps(
            {
                "arch_results": {
                    "wormhole": {
                        "status": "done",
                        "verdict": "SUCCESS",
                        "tests_total": 32,
                    }
                },
                "tests_total": 32,
            }
        ),
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["tests_total"] == 32
    assert doc["arch_results"]["wormhole"]["status"] == "done"
    assert doc["arch_results"]["wormhole"]["verdict"] == "SUCCESS"
    assert doc["arch_results"]["blackhole"] == {"status": "pending", "tests_total": 0}


def test_metric_accepts_dotted_keys_as_nested_compatibility(tmp_path):
    """metric expands dotted keys (e.g. arch_results.wormhole.verdict) into nested dicts."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "issue_11384_multi",
        "--kernel",
        "issue_11384",
        "--arch",
        "multi",
        "--first-step",
        "tester",
        "--first-message",
        "go",
    )
    _run(
        tmp_path,
        "metric",
        "--patch-json",
        '{"arch_results.wormhole.verdict": "SUCCESS"}',
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["arch_results"]["wormhole"]["verdict"] == "SUCCESS"
    assert "arch_results.wormhole.verdict" not in doc


def test_link_siblings_replaces_sibling_runs(tmp_path):
    """link-siblings patches the sibling_runs list on an existing run.json."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_bh",
        "--kernel",
        "issue_1089",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
    )
    siblings = [
        {"arch": "wormhole", "run_id": "r_wh"},
        {"arch": "quasar", "run_id": "r_qs"},
    ]
    _run(tmp_path, "link-siblings", "--siblings", json.dumps(siblings))
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["sibling_runs"] == siblings


def test_link_siblings_sets_issue_run_id(tmp_path):
    """link-siblings --issue-run-id sets the shared issue_run_id used for dashboard grouping."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_bh",
        "--kernel",
        "issue_1089",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
    )
    _run(
        tmp_path,
        "link-siblings",
        "--issue-run-id",
        "issue-1089-shared",
        "--siblings",
        "[]",
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["issue_run_id"] == "issue-1089-shared"
    # link-siblings also accepts an empty siblings list (valid — single-arch
    # runs may still want to set issue_run_id for dashboard grouping).
    assert doc["sibling_runs"] == []


def test_link_siblings_preserves_other_fields(tmp_path):
    """Regression: link-siblings must not overwrite unrelated run.json state."""
    _run(
        tmp_path,
        "init",
        "--run-id",
        "r_bh",
        "--kernel",
        "issue_1089",
        "--arch",
        "blackhole",
        "--first-step",
        "analyzer",
        "--first-message",
        "go",
    )
    # Advance so step_history has a closed entry; link-siblings must not reset it.
    _run(
        tmp_path,
        "advance",
        "--new-step",
        "planner",
        "--new-message",
        "planning",
        "--prev-result",
        "success",
    )
    _run(
        tmp_path,
        "link-siblings",
        "--siblings",
        json.dumps([{"arch": "wormhole", "run_id": "r_wh"}]),
    )
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc["current_step"] == "planner"
    assert len(doc["step_history"]) == 2
    assert doc["step_history"][0]["result"] == "success"
    assert doc["sibling_runs"] == [{"arch": "wormhole", "run_id": "r_wh"}]


@pytest.mark.parametrize(
    "container,explicit,forwarded,local_port,expected",
    [
        (False, "", "", "", "tcp://runner-special-1:5555 5555"),
        (False, "", "", "6000", "tcp://runner-special-1:6000 6000"),
        (True, "", "54910", "", "tcp://runner:54910 5555"),
        (True, "tcp://override:6001", "invalid", "6000", "tcp://override:6001 6000"),
        (True, "", "invalid", "", None),
    ],
)
def test_nng_callback_matches_bind_or_forwarded_port(
    tmp_path, container, explicit, forwarded, local_port, expected
):
    helper = SCRIPT.with_name("nng_channel.sh").read_text()
    marker = tmp_path / "dockerenv"
    if container:
        marker.touch()
    helper = helper.replace("/.dockerenv", str(marker))
    proc = subprocess.run(
        [
            "bash",
            "-c",
            helper + "\nhostname() { return 127; }\n"
            "uname() { echo runner-special-1; }\n"
            '_resolve_nng_channel || exit $?\nprintf "%s %s" "$NNG_ADDR" "$NNG_LOCAL"',
        ],
        env={
            **os.environ,
            "NNG_SOCKET_ADDR": explicit,
            "P_USER_DBD_PORT": forwarded,
            "NNG_SOCKET_LOCAL_PORT": local_port,
        },
        capture_output=True,
        text=True,
    )
    if expected is None:
        assert proc.returncode == 3
        assert "valid P_USER_DBD_PORT" in proc.stderr
    else:
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == expected


def test_qsr_wrapper_reaps_previous_lock_owner_and_only_its_failed_job(tmp_path):
    scripts = tmp_path / "llk" / ".claude" / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / "run_qsr_metal_test.sh"
    wrapper.write_text(RUN_TEST.with_name(wrapper.name).read_text())
    reap = scripts.parents[1] / "codegen" / "scripts" / "reap_stale_emu.sh"
    reap.parent.mkdir(parents=True)
    (reap.parent / "nng_channel.sh").write_text(
        SCRIPT.with_name("nng_channel.sh").read_text()
    )
    reap.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$REAP_LOG"\n')
    reap.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    hostname = fake_bin / "hostname"
    hostname.write_text('#!/bin/bash\necho "$TEST_HOST"\n')
    hostname.chmod(0o755)
    log = tmp_path / "reaped.txt"
    lock = tmp_path / "aether.lock"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "QSR_AETHER_LOCK": str(lock),
        "QSR_AETHER_LOCK_SCOPE": "host",
        "NNG_SOCKET_ADDR": "tcp://callback:54910",
        "NNG_SOCKET_NAME": "",
        "QSR_SIM_BACKEND": "emu",
        "QSR_EMU_SIM_PATH": str(tmp_path),
        "EMU_HOST": "remote",
        "REAP_LOG": str(log),
    }

    def run(host, command="true"):
        proc = subprocess.run(
            [
                "bash",
                str(wrapper),
                "--tt-metal-home",
                str(tmp_path),
                "--cache",
                str(tmp_path / "cache"),
                "--",
                command,
            ],
            env={**env, "TEST_HOST": host},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode == (0 if command == "true" else 1), proc.stderr
        return Path(f"{lock}.{host}").read_text().split()[1]

    # Seed the identity left by a hard-killed predecessor on this reservation.
    Path(f"{lock}.runner-a").write_text("old-remote previous-pid-tag\n")
    first = run("runner-a")
    assert "--emu-host old-remote" in log.read_text()
    assert "--tag previous-pid-tag --force" in log.read_text()
    lines = log.read_text().splitlines()
    peer = run("runner-b")
    assert log.read_text().splitlines() == lines
    second = run("runner-a", "false")
    lines = log.read_text().splitlines()
    assert len(lines) == 3
    assert f"--tag {first} --force" in lines[1]
    assert f"--tag {second} --force" in lines[2]
    assert peer not in log.read_text()


@pytest.mark.parametrize(
    "node, k_filter, summary",
    [
        ("", "", "4 tests collected"),
        ("::test_exact", "", "2 tests collected"),
        ("", "keep", "2/4 tests collected"),
        ("::test_exact", "keep", "1/2 tests collected"),
    ],
)
def test_local_pytest_target_preserves_node_and_filter(
    tmp_path, node, k_filter, summary
):
    import shlex

    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_probe.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', ['keep', 'drop'])\n"
        "def test_exact(value): pass\n"
        "@pytest.mark.parametrize('value', ['keep', 'drop'])\n"
        "def test_other(value): pass\n"
    )
    source = RUN_TEST.read_text()
    target_function = (
        source[source.index("_build_target() {") :].split("\n}", 1)[0] + "\n}"
    )
    script = (
        target_function
        + "\n_build_target\n"
        + shlex.quote(sys.executable)
        + ' -m pytest --collect-only --verbosity=-1 "${TARGET[@]}"'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TEST_FILE": "test_probe.py",
            "TEST_ID": "test_probe.py" + node if node else "",
            "K_FILTER": k_filter,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": "",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert summary in proc.stdout


@pytest.fixture
def reviewed_candidate(tmp_path):
    wt = tmp_path / "candidate"
    wt.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(wt), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.com")
    (wt / "kernel.h").write_text("base\n")
    git("add", "-A")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (wt / "kernel.h").write_text("fix\n")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "attempt_id": "queue-1",
                "required_verification": {"attempt_id": "attempt-001"},
            }
        )
    )

    def review(action):
        return _run(
            logs,
            "review",
            "--action",
            action,
            "--worktree",
            str(wt),
            "--expected-base-sha",
            base,
        )

    review("prepare")
    result = {
        "identity": json.loads((logs / "review_context.json").read_text()),
        "reviewed": True,
        "findings": [],
        "findings_total": 0,
        "blocking_total": 0,
        "verdict": "clean",
        "requirements_complete": True,
        "unresolved": [],
        "skills_used": [],
    }
    (logs / "review_result.json").write_text(json.dumps(result))
    return wt, logs, git, review, result


@pytest.mark.parametrize(
    "mutation", ["staged", "untracked", "committed", "manifest", "run"]
)
def test_review_rejects_stale_candidate(reviewed_candidate, mutation):
    wt, logs, git, review, result = reviewed_candidate
    review("record")
    if mutation in {"staged", "untracked", "committed"}:
        (wt / "new.h").write_text("new fix\n")
        if mutation != "untracked":
            git("add", "-A")
        if mutation == "committed":
            git("commit", "-qm", "changed")
    else:
        run = json.loads((logs / "run.json").read_text())
        if mutation == "manifest":
            run["required_verification"]["attempt_id"] = "attempt-002"
        else:
            run["run_id"] = "different-run"
        (logs / "run.json").write_text(json.dumps(run))
    with pytest.raises(subprocess.CalledProcessError):
        review("check")


@pytest.mark.parametrize(
    "patch",
    [
        {"reviewed": False},
        {"blocking_total": False},
        {"findings_total": 1},
        {"identity": {}},
        {"verdict": "changes_requested"},
        {"findings": [{}]},
        {"requirements_complete": 1},
        {"unresolved": [None]},
    ],
)
def test_review_rejects_malformed_success(reviewed_candidate, patch):
    wt, logs, git, review, result = reviewed_candidate
    (logs / "review_result.json").write_text(json.dumps({**result, **patch}))
    with pytest.raises(subprocess.CalledProcessError):
        review("record")


def test_review_preserves_index_and_accepts_packaging_commit(reviewed_candidate):
    wt, logs, git, review, result = reviewed_candidate
    index = (wt / ".git/index").read_bytes()
    review("record")
    assert (wt / ".git/index").read_bytes() == index
    git("add", "-A")
    git("commit", "-qm", "package")
    review("check")
    assert len(list((logs / "reviews").glob("*.json"))) == 1
    review("prepare")
    assert not (logs / "review_result.json").exists()


def test_review_abbreviated_legacy_identity_requires_reprepare(reviewed_candidate):
    wt, logs, git, review, result = reviewed_candidate
    context = result["identity"]
    canonical_digest = context["patch_sha256"]
    git("config", "core.abbrev", "7")
    legacy_patch = subprocess.check_output(
        ["git", "-C", str(wt), "diff", "--binary", context["base_commit"], "--"]
    )
    context["patch_sha256"] = hashlib.sha256(legacy_patch).hexdigest()
    assert context["patch_sha256"] != canonical_digest
    (logs / "review_context.json").write_text(json.dumps(context))
    (logs / "review_result.json").write_text(json.dumps(result))
    for action in ("validate", "record", "check"):
        with pytest.raises(subprocess.CalledProcessError) as error:
            review(action)
        assert "review context is stale" in error.value.stderr
    review("prepare")
    assert not (logs / "review_result.json").exists()
    assert (
        json.loads((logs / "review_context.json").read_text())["patch_sha256"]
        == canonical_digest
    )


@pytest.mark.parametrize(
    "patch",
    [{"requirements_complete": None}, {"unresolved": ["Need SRCB state evidence"]}],
)
def test_review_records_pending_evidence_but_cannot_succeed(reviewed_candidate, patch):
    wt, logs, git, review, result = reviewed_candidate
    (logs / "review_result.json").write_text(json.dumps({**result, **patch}))
    review("record")
    with pytest.raises(subprocess.CalledProcessError):
        review("check")


@pytest.mark.parametrize("blocking", [False, True])
def test_review_validate_accepts_pending_review_without_mutation(
    reviewed_candidate, blocking
):
    wt, logs, git, review, result = reviewed_candidate
    # Preserve an existing accepted record/archive as well as the new handoff.
    review("record")
    if blocking:
        result.update(
            findings=[
                {
                    "severity": "correctness",
                    "blocking": True,
                    "file": "kernel.h",
                    "line": "1",
                    "title": "Unsupported architecture",
                    "comment": "Gate the new pool on the supported architecture.",
                }
            ],
            findings_total=1,
            blocking_total=1,
            verdict="changes_requested",
        )
    result["unresolved"] = ["Need the Blackhole dependency-stall rule"]
    (logs / "review_result.json").write_text(json.dumps(result))
    snapshot = {
        p.relative_to(logs): p.read_bytes() for p in logs.rglob("*") if p.is_file()
    }
    index = (wt / ".git/index").read_bytes()
    status = git("status", "--porcelain", "-uall")

    review("validate")

    assert snapshot == {
        p.relative_to(logs): p.read_bytes() for p in logs.rglob("*") if p.is_file()
    }
    assert (wt / ".git/index").read_bytes() == index
    assert git("status", "--porcelain", "-uall") == status
    assert not list((wt / ".git").glob(".candidate-index-*"))
    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero"):
        review("check")


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        ("unresolved_object", "unresolved entries must name the missing evidence"),
        ("identity", "review result is missing the current review identity"),
        ("candidate", "review context is stale"),
    ],
)
def test_review_validate_rejects_bad_handoff_without_mutation(
    reviewed_candidate, mutation, expected_error
):
    wt, logs, git, review, result = reviewed_candidate
    if mutation == "unresolved_object":
        result["unresolved"] = [{"item": "SFPMUL", "evidence_needed": "ISA rule"}]
    elif mutation == "identity":
        result["identity"]["run_id"] = "another-run"
    else:
        (wt / "kernel.h").write_text("another fix\n")
    (logs / "review_result.json").write_text(json.dumps(result))
    snapshot = {
        p.relative_to(logs): p.read_bytes() for p in logs.rglob("*") if p.is_file()
    }
    index = (wt / ".git/index").read_bytes()

    with pytest.raises(subprocess.CalledProcessError) as error:
        review("validate")

    assert expected_error in error.value.stderr
    assert snapshot == {
        p.relative_to(logs): p.read_bytes() for p in logs.rglob("*") if p.is_file()
    }
    assert (wt / ".git/index").read_bytes() == index


@pytest.mark.parametrize("ending", ["exit 0", "exit 1", "sleep 30"])
def test_autodebug_launcher_is_bounded_and_archives_report(
    tmp_path, monkeypatch, ending
):
    wt = tmp_path / "wt"
    wt.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    package = tmp_path / "plugin"
    launcher = package / "skills/autodebug/scripts/autodebug.sh"
    launcher.parent.mkdir(parents=True)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_claude = fake_bin / "claude"
    fake_claude.write_text("#!/bin/sh\nexit 97\n")
    fake_claude.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.delenv("CODEGEN_AUTODEBUG_BUDGET_USD", raising=False)
    launcher.write_text(
        'test -z "${CLAUDECODE:-}" || exit 9\nprintf "diagnosis" > AUTODEBUG.md\n'
        + ending
        + "\n"
    )
    (logs / "run.json").write_text(
        json.dumps(
            {
                "run_id": "bounded-launcher",
                "solver_plugins": {"tt-autodebug": {"path": str(package)}},
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_UTILS),
            "autodebug",
            "--worktree",
            str(wt),
            "--log-dir",
            str(logs),
            "--problem",
            "test failure",
            "--timeout",
            "1",
        ],
        env={**os.environ, "CLAUDECODE": "1"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) == (ending == "exit 0")
    assert not (wt / "AUTODEBUG.md").exists()
    assert [p.read_text() for p in logs.glob("autodebug-*/AUTODEBUG.md")] == [
        "diagnosis"
    ]
    (wt / "AUTODEBUG.md").write_text("unrelated")
    refused = subprocess.run(
        [
            sys.executable,
            str(RUN_UTILS),
            "autodebug",
            "--worktree",
            str(wt),
            "--log-dir",
            str(logs),
            "--problem",
            "failure",
        ],
        capture_output=True,
    )
    assert refused.returncode != 0
    assert (wt / "AUTODEBUG.md").read_text() == "unrelated"


@pytest.fixture
def autodebug_sandbox(tmp_path, monkeypatch):
    """Execute the real launcher/shim against a fake Claude executable only."""
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location("isolated_run_utils", RUN_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logs, worktree, fake_bin = (tmp_path / n for n in ("logs", "worktree", "fake-bin"))
    for p in (logs, worktree, fake_bin):
        p.mkdir()
    package = tmp_path / "plugin"
    launcher = package / "skills/autodebug/scripts/autodebug.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        'test -z "${CLAUDECODE:-}" || exit 9\nclaude --print "diagnose"\n'
    )
    fake_claude = fake_bin / "claude"
    fake_claude.write_text(
        "#!" + sys.executable + "\nimport json,os,sys,time\n"
        "from pathlib import Path\n"
        'registry=json.loads(Path(os.environ["FAKE_REGISTRY"]).read_text())\n'
        'sid=sys.argv[sys.argv.index("--session-id")+1]\n'
        'assert any(row["session_id"]==sid for row in registry["sessions"])\n'
        'Path("claude-argv.json").write_text(json.dumps(sys.argv[1:]))\n'
        'Path("AUTODEBUG.md").write_text("partial diagnosis")\n'
        'time.sleep(float(os.environ.get("FAKE_CLAUDE_SLEEP", "0")))\n'
    )
    fake_claude.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("FAKE_REGISTRY", str(logs / "session_registry.json"))
    monkeypatch.delenv("CODEGEN_AUTODEBUG_BUDGET_USD", raising=False)
    run = {
        "run_id": "isolated-run",
        "solver_plugins": {"tt-autodebug": {"path": str(package)}},
    }
    (logs / "run.json").write_text(json.dumps(run))
    state = {
        "RUN_ID": "isolated-run",
        "SESSION_ID": "00000000-0000-4000-8000-000000000001",
    }
    (logs / "state.json").write_text(json.dumps(state))
    exports = []

    def fake_export(argv, **kwargs):
        assert Path(argv[1]).name == "extract_run_transcripts.py"
        exports.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_export)
    args = argparse.Namespace(
        log_dir=str(logs), worktree=str(worktree), problem="exact failure", timeout=2
    )
    return module, args, logs, worktree, exports


def test_autodebug_pins_child_identity_budget_and_exports(
    autodebug_sandbox, monkeypatch
):
    import uuid

    module, args, logs, worktree, exports = autodebug_sandbox
    monkeypatch.setenv("CODEGEN_AUTODEBUG_BUDGET_USD", "12.50")
    module.cmd_autodebug(args)
    registry = json.loads((logs / "session_registry.json").read_text())
    assert registry["run_id"] == "isolated-run"
    assert len(registry["sessions"]) == 1
    child = registry["sessions"][0]
    assert str(uuid.UUID(child["session_id"])) == child["session_id"]
    assert child["parent_session_id"] == "00000000-0000-4000-8000-000000000001"
    assert child["project_cwd"] == str(worktree)
    assert child["allocated_budget_usd"] == 12.5
    argv = json.loads((worktree / "claude-argv.json").read_text())
    assert argv[:4] == ["--session-id", child["session_id"], "--max-budget-usd", "12.5"]
    assert len(exports) == 1 and exports[0][-4:] == [
        "--session-id",
        child["session_id"],
        "--project-cwd",
        str(worktree),
    ]
    assert (
        Path(child["artifact_dir"]) / "AUTODEBUG.md"
    ).read_text() == "partial diagnosis"
    assert not (worktree / "AUTODEBUG.md").exists()


def test_autodebug_timeout_retains_registry_report_and_exports(
    autodebug_sandbox, monkeypatch
):
    module, args, logs, worktree, exports = autodebug_sandbox
    args.timeout = 0.3
    monkeypatch.setenv("FAKE_CLAUDE_SLEEP", "30")
    with pytest.raises(subprocess.TimeoutExpired):
        module.cmd_autodebug(args)
    child = json.loads((logs / "session_registry.json").read_text())["sessions"][0]
    assert (
        Path(child["artifact_dir"]) / "AUTODEBUG.md"
    ).read_text() == "partial diagnosis"
    assert len(exports) == 1
    assert child["session_id"] in exports[0]
    assert not (worktree / "AUTODEBUG.md").exists()


def test_autodebug_export_failure_preserves_success(autodebug_sandbox, monkeypatch):
    module, args, logs, worktree, _ = autodebug_sandbox

    def export_failure(*a, **kw):
        raise subprocess.TimeoutExpired("exporter", 30)

    monkeypatch.setattr(module.subprocess, "run", export_failure)
    module.cmd_autodebug(args)
    assert any(logs.glob("autodebug-*/AUTODEBUG.md"))
    assert (
        "Transcript export unavailable"
        in next(logs.glob("autodebug-*/launcher.log")).read_text()
    )


@pytest.mark.parametrize("cap", ["0", "0.009"])
def test_autodebug_exhausted_budget_never_launches(autodebug_sandbox, monkeypatch, cap):
    module, args, logs, worktree, exports = autodebug_sandbox
    monkeypatch.setenv("CODEGEN_AUTODEBUG_BUDGET_USD", cap)
    with pytest.raises(SystemExit, match="exhausted"):
        module.cmd_autodebug(args)
    assert not (worktree / "claude-argv.json").exists()
    assert not exports
    assert not (logs / "session_registry.json").exists()


def test_autodebug_allocated_budget_cannot_be_reused(autodebug_sandbox, monkeypatch):
    module, args, logs, worktree, exports = autodebug_sandbox
    monkeypatch.setenv("CODEGEN_AUTODEBUG_BUDGET_USD", "2")
    module.cmd_autodebug(args)
    with pytest.raises(SystemExit, match="exhausted"):
        module.cmd_autodebug(args)
    assert (
        len(json.loads((logs / "session_registry.json").read_text())["sessions"]) == 1
    )
    assert len(exports) == 1


def test_autodebug_rejects_foreign_registry_before_launch(autodebug_sandbox):
    module, args, logs, worktree, exports = autodebug_sandbox
    original = {
        "schema": "issue-solver.session-registry",
        "version": 1,
        "run_id": "other",
        "sessions": [],
    }
    (logs / "session_registry.json").write_text(json.dumps(original))
    with pytest.raises(SystemExit, match="another run"):
        module.cmd_autodebug(args)
    assert json.loads((logs / "session_registry.json").read_text()) == original
    assert not (worktree / "claude-argv.json").exists()


def test_autodebug_rejects_foreign_state_before_launch(autodebug_sandbox):
    module, args, logs, worktree, exports = autodebug_sandbox
    (logs / "state.json").write_text(json.dumps({"RUN_ID": "another-run"}))
    with pytest.raises(SystemExit, match="run"):
        module.cmd_autodebug(args)
    assert not (worktree / "claude-argv.json").exists()


def test_autodebug_rejects_duplicate_child_identity(autodebug_sandbox, monkeypatch):
    import uuid

    module, args, logs, worktree, exports = autodebug_sandbox
    child_id = "00000000-0000-4000-8000-000000000002"
    monkeypatch.setattr(module.uuid, "uuid4", lambda: uuid.UUID(child_id))
    original = {
        "schema": "issue-solver.session-registry",
        "version": 1,
        "run_id": "isolated-run",
        "sessions": [{"session_id": child_id, "kind": "autodebug"}],
    }
    (logs / "session_registry.json").write_text(json.dumps(original))
    with pytest.raises(SystemExit, match="session|duplicate"):
        module.cmd_autodebug(args)
    assert json.loads((logs / "session_registry.json").read_text()) == original
    assert not (worktree / "claude-argv.json").exists()


def test_transcript_export_explicit_missing_session_never_discovers_another(
    tmp_path, monkeypatch
):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "isolated_transcript_export", RUN_UTILS.with_name("extract_run_transcripts.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_find_by_session_id", lambda sid: None)

    def forbidden_discovery(pid):
        pytest.fail(
            "explicit session identity must never fall back to a different live session"
        )

    monkeypatch.setattr(module, "_discover_session", forbidden_discovery)
    assert (
        module.run(
            str(tmp_path / "logs"), "00000000-0000-4000-8000-000000000099", None, None
        )
        != 0
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"schema": "another-registry"},
        {"version": 99},
        {"sessions": {}},
        {"sessions": [{"session_id": "duplicate"}, {"session_id": "duplicate"}]},
    ],
)
def test_autodebug_rejects_invalid_registry_without_rewriting(autodebug_sandbox, patch):
    module, args, logs, worktree, exports = autodebug_sandbox
    registry = {
        "schema": "issue-solver.session-registry",
        "version": 1,
        "run_id": "isolated-run",
        "sessions": [],
        **patch,
    }
    original = json.dumps(registry)
    (logs / "session_registry.json").write_text(original)
    with pytest.raises(SystemExit, match="schema|identity"):
        module.cmd_autodebug(args)
    assert (logs / "session_registry.json").read_text() == original
    assert not (worktree / "claude-argv.json").exists()


def test_autodebug_negative_allocation_cannot_expand_budget(
    autodebug_sandbox, monkeypatch
):
    module, args, logs, worktree, exports = autodebug_sandbox
    monkeypatch.setenv("CODEGEN_AUTODEBUG_BUDGET_USD", "2")
    registry = {
        "schema": "issue-solver.session-registry",
        "version": 1,
        "run_id": "isolated-run",
        "sessions": [
            {
                "session_id": "00000000-0000-4000-8000-000000000003",
                "allocated_budget_usd": -10,
            }
        ],
    }
    (logs / "session_registry.json").write_text(json.dumps(registry))
    with pytest.raises(SystemExit, match="budget|allocation"):
        module.cmd_autodebug(args)
    assert not (worktree / "claude-argv.json").exists()


@pytest.mark.parametrize("cap", ["-1", "nan", "inf"])
def test_autodebug_nonfinite_or_negative_budget_rejected(
    autodebug_sandbox, monkeypatch, cap
):
    module, args, logs, worktree, exports = autodebug_sandbox
    monkeypatch.setenv("CODEGEN_AUTODEBUG_BUDGET_USD", cap)
    with pytest.raises(SystemExit, match="finite and nonnegative"):
        module.cmd_autodebug(args)
    assert not (worktree / "claude-argv.json").exists()


def test_autodebug_export_failure_does_not_mask_launcher_timeout(
    autodebug_sandbox, monkeypatch
):
    module, args, logs, worktree, exports = autodebug_sandbox
    args.timeout = 0.3
    monkeypatch.setenv("FAKE_CLAUDE_SLEEP", "30")

    def failed_export(*a, **kw):
        raise OSError("synthetic export failure")

    monkeypatch.setattr(module.subprocess, "run", failed_export)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        module.cmd_autodebug(args)
    assert caught.value.timeout == 0.3
    assert any(logs.glob("autodebug-*/AUTODEBUG.md"))


@pytest.fixture
def issue_bootstrap_sandbox(tmp_path):
    worktree = tmp_path / "source"
    llk = worktree / "tt_metal" / "tt-llk"
    version = llk / "codegen" / "agents" / "issue-solver" / "VERSION"
    version.parent.mkdir(parents=True)
    version.write_text("2.5.0\n")
    for command in (
        ["git", "init", "-q", "-b", "fixture-base", str(worktree)],
        ["git", "-C", str(worktree), "add", "-A"],
        [
            "git",
            "-C",
            str(worktree),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(command, check=True, capture_output=True)
    linked = tmp_path / "worktree"
    subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "worktree",
            "add",
            "-qb",
            "issue-bootstrap",
            str(linked),
        ],
        check=True,
        capture_output=True,
    )
    worktree = linked
    llk = worktree / "tt_metal" / "tt-llk"
    issue = {
        "number": 123,
        "title": "'Quoted' \"title\" $(touch SHOULD_NOT_EXIST)\n\n",
        "body": "Unicode λ, `code`, $HOME, quotes '\"\nsecond line\n\n",
        "labels": [{"name": "bug, regression"}, {"name": "wormhole"}],
        "comments": [
            {
                "id": "IC_123",
                "author": {"login": "someone"},
                "createdAt": "2026-09-20T00:00:00Z",
                "body": "Comment '\"`$()\nverbatim\n\n",
            }
        ],
        "url": "https://github.com/example/project/issues/123",
    }
    snapshot = tmp_path / "issue.json"
    snapshot.write_text(json.dumps(issue, ensure_ascii=False, indent=2) + "\n")
    bootstrap = llk / ".codegen_run_state.json"
    bootstrap.write_text(json.dumps({"QUEUE_ATTEMPT_ID": "preserved-admission"}))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env python3\nimport json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['GH_MARKER']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.stdout.write(pathlib.Path(os.environ['GH_FIXTURE']).read_text())\n"
    )
    fake_gh.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CODEGEN_", "TTSIM_", "CLAUDE_"))
    }
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "HOME": str(tmp_path / "empty-home"),
            "CODEGEN_ISSUE_SNAPSHOT": str(snapshot),
            "CODEGEN_LOGS_ROOT": str(tmp_path / "logs"),
            "LOG_DIR": str(tmp_path / "unrelated-log-dir"),
            "GH_FIXTURE": str(snapshot),
            "GH_MARKER": str(tmp_path / "gh-called"),
        }
    )
    command = [
        sys.executable,
        str(SCRIPT.parent / "load_issue.py"),
        "123",
        "--seed-state",
        "--worktree-dir",
        str(worktree),
        "--worktree-branch",
        "issue-bootstrap",
        "--arches",
        "bh",
        "--test-backend",
        "local",
        "--create-local-branch",
        "yes",
        "--create-pr",
        "no",
    ]
    return worktree, issue, snapshot, bootstrap, env, command


@pytest.mark.parametrize("legacy_labels", [False, True])
def test_issue_bootstrap_snapshot_preserved_through_setup_run(
    issue_bootstrap_sandbox, legacy_labels
):
    worktree, issue, snapshot, bootstrap, env, command = issue_bootstrap_sandbox
    result = subprocess.run(
        command, env=env, text=True, capture_output=True, check=True
    )
    assert json.loads(result.stdout)["target_arches"] == ["blackhole"]
    seeded = json.loads(bootstrap.read_text())
    assert seeded["QUEUE_ATTEMPT_ID"] == "preserved-admission"
    assert seeded["TARGET_ARCH"] == "blackhole"  # Explicit arch beats label.
    assert not Path(env["GH_MARKER"]).exists()
    assert not Path(env["LOG_DIR"]).exists()
    if legacy_labels:
        # Older routers stored only a comma-separated string; retain support.
        seeded.pop("ISSUE_LABELS_JSON")
        seeded["ISSUE_LABELS"] = "bug,wormhole"
        bootstrap.write_text(json.dumps(seeded))
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; execute_step_setup_run && execute_step_write_initial_run_json',
            "bootstrap-test",
            str(ORCHESTRATOR_STEPS),
        ],
        cwd=worktree / "tt_metal" / "tt-llk",
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    logs = Path(json.loads(bootstrap.read_text())["LOG_DIR"])
    final = json.loads((logs / "state.json").read_text())
    for state in (seeded, final):
        assert state["ISSUE_TITLE"] == issue["title"]
        assert state["ISSUE_BODY"] == issue["body"]
        assert json.loads(state["ISSUE_COMMENTS"]) == issue["comments"]
        if not legacy_labels:
            assert state["ISSUE_LABELS_JSON"] == [
                label["name"] for label in issue["labels"]
            ]
    run = json.loads((logs / "run.json").read_text())
    assert run["issue"]["title"] == issue["title"]
    assert run["issue"]["labels"] == (
        ["bug", "wormhole"]
        if legacy_labels
        else [label["name"] for label in issue["labels"]]
    )
    assert final["QUEUE_ATTEMPT_ID"] == "preserved-admission"
    assert not (worktree / "SHOULD_NOT_EXIST").exists()


def test_issue_bootstrap_live_fetch_and_snapshot_print_compatibility(
    issue_bootstrap_sandbox,
):
    worktree, issue, snapshot, bootstrap, env, command = issue_bootstrap_sandbox
    printed = subprocess.run(
        command[:3], env=env, text=True, capture_output=True, check=True
    )
    assert printed.stdout == snapshot.read_text()
    assert not Path(env["GH_MARKER"]).exists()
    del env["CODEGEN_ISSUE_SNAPSHOT"]
    subprocess.run(
        command + ["--repo", "example/project"],
        env=env,
        check=True,
        capture_output=True,
    )
    assert json.loads(Path(env["GH_MARKER"]).read_text()) == [
        "issue",
        "view",
        "123",
        "--json",
        "number,title,body,labels,comments,url",
        "--repo",
        "example/project",
    ]
    assert (
        json.loads(json.loads(bootstrap.read_text())["ISSUE_COMMENTS"])
        == issue["comments"]
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("--arches", "not-an-arch"),
        ("--arches", "[]"),
        ("--arches", '["bh", 2]'),
        ("--test-backend", "silicon"),
        ("--worktree-branch", "different-branch"),
        ("--test-backend", "ttsim"),
    ],
)
def test_issue_bootstrap_invalid_inputs_do_not_mutate(
    issue_bootstrap_sandbox, option, value
):
    worktree, issue, snapshot, bootstrap, env, command = issue_bootstrap_sandbox
    before = bootstrap.read_bytes()
    command[command.index(option) + 1] = value
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert bootstrap.read_bytes() == before
    assert not Path(env["GH_MARKER"]).exists()


@pytest.mark.parametrize("mismatch", ["snapshot", "bootstrap", "bound"])
def test_issue_bootstrap_identity_conflicts_do_not_mutate(
    issue_bootstrap_sandbox, mismatch
):
    worktree, issue, snapshot, bootstrap, env, command = issue_bootstrap_sandbox
    if mismatch == "snapshot":
        issue["number"] = 456
        snapshot.write_text(json.dumps(issue))
    elif mismatch == "bootstrap":
        bootstrap.write_text(json.dumps({"ISSUE_NUMBER": "456"}))
    else:
        bootstrap.write_text(json.dumps({"RUN_ID": "existing-run"}))
    before = bootstrap.read_bytes()
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert bootstrap.read_bytes() == before


def test_issue_bootstrap_multi_simulator_and_no_push(issue_bootstrap_sandbox):
    worktree, issue, snapshot, bootstrap, env, command = issue_bootstrap_sandbox
    bh = snapshot.parent / "libbh.so"
    wh = snapshot.parent / "libwh.so"
    bh.touch()
    wh.touch()
    command[command.index("--arches") + 1] = '["bh", "wh", "bh"]'
    command[command.index("--test-backend") + 1] = "ttsim"
    command[command.index("--create-local-branch") + 1] = "no"
    command[command.index("--create-pr") + 1] = "yes"
    env["CODEGEN_NO_PUSH"] = "1"
    before = bootstrap.read_bytes()
    result = subprocess.run(
        command + ["--ttsim-so-paths", json.dumps({"bh": str(bh)})],
        env=env,
        capture_output=True,
    )
    assert result.returncode != 0
    assert bootstrap.read_bytes() == before
    subprocess.run(
        command + ["--ttsim-so-paths", json.dumps({"bh": str(bh), "wh": str(wh)})],
        env=env,
        check=True,
        capture_output=True,
    )
    state = json.loads(bootstrap.read_text())
    assert state["RUN_MODE"] == "multi"
    assert json.loads(state["TARGET_ARCHES"]) == ["blackhole", "wormhole"]
    assert json.loads(state["TTSIM_SO_PATHS"]) == {
        "blackhole": str(bh),
        "wormhole": str(wh),
    }
    assert state["CREATE_LOCAL_BRANCH"] == "yes"
    assert state["CREATE_PR"] == "no"
    assert "TARGET_ARCH" not in state


def test_issue_bootstrap_rejects_main_checkout(issue_bootstrap_sandbox):
    worktree, issue, snapshot, bootstrap, env, command = issue_bootstrap_sandbox
    main_checkout = snapshot.parent / "source"
    command[command.index("--worktree-dir") + 1] = str(main_checkout)
    command[command.index("--worktree-branch") + 1] = "fixture-base"
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "linked worktree" in result.stderr
    assert not (main_checkout / "tt_metal/tt-llk/.codegen_run_state.json").exists()


def test_setup_lock_reuses_existing_file_with_inherited_noclobber(
    tmp_path, monkeypatch
):
    original_run = subprocess.run
    lock_paths = {}
    sentinel = b"existing shared flock inode; do not truncate\n"

    def run_with_noclobber(argv, *args, **kwargs):
        if (
            isinstance(argv, list)
            and argv[:2] == ["bash", "-c"]
            and "setup_worktree " in argv[2]
        ):
            repo = Path(argv[5])
            lock_path = repo / ".git" / "codegen-worktree-setup.lock"
            if lock_path not in lock_paths:
                lock_path.write_bytes(sentinel)
                lock_paths[lock_path] = lock_path.stat().st_ino
            argv = [argv[0], "-C", *argv[1:]]
        return original_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_with_noclobber)
    # Reuse the real-Git setup/import fixture, including failed resume rejection
    # and a subsequent ordinary setup on the same pre-existing lock file.
    test_setup_worktree_records_exact_base_before_bootstrap(
        tmp_path, "outer_timeout", False, "supervisor-checkpoint.patch"
    )
    assert lock_paths
    assert all(
        path.read_bytes() == sentinel and path.stat().st_ino == inode
        for path, inode in lock_paths.items()
    )


@pytest.mark.parametrize(
    "invalid",
    [None, "unset", "wrong_task", "wrong_branch", "foreign_repo", "locked", "symlink"],
)
def test_cleanup_removes_only_owned_attempt(tmp_path, invalid):
    """Real Git: missing/mismatched identity and locks cannot delete either run."""
    repo = tmp_path / "repo"
    script = repo / "tt_metal/tt-llk/codegen/scripts/setup_worktree.sh"
    script.parent.mkdir(parents=True)
    script.write_bytes(SETUP_WORKTREE.read_bytes())

    def git(*args, cwd=repo):
        return subprocess.run(
            ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.com")
    git("add", "-A")
    git("commit", "-qm", "base")
    root = tmp_path / "worktrees"
    own, other = root / "issue-5-v1", root / "issue-5-v2"
    for wt, version in [(own, 1), (other, 2)]:
        git("worktree", "add", "-b", f"llk_code_gen/issue-5-v{version}", str(wt))
        (wt / "uncommitted.txt").write_text(f"attempt {version}")
    env = dict(
        os.environ,
        CODEGEN_WORKTREE_ROOT=str(root),
        CODEGEN_KEEP_WORKTREE="false",
        WORKTREE_DIR=str(own),
        WORKTREE_BRANCH="llk_code_gen/issue-5-v1",
    )
    task = "issue-5"
    if invalid == "unset":
        env.pop("WORKTREE_DIR")
    elif invalid == "wrong_task":
        task = "issue-6"
    elif invalid == "wrong_branch":
        env["WORKTREE_BRANCH"] = "llk_code_gen/issue-5-v2"
    elif invalid == "locked":
        git("worktree", "lock", str(own))
    elif invalid == "symlink":
        link = root / "alias"
        link.symlink_to(own, target_is_directory=True)
        env["WORKTREE_DIR"] = str(link)
    elif invalid == "foreign_repo":
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        git("init", "-q", cwd=foreign)
        git("config", "user.name", "test", cwd=foreign)
        git("config", "user.email", "test@example.com", cwd=foreign)
        git("commit", "--allow-empty", "-qm", "foreign", cwd=foreign)
        foreign_wt = root / "issue-5-v3"
        git(
            "worktree",
            "add",
            "-b",
            "llk_code_gen/issue-5-v3",
            str(foreign_wt),
            cwd=foreign,
        )
        env.update(
            WORKTREE_DIR=str(foreign_wt), WORKTREE_BRANCH="llk_code_gen/issue-5-v3"
        )
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; cleanup_worktree "$2"', "_", str(script), task],
        env=env,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is (invalid is None), result.stdout + result.stderr
    assert own.exists() is (invalid is not None)
    assert (other / "uncommitted.txt").read_text() == "attempt 2"
    assert git("rev-parse", "--verify", "llk_code_gen/issue-5-v1")
    if invalid == "foreign_repo":
        assert foreign_wt.exists()


def _measurement_plan():
    contract = {
        "primary_metric": "mean(L1_TO_L1)",
        "marker": "TILE_LOOP",
        "normalization": "loop_factor*tile_cnt",
        "variants": [
            {
                "mathop": "copy",
                "marker": "TILE_LOOP",
                "loop_factor": "16",
                "tile_cnt": "8",
            }
        ],
    }
    analysis = """## Scope
arch_scope:
  blackhole: in_scope
perf_intent: measure
## Verification
fix_layer: llk_lib
verification_required: yes
verifiable_in_llk_suite: yes
llk_coverage: existing
"""
    plan = (
        """## Test Strategy
reproduction_tests:
- arch: blackhole
  test: test_reduce.py
regression_tests:
- arch: blackhole
  test: perf_reduce.py
  required_measurements: ["cycle_measurement"]
  measurement_contract: """
        + json.dumps(contract)
        + "\n"
    )
    return analysis, plan, contract


def test_measurement_intent_seals_v2_and_cannot_reinterpret_v1(tmp_path):
    analysis, plan, contract = _measurement_plan()
    _, output = _required_manifest(tmp_path, analysis, plan)
    manifest = json.loads(output.read_text())
    assert manifest["version"] == 2
    assert manifest["requirements"][-1]["measurement_contract"] == contract
    manifest["version"] = 1
    manifest["manifest_id"] = _content_id(manifest, {"manifest_id"})
    output.write_text(json.dumps(manifest))
    with pytest.raises(subprocess.CalledProcessError):
        _reduce(tmp_path, output)


@pytest.mark.parametrize("intent", ["maintain", "optimize", ""])
def test_measurement_intent_cannot_be_inferred_or_replace_comparison(tmp_path, intent):
    analysis, plan, _ = _measurement_plan()
    analysis = analysis.replace("perf_intent: measure", f"perf_intent: {intent}")
    proc, _ = _required_manifest(tmp_path, analysis, plan, check=False)
    assert proc.returncode != 0
    assert "predeclared perf_intent" in proc.stderr


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "wrong_variant",
        "normalization",
        "missing_raw",
        "modified_artifact",
        "outside_run",
        "wrong_goal",
        "fake_flag",
        "rehashed_forgery",
        "wrong_job",
        "raw_swapped",
    ],
)
def test_measurement_reducer_rechecks_exact_artifacts_not_model_flags(tmp_path, defect):
    _, _, contract = _measurement_plan()
    requirement = _requirement(
        suite="perf",
        selector={"test": "perf_reduce.py", "test_id": None, "k": None},
        required_measurements=["cycle_measurement"],
        measurement_contract=contract,
    )
    manifest, path = _reducer_manifest(tmp_path, [requirement])
    manifest["version"] = 2
    manifest["manifest_id"] = _content_id(manifest, {"manifest_id"})
    path.write_text(json.dumps(manifest))
    results = tmp_path / "verification-results"
    results.mkdir()
    receipt = _sealed_result(manifest, requirement)
    (results / "perf.json").write_text(json.dumps(receipt))
    header = "mathop,marker,loop_factor,tile_cnt,mean(L1_TO_L1)\n"
    current = tmp_path / "current.post.csv"
    raw = tmp_path / "current.csv"
    current.write_text(header + "copy,TILE_LOOP,16,8,2.5\n")
    raw.write_text(header + "copy,TILE_LOOP,16,8,320\n")
    receipt["version"] = 4
    receipt["measurement_artifacts"] = {
        name: {
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "size": artifact.stat().st_size,
        }
        for name, artifact in (("current", current), ("raw_current", raw))
    }
    receipt["result_id"] = _content_id(receipt, {"result_id"})
    (results / "perf.json").write_text(json.dumps(receipt))
    output = tmp_path / "perf_result.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT.parent / "perf_eval.py"),
            "--goal",
            "measure",
            "--current",
            str(current),
            "--raw-current",
            str(raw),
            "--required-manifest",
            str(path),
            "--requirement-id",
            requirement["requirement_id"],
            "--verification-result",
            str(results / "perf.json"),
            "--json-out",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    perf = json.loads(output.read_text())
    assert perf["verdict"] == "measured"
    assert "delta_pct_median" not in perf and "baseline_source" not in perf
    perf.update(outcome="PERF_OK", patch_sha256=receipt["provenance"]["patch_sha256"])
    if defect == "wrong_variant":
        current.write_text(header + "other,TILE_LOOP,16,8,2.5\n")
        perf["current_sha256"] = hashlib.sha256(current.read_bytes()).hexdigest()
    elif defect == "normalization":
        current.write_text(header + "copy,TILE_LOOP,16,8,320\n")
        perf["current_sha256"] = hashlib.sha256(current.read_bytes()).hexdigest()
    elif defect == "missing_raw":
        raw.unlink()
    elif defect == "modified_artifact":
        current.write_text(header + "copy,TILE_LOOP,16,8,3\n")
    elif defect == "outside_run":
        perf["current_source"] = str(tmp_path.parent / "outside.csv")
    elif defect == "wrong_goal":
        perf["goal"] = "improve"
    elif defect == "fake_flag":
        perf["measurements"]["cycle_measurement"]["measured"] = False
    elif defect == "rehashed_forgery":
        current.write_text(header + "copy,TILE_LOOP,16,8,5\n")
        raw.write_text(header + "copy,TILE_LOOP,16,8,640\n")
        for prefix, artifact in (("current", current), ("raw_current", raw)):
            perf[f"{prefix}_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    elif defect == "wrong_job":
        perf["job_id"] = "different-hardware-job"
    elif defect == "raw_swapped":
        perf["current_source"], perf["raw_current_source"] = (
            perf["raw_current_source"],
            perf["current_source"],
        )
        perf["current_sha256"], perf["raw_current_sha256"] = (
            perf["raw_current_sha256"],
            perf["current_sha256"],
        )
    output.write_text(json.dumps(perf))
    _reduce(tmp_path, path, perf_result=output)
    reduced = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduced["classification"] == (
        "success" if defect is None else "coverage_error"
    )
    assert bool(reduced["success_token"]) == (defect is None)


def test_sealed_comparison_cannot_be_downgraded_to_measurement(tmp_path):
    analysis, plan, _ = _measurement_plan()
    comparison_plan = "\n".join(
        line
        for line in plan.replace(
            '"cycle_measurement"', '"cycle_comparison"'
        ).splitlines()
        if "measurement_contract:" not in line
    )
    _required_manifest(
        tmp_path,
        analysis.replace("perf_intent: measure", "perf_intent: optimize"),
        comparison_plan,
    )
    proc, _ = _required_manifest(
        tmp_path, analysis, plan, "--supersedes-reason", "no baseline", check=False
    )
    assert proc.returncode != 0
    assert "cannot downgrade a sealed comparison" in proc.stderr


@pytest.mark.parametrize("change", ["variant", "remove", "selector"])
def test_measurement_contract_cannot_shrink_or_change_after_execution(tmp_path, change):
    analysis, plan, _ = _measurement_plan()
    _required_manifest(tmp_path, analysis, plan)
    if change == "variant":
        plan = plan.replace('"tile_cnt": "8"', '"tile_cnt": "4"')
    elif change == "remove":
        plan = plan[: plan.index("regression_tests:")]
    else:
        plan = plan.replace("test: perf_reduce.py", "test: perf_reduce.py::test_reduce")
    proc, _ = _required_manifest(
        tmp_path, analysis, plan, "--supersedes-reason", "observed output", check=False
    )
    assert proc.returncode != 0
    assert "cannot remove or change a predeclared measurement contract" in proc.stderr


def _host_worktree(tmp_path, body):
    """Real pytest + real Git; no model, device, compiler, or mocked outcomes."""
    import shutil

    tree = tmp_path / "worktree/tt_metal/tt-llk"
    tests = tree / "tests/python_tests"
    tests.mkdir(parents=True)
    (tests / "conftest.py").write_bytes(LLK_CONFTEST.read_bytes())
    (tests / "test_host.py").write_text(body)
    (tests / "test_device.py").write_text("def test_device(): pass\n")
    scripts = tree / "codegen/scripts"
    scripts.mkdir(parents=True)
    shutil.copy(SCRIPT, scripts / SCRIPT.name)
    shutil.copy(SCRIPT.parent / "nng_channel.sh", scripts / "nng_channel.sh")
    wrapper = tree / ".claude/scripts/run_test.sh"
    wrapper.parent.mkdir(parents=True)
    shutil.copy(RUN_TEST, wrapper)
    (tree / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    for args in (
        ["init", "-q"],
        ["config", "user.email", "host@test.invalid"],
        ["config", "user.name", "Host Test"],
        ["add", "-A"],
        ["commit", "-qm", "base"],
    ):
        subprocess.run(["git", "-C", str(tree.parents[1]), *args], check=True)
    base = subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()
    return tree, wrapper, base


def _host_run(tmp_path, body, *, extra=(), env_extra=None, prepare=None):
    tree, wrapper, base = _host_worktree(tmp_path, body)
    if prepare is not None:
        prepare(tree)
    result = tmp_path / "result.json"
    env = {
        **os.environ,
        "TT_LLK_LOCAL_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "CODEGEN_BASE_COMMIT": base,
        "CODEGEN_RUN_ID": "host-run",
        "CODEGEN_ATTEMPT_ID": "attempt-001",
        "CODEGEN_HOST_TIMEOUT_SECS": "10",
        **(env_extra or {}),
    }
    for name in (
        "CODEGEN_REQUIRED_VERIFICATION_MANIFEST",
        "CODEGEN_VERIFICATION_BACKEND",
        "CODEGEN_PATCH_SHA256",
        "CODEGEN_VERIFICATION_SUITE",
    ):
        env.pop(name, None)
    proc = subprocess.run(
        [
            "bash",
            str(wrapper),
            "host",
            "--worktree",
            str(tree),
            "--arch",
            "blackhole",
            "--test",
            "test_host.py",
            "--result-json-out",
            str(result),
            *extra,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc, json.loads(result.read_text()) if result.exists() else None, tree


def test_host_wrapper_real_pytest_no_device_import_or_elf(tmp_path):
    proc, result, tree = _host_run(
        tmp_path,
        """import sys, pytest
pytestmark = pytest.mark.llk_host
@pytest.mark.parametrize("value", [1, 2, 3])
def test_schema(value):
    assert value > 0
    assert "ttexalens" not in sys.modules
    assert "helpers.test_config" not in sys.modules
""",
    )
    assert proc.returncode == 0, proc.stderr
    assert result["version"] == 3 and result["backend"] == "host"
    assert result["execution"]["passed"] == 3
    assert result["classification"] == "success"
    assert (
        result["provenance"]["selected_nodeids"]
        == result["provenance"]["observed_nodeids"]
    )
    assert "artifact_set_sha256" not in result["provenance"]
    assert not (tree / "tests/sfpi").exists()
    assert not list(tmp_path.rglob("*.elf"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("host_writer", SCRIPT)
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    assert writer._load_verification_result(tmp_path / "result.json") == result


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            "import pytest\npytestmark=pytest.mark.llk_host\ndef test_bad(): assert False\n",
            "candidate_failure",
        ),
        (
            'import pytest\npytestmark=pytest.mark.llk_host\n@pytest.mark.skip(reason="no")\ndef test_skip(): pass\n',
            "coverage_error",
        ),
        ("def test_unmarked(): pass\n", "infra_error"),
        (
            'import pytest\npytestmark=pytest.mark.llk_host\n@pytest.fixture\ndef device(): raise AssertionError("DEVICE SETUP RAN")\n@pytest.fixture\ndef hidden(device): return device\ndef test_bad(hidden): pass\n',
            "infra_error",
        ),
        ("import missing_host_dependency\n", "infra_error"),
    ],
)
def test_host_wrapper_fails_closed_with_structured_evidence(tmp_path, body, expected):
    proc, result, _ = _host_run(tmp_path, body)
    assert proc.returncode != 0, proc.stderr
    assert result is not None, proc.stderr
    assert result["classification"] == expected
    assert "DEVICE SETUP RAN" not in proc.stderr


def test_host_wrapper_zero_selected_and_timeout(tmp_path):
    zero = tmp_path / "zero"
    zero.mkdir()
    proc, result, _ = _host_run(
        zero,
        "import pytest\npytestmark=pytest.mark.llk_host\ndef test_one(): pass\n",
        extra=("--k", "missing"),
    )
    assert proc.returncode != 0 and result["classification"] != "success"
    timed = tmp_path / "timed"
    timed.mkdir()
    proc, result, _ = _host_run(
        timed,
        "import time,pytest\npytestmark=pytest.mark.llk_host\ndef test_slow(): time.sleep(10)\n",
        env_extra={"CODEGEN_HOST_TIMEOUT_SECS": "1", "GRACE_SECS": "1"},
    )
    assert proc.returncode == 5, proc.stderr
    assert result["classification"] == "timed_out"


def test_host_wrapper_rejects_source_mutation(tmp_path):
    proc, result, _ = _host_run(
        tmp_path,
        'import pytest\nfrom pathlib import Path\npytestmark=pytest.mark.llk_host\ndef test_mutation(): Path(__file__).write_text("# changed\\n")\n',
    )
    assert proc.returncode == 3, proc.stderr
    assert result["classification"] == "infra_error"
    assert "host_inputs_mutated_during_execution" in result["reason_codes"]


def test_sealed_host_and_silicon_require_independent_evidence(tmp_path):
    tree, wrapper, base = _host_worktree(
        tmp_path,
        "import pytest\npytestmark=pytest.mark.llk_host\ndef test_schema(): pass\n",
    )
    analysis = tmp_path / "analysis.md"
    analysis.write_text(
        "## Verification\nverification_required: yes\nverifiable_in_llk_suite: yes\nllk_coverage: existing\n"
    )
    plan = tmp_path / "plan.md"
    plan.write_text(
        "## Test Strategy\nreproduction_tests:\n- arch: blackhole\n  test: test_host.py\n  execution: host\nregression_tests:\n- arch: blackhole\n  test: test_device.py\n"
    )
    manifest_path = tmp_path / "required_verification_manifest.json"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "required-verification",
        "--log-dir",
        str(tmp_path),
        "--analysis",
        str(analysis),
        "--plan",
        str(plan),
        "--worktree",
        str(tree.parents[1]),
        "--run-id",
        "host-run",
        "--expected-base-sha",
        base,
        "--architectures-json",
        '["blackhole"]',
        "--backend",
        "local",
        "--output",
        str(manifest_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == 2
    assert [r["backend"] for r in manifest["requirements"]] == ["host", "silicon"]
    results = tmp_path / "verification-results"
    results.mkdir()
    env = {
        **os.environ,
        "TT_LLK_LOCAL_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "CODEGEN_REQUIRED_VERIFICATION_MANIFEST": str(manifest_path),
        "CODEGEN_RUN_ID": "host-run",
        "CODEGEN_ATTEMPT_ID": "attempt-001",
    }
    proc = subprocess.run(
        [
            "bash",
            str(wrapper),
            "host",
            "--worktree",
            str(tree),
            "--arch",
            "blackhole",
            "--test",
            "test_host.py",
            "--result-json-out",
            str(results / "host.json"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    host = json.loads((results / "host.json").read_text())
    _reduce(tmp_path, manifest_path)
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert reduction["classification"] == "partial"
    assert reduction["success_token"] is None
    device = _sealed_result(
        manifest,
        manifest["requirements"][1],
        patch_sha256=host["provenance"]["patch_sha256"],
    )
    device["provenance"]["actual_base_sha"] = base
    device["result_id"] = _content_id(device, {"result_id"})
    (results / "device.json").write_text(json.dumps(device))
    _reduce(tmp_path, manifest_path)
    assert (
        json.loads((tmp_path / "verification_reduction.json").read_text())[
            "classification"
        ]
        == "success"
    )
    # A host result cannot claim the silicon leaf, even with a recomputed receipt hash.
    fake = json.loads(json.dumps(host))
    fake["backend"] = "silicon"
    fake["result_id"] = _content_id(fake, {"result_id"})
    (results / "host.json").write_text(json.dumps(fake))
    _reduce(tmp_path, manifest_path)
    assert (
        json.loads((tmp_path / "verification_reduction.json").read_text())[
            "classification"
        ]
        == "infra_error"
    )
    # Existing wrapper entry points cannot route the host leaf through a device compile.
    rejected = subprocess.run(
        [
            "bash",
            str(wrapper),
            "host",
            "--worktree",
            str(tree),
            "--arch",
            "blackhole",
            "--test",
            "test_device.py",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert rejected.returncode == 3
    assert "host/device execution must match" in rejected.stderr
    # A later revision cannot silently downgrade the manifest's version.
    plan.write_text(plan.read_text().replace("  execution: host\n", ""))
    subprocess.run(
        [*cmd, "--supersedes-reason", "explicit route revision"],
        check=True,
        capture_output=True,
    )
    assert json.loads(manifest_path.read_text())["version"] == 2


@pytest.mark.parametrize("execution", ["gpu", "HOST", "silicon"])
def test_host_plan_requires_typed_execution(tmp_path, execution):
    proc, _ = _required_manifest(
        tmp_path,
        "## Verification\nverification_required: yes\nverifiable_in_llk_suite: yes\nllk_coverage: existing\n",
        "## Test Strategy\nreproduction_tests:\n- arch: blackhole\n  test: test_reduce.py\n  execution: "
        + execution
        + "\n",
        check=False,
    )
    assert proc.returncode != 0 and "execution must be device|host" in proc.stderr


def test_host_versioned_harness_does_not_import_pinned_device_conftest(tmp_path):
    def prepare(tree):
        writer = tree / "codegen/scripts/run_json_writer.py"
        writer.unlink()
        writer.symlink_to(SCRIPT)
        (tree / "tests/python_tests/conftest.py").write_text(
            'raise AssertionError("PINNED DEVICE HARNESS LOADED")\n'
        )

    proc, result, _ = _host_run(
        tmp_path,
        "import pytest\npytestmark=pytest.mark.llk_host\ndef test_schema(): pass\n",
        prepare=prepare,
    )
    assert proc.returncode == 0, proc.stderr
    assert result["execution"]["passed"] == 1
    assert "PINNED DEVICE HARNESS LOADED" not in proc.stderr
    assert result["provenance"]["host_inputs"]["harness_sha256"]


@pytest.fixture(scope="module")
def host_success_receipt(tmp_path_factory):
    proc, result, _ = _host_run(
        tmp_path_factory.mktemp("host-receipt"),
        "import pytest\npytestmark=pytest.mark.llk_host\ndef test_schema(): pass\n",
    )
    assert proc.returncode == 0, proc.stderr
    return result


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "attempt_id",
        "selector",
        "expected_base_sha",
        "patch_sha256",
        "source_tree_sha256",
        "dependencies_sha256",
        "observed_nodeids",
        "junit_sha256",
        "version",
    ],
)
def test_host_receipt_rejects_rehashed_identity_or_evidence_tampering(
    tmp_path, host_success_receipt, field
):
    import importlib.util

    spec = importlib.util.spec_from_file_location("host_validator", SCRIPT)
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    record = json.loads(json.dumps(host_success_receipt))
    if field in {"run_id", "attempt_id"}:
        record[field] += "-other"
    elif field == "selector":
        record[field]["test"] = "other.py"
    elif field == "version":
        record[field] = 2
    elif field in {"source_tree_sha256", "dependencies_sha256"}:
        record["provenance"]["host_inputs"][field] = "0" * 64
        if field == "dependencies_sha256":
            record["provenance"]["host_inputs_sha256"] = writer._canonical_digest(
                record["provenance"]["host_inputs"]
            )
    elif field == "observed_nodeids":
        record["provenance"][field] = ["test_host.py::other"]
    elif field == "junit_sha256":
        record["provenance"][field] = None
    else:
        record["provenance"][field] = "0" * (40 if field == "expected_base_sha" else 64)
    record["result_id"] = _content_id(record, {"result_id"})
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        writer._load_verification_result(path)


def test_host_skips_cannot_certify_complete_coverage(tmp_path):
    proc, result, _ = _host_run(
        tmp_path,
        'import pytest\npytestmark=pytest.mark.llk_host\ndef test_ok(): pass\n@pytest.mark.skip(reason="missing")\ndef test_missing(): pass\n',
    )
    assert proc.returncode == 1, proc.stderr
    assert result["execution"]["passed"] == 1 and result["execution"]["skipped"] == 1
    assert result["classification"] == "coverage_error"
    assert result["reason_codes"] == ["host_execution_outcome_incomplete"]


def test_device_manifest_still_rejects_empty_artifact_root(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "artifact-manifest",
            "--output",
            str(tmp_path / "manifest.json"),
            "--artifact-root",
            str(root),
            "--owner-id",
            "owner",
            "--build-input-digest",
            "1" * 64,
            "--source-tree-sha256",
            "2" * 64,
            "--compiler-sha256",
            "3" * 64,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0 and "artifact root contains no files" in proc.stderr
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize(
    "body",
    [
        "import pytest\n@pytest.mark.llk_host\ndef test_pure(): pass\ndef test_device(): pass\n",
        'import pytest\npytestmark=pytest.mark.llk_host\n@pytest.fixture\ndef tmp_path(): raise AssertionError("DEVICE FIXTURE RAN")\ndef test_masked(tmp_path): pass\n',
        'import pytest\npytestmark=pytest.mark.llk_host\n@pytest.fixture\ndef device(): raise AssertionError("DEVICE FIXTURE RAN")\ndef test_dynamic(request): request.getfixturevalue("device")\n',
    ],
)
def test_host_rejects_mixed_and_hidden_fixture_execution(tmp_path, body):
    proc, result, _ = _host_run(tmp_path, body)
    assert proc.returncode != 0, proc.stderr
    assert result is not None and result["classification"] != "success"
    assert "DEVICE FIXTURE RAN" not in proc.stderr


def test_host_approved_pytest_fixture_and_exact_node_selection(tmp_path):
    proc, result, _ = _host_run(
        tmp_path,
        'import pytest\npytestmark=pytest.mark.llk_host\ndef test_safe(tmp_path,monkeypatch):\n    monkeypatch.setenv("HOST_CHECK","1")\n    (tmp_path/"x").write_text("ok")\ndef test_not_selected(): assert False\n',
        extra=("--test-id", "test_host.py::test_safe"),
        env_extra={
            "PYTEST_PLUGINS": "missing_unsafe_plugin",
            "PYTEST_ADDOPTS": "--run-simulator",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert result["selector"]["test_id"] == "test_host.py::test_safe"
    assert result["provenance"]["observed_nodeids"] == ["test_host.py::test_safe"]


def test_measurement_counts_are_canonical_before_manifest_hash(tmp_path):
    analysis, plan, _ = _measurement_plan()
    plan = plan.replace('"tile_cnt": "8"', '"tile_cnt": "8.0"').replace(
        '"loop_factor": "16"', '"loop_factor": "16.0"'
    )
    _, path = _required_manifest(tmp_path, analysis, plan)
    manifest = json.loads(path.read_text())
    variant = manifest["requirements"][-1]["measurement_contract"]["variants"][0]
    assert variant["tile_cnt"] == "8" and variant["loop_factor"] == "16"
    assert manifest["manifest_id"] == _content_id(manifest, {"manifest_id"})


def test_legacy_v2_integer_spelling_load_and_reseal_preserve_old_manifest(tmp_path):
    analysis, plan, _ = _measurement_plan()
    _, path = _required_manifest(tmp_path, analysis, plan)
    old = json.loads(path.read_text())
    old["requirements"][-1]["measurement_contract"]["variants"][0]["tile_cnt"] = "8.0"
    old["manifest_id"] = _content_id(old, {"manifest_id"})
    old_bytes = (json.dumps(old) + "\n").encode()
    revision = tmp_path / "required_verification_manifests/revision-001.json"
    path.write_bytes(old_bytes)
    revision.write_bytes(old_bytes)
    _reduce(tmp_path, path)
    assert path.read_bytes() == old_bytes
    _, output = _required_manifest(
        tmp_path, analysis, plan, "--supersedes-reason", "retry unchanged scope"
    )
    resealed = json.loads(output.read_text())
    assert resealed["parent_manifest_id"] == old["manifest_id"]
    assert (
        resealed["requirements"][-1]["measurement_contract"]["variants"][0]["tile_cnt"]
        == "8"
    )
    assert revision.read_bytes() == old_bytes


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("custom_hooks", [False, True])
def test_worktree_toolchain_setup_preserves_live_shared_hooks(
    tmp_path, cached, custom_hooks
):
    """Actual Git hooks survive setup and deletion; SFPI stays pinned to the child."""
    import tarfile

    repo = tmp_path / "repo"
    tests = repo / "tt_metal/tt-llk/tests"
    tests.mkdir(parents=True)
    (tests.parent / ".gitignore").write_text("tests/sfpi/\n*.observed\n")
    # An old base setup would install into the common Git directory. The caller
    # must select the versioned toolchain entrypoint instead of executing this.
    (tests / "setup_testing_env.sh").write_text("#!/bin/bash\npre-commit install\n")
    (tests / "setup_testing_env.sh").chmod(0o755)
    payload = tmp_path / "payload"
    (payload / "sfpi").mkdir(parents=True)
    (payload / "sfpi/compiler").write_text("pinned compiler\n")
    archive = tmp_path / "pinned.txz"
    with tarfile.open(archive, "w:xz") as output:
        output.add(payload / "sfpi", arcname="sfpi")
    sfpi = tests / "sfpi-info.sh"
    sfpi.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$CHIP_ARCH:$*" > "$(dirname "$0")/sfpi.observed"\n'
        f"echo sfpi_version=pinned-fixture sfpi_hash={hashlib.sha256(archive.read_bytes()).hexdigest()} "
        "sfpi_hashtype=sha256 sfpi_url=https://fixture.invalid sfpi_filename=pinned.txz\n"
    )
    sfpi.chmod(0o755)
    git = lambda *args: subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.invalid")
    git("add", "-A")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD").stdout.strip()
    # Cached case is provided by a tracked fixture path; real Git setup creates
    # the child before this existing metadata check, just as retained SFPI does.
    if cached:
        git("add", "-f", "tt_metal/tt-llk/tests/sfpi-info.sh")
        (tests / "sfpi").mkdir()
        (tests / "sfpi/sfpi.version").write_text("pinned-fixture\n")
        (tests / "sfpi/compiler").write_text("pinned compiler\n")
        git("add", "-f", "tt_metal/tt-llk/tests/sfpi")
        git("commit", "-qm", "cached fixture only")
        base = git("rev-parse", "HEAD").stdout.strip()
    hooks = tmp_path / "human-hooks" if custom_hooks else repo / ".git/hooks"
    hooks.mkdir(exist_ok=True)
    if custom_hooks:
        git("config", "core.hooksPath", str(hooks))
    hook = hooks / "pre-commit"
    hook_bytes = b'#!/bin/sh\nprintf "human-hook\\n" >> "$HOOK_LOG"\n'
    hook.write_bytes(hook_bytes)
    hook.chmod(0o755)
    hook_log = tmp_path / "hook.log"
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    installer = fakebin / "pre-commit"
    installer.write_text(
        "#!/bin/bash\n"
        'printf "#!/deleted/worktree/python\\n" > "$(git rev-parse --git-path hooks)/pre-commit"\n'
        'touch "$INSTALL_MARKER"\n'
    )
    installer.chmod(0o755)
    wget = fakebin / "wget"
    wget.write_text('#!/bin/bash\ncp "$SFPI_ARCHIVE" "$3/pinned.txz"\n')
    wget.chmod(0o755)
    worktrees = tmp_path / "worktrees"
    child = worktrees / "issue-hooks-v1"
    env = {
        **os.environ,
        "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
        "INSTALL_MARKER": str(tmp_path / "installed"),
        "HOOK_LOG": str(hook_log),
        "SFPI_ARCHIVE": str(archive),
        "CHIP_ARCH": "blackhole",
    }
    command = r"""
source "$1"
REPO_ROOT="$2"
LLK_REL="tt_metal/tt-llk"
CODEGEN_GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --absolute-git-dir)"
CODEGEN_SETUP_LOCK="$CODEGEN_GIT_DIR/codegen-worktree-setup.lock"
CODEGEN_WORKTREE_ROOT="$3"
CODEGEN_BASE_COMMIT="$4"
setup_worktree issue-hooks
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(SETUP_WORKTREE),
            str(repo),
            str(worktrees),
            base,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not Path(env["INSTALL_MARKER"]).exists()
    assert hook.read_bytes() == hook_bytes
    child_tests = child / "tt_metal/tt-llk/tests"
    assert (child_tests / "sfpi/sfpi.version").read_text().strip() == "pinned-fixture"
    assert (child_tests / "sfpi/compiler").read_text() == "pinned compiler\n"
    expected_arch = "blackhole" if Path("/dev/tenstorrent").exists() else "quasar"
    assert (
        child_tests / "sfpi.observed"
    ).read_text().strip() == f"{expected_arch}:SHELL txz"
    assert (child_tests / "setup_testing_env.sh").read_bytes() == (
        tests / "setup_testing_env.sh"
    ).read_bytes()
    assert not (tests / "sfpi.observed").exists()
    if not cached:
        assert not (tests / "sfpi").exists()
    subprocess.run(
        ["git", "-C", str(child), "commit", "--allow-empty", "-qm", "child"],
        env=env,
        check=True,
    )
    git("worktree", "remove", "--force", str(child))
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "commit",
            "--allow-empty",
            "-qm",
            "human after cleanup",
        ],
        env=env,
        check=True,
    )
    assert hook.read_bytes() == hook_bytes
    assert hook_log.read_text().splitlines() == ["human-hook", "human-hook"]


def test_testing_setup_default_still_installs_hooks_and_rejects_bad_mode(tmp_path):
    import shutil

    source = SETUP_WORKTREE.parents[2] / "tests/setup_testing_env.sh"
    repo = tmp_path / "human"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    tmp_path = repo / "tests"
    tmp_path.mkdir()
    script = tmp_path / "setup_testing_env.sh"
    shutil.copyfile(source, script)
    (tmp_path / "sfpi").mkdir()
    (tmp_path / "sfpi/sfpi.version").write_text("fixture\n")
    metadata = tmp_path / "sfpi-info.sh"
    metadata.write_text("#!/bin/bash\necho sfpi_version=fixture\n")
    metadata.chmod(0o755)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    installer = fakebin / "pre-commit"
    installer.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$INSTALL_LOG"\n')
    installer.chmod(0o755)
    log = tmp_path / "install.log"
    env = {
        **os.environ,
        "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
        "INSTALL_LOG": str(log),
    }
    default = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True
    )
    assert default.returncode == 0, default.stderr
    assert log.read_text() == "install\n"
    for args in [
        ["--toolchain-only"],
        ["--skip-hooks", str(tmp_path)],
        ["--toolchain-only", str(tmp_path / "absent")],
    ]:
        invalid = subprocess.run(
            ["bash", str(script), *args], env=env, capture_output=True, text=True
        )
        assert invalid.returncode == 2
        assert "Usage:" in invalid.stderr
    assert log.read_text() == "install\n"


def _retry_context_fixture(tmp_path, outcomes):
    requirements = [_requirement(index=i + 1) for i in range(len(outcomes))]
    manifest, manifest_path = _reducer_manifest(tmp_path, requirements)
    _run(
        tmp_path,
        "init",
        "--run-id",
        manifest["run_id"],
        "--kernel",
        "issue_5",
        "--arch",
        "blackhole",
        "--first-step",
        "tester",
        "--first-message",
        "verify",
    )
    run_path = tmp_path / "run.json"
    run = json.loads(run_path.read_text())
    run["base_commit"] = manifest["expected_base_sha"]
    run["required_verification"] = {
        key: manifest[key] for key in ("manifest_id", "attempt_id")
    }
    run_path.write_text(json.dumps(run))
    results = tmp_path / "verification-results"
    results.mkdir()
    for requirement, outcome in zip(requirements, outcomes):
        if outcome is not None:
            receipt = _sealed_result(manifest, requirement, **outcome)
            (results / f"{requirement['requirement_id']}.json").write_text(
                json.dumps(receipt)
            )
    _reduce(tmp_path, manifest_path, scope="functional")
    return manifest


@pytest.mark.parametrize(
    ("outcomes", "failure_class", "retry"),
    [
        (
            [{"selected": 48, "executed": 35, "passed": 35, "skipped": 13}],
            "VERIFICATION_PLAN_ERROR",
            True,
        ),
        (
            [{"selected": 2, "executed": 1, "passed": 1, "skipped": 1}],
            "VERIFICATION_PLAN_ERROR",
            True,
        ),
        ([{"selected": 0, "executed": 0, "passed": 0}], "MISSING_TEST_COVERAGE", True),
        ([{"passed": 0, "failed": 1, "returncode": 1}], "TESTS_FAILED", True),
        (
            [
                {
                    "selected": 48,
                    "executed": 35,
                    "passed": 34,
                    "failed": 1,
                    "skipped": 13,
                    "returncode": 1,
                }
            ],
            "TESTS_FAILED",
            True,
        ),
        (
            [
                {"selected": 2, "executed": 1, "passed": 1, "skipped": 1},
                {"passed": 0, "failed": 1, "returncode": 1},
            ],
            "TESTS_FAILED",
            True,
        ),
        ([{}, {"patch_sha256": "e" * 64}], "ENV_ERROR", False),
        ([None], "ENV_ERROR", False),
        ([{"markers": ["tt_fatal"]}], "ENV_ERROR", False),
        ([{"selected": 2, "executed": 1, "passed": 1}], "ENV_ERROR", False),
        ([{}], None, False),
    ],
)
def test_verification_retry_context_preserves_failure_kind(
    tmp_path, outcomes, failure_class, retry
):
    manifest = _retry_context_fixture(tmp_path, outcomes)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    context = json.loads(_run(tmp_path, "verification-retry-context").stdout)
    assert context["failure_class"] == failure_class
    assert context["retry_allowed"] is retry
    assert context["manifest_id"] == manifest["manifest_id"]
    assert context["attempt_id"] == manifest["attempt_id"]
    assert context["results_dir"] == str(tmp_path / "verification-results")
    reduction = json.loads((tmp_path / "verification_reduction.json").read_text())
    assert context["reason_codes"] == reduction["reason_codes"]
    assert len(context["leaves"]) == sum(
        leaf["classification"] != "success" for leaf in reduction["leaves"]
    )
    assert all(p.read_bytes() == contents for p, contents in before.items())


@pytest.mark.parametrize(
    "changed", ["run", "attempt", "manifest", "reduction_pointer", "scope"]
)
def test_verification_retry_context_rejects_stale_evidence(tmp_path, changed):
    _retry_context_fixture(
        tmp_path, [{"selected": 2, "executed": 1, "passed": 1, "skipped": 1}]
    )
    path = tmp_path / "run.json"
    run = json.loads(path.read_text())
    if changed == "run":
        run["run_id"] = "another"
    elif changed == "attempt":
        run["required_verification"]["attempt_id"] = "attempt-002"
    elif changed == "manifest":
        run["required_verification"]["manifest_id"] = "f" * 64
    elif changed == "reduction_pointer":
        run["verification_reduction"]["reduction_id"] = "f" * 64
    else:
        _reduce(tmp_path, tmp_path / "required_verification_manifest.json", scope="all")
        run = json.loads(path.read_text())
    path.write_text(json.dumps(run))
    before = path.read_bytes()
    with pytest.raises(subprocess.CalledProcessError):
        _run(tmp_path, "verification-retry-context")
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("outcome", "advance"),
    [
        ({"selected": 48, "executed": 35, "passed": 35, "skipped": 13}, True),
        (None, False),
    ],
)
@pytest.mark.parametrize("review_round", [False, True])
def test_feedback_wrapper_uses_typed_coverage_and_blocks_missing_receipts(
    tmp_path, outcome, advance, review_round
):
    _retry_context_fixture(tmp_path, [outcome])
    llk = tmp_path / "worktree/tt_metal/tt-llk"
    llk.mkdir(parents=True)
    (llk / ".codegen_run_state.json").write_text(json.dumps({"LOG_DIR": str(tmp_path)}))
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "ISSUE_NUMBER": "5",
                "PR_NUMBER": "9",
                "DEBUG_CYCLES": 0,
                "MAX_DEBUG_CYCLES": 3,
            }
        )
    )
    command = (
        'execute_step_review_round_feedback tester "observed failure"'
        if review_round
        else 'execute_step_debug_feedback "observed failure"'
    )
    result = subprocess.run(
        ["bash", "-c", f'source "$1"; {command}', "bash", str(ORCHESTRATOR_STEPS)],
        cwd=llk,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is advance, result.stdout + result.stderr
    run = json.loads((tmp_path / "run.json").read_text())
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["FAILURE_CLASS"] == (
        "VERIFICATION_PLAN_ERROR" if advance else "ENV_ERROR"
    )
    assert state["VERIFICATION_RETRY_CONTEXT"]["leaves"][0]["reason_codes"]
    assert run["current_step"] == ("fix_tests" if advance else "tester")


def test_feedback_wrapper_preserves_legacy_without_reduction(tmp_path):
    _run(
        tmp_path,
        "init",
        "--run-id",
        "legacy",
        "--kernel",
        "issue_5",
        "--arch",
        "blackhole",
        "--first-step",
        "tester",
        "--first-message",
        "verify",
    )
    llk = tmp_path / "worktree/tt_metal/tt-llk"
    llk.mkdir(parents=True)
    (llk / ".codegen_run_state.json").write_text(json.dumps({"LOG_DIR": str(tmp_path)}))
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "ISSUE_NUMBER": "5",
                "DEBUG_CYCLES": 0,
                "MAX_DEBUG_CYCLES": 3,
                "FAILURE_CLASS": "COMPILE_ERROR",
                "VERIFICATION_RETRY_CONTEXT": {"stale": True},
            }
        )
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; execute_step_debug_feedback "compiler error"',
            "bash",
            str(ORCHESTRATOR_STEPS),
        ],
        cwd=llk,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        json.loads((tmp_path / "run.json").read_text())["current_step"] == "fix_tests"
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["FAILURE_CLASS"] == "COMPILE_ERROR"
    assert state["VERIFICATION_RETRY_CONTEXT"] == {}


def test_feedback_wrapper_rejects_missing_audit_reduction(tmp_path):
    _retry_context_fixture(
        tmp_path, [{"selected": 2, "executed": 1, "passed": 1, "skipped": 1}]
    )
    path = tmp_path / "run.json"
    run = json.loads(path.read_text())
    run["runner_pool"] = "audit"
    path.write_text(json.dumps(run))
    before = path.read_bytes()
    (tmp_path / "verification_reduction.json").unlink()
    llk = tmp_path / "worktree/tt_metal/tt-llk"
    llk.mkdir(parents=True)
    (llk / ".codegen_run_state.json").write_text(json.dumps({"LOG_DIR": str(tmp_path)}))
    (tmp_path / "state.json").write_text(
        json.dumps({"ISSUE_NUMBER": "5", "DEBUG_CYCLES": 0, "MAX_DEBUG_CYCLES": 3})
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; execute_step_debug_feedback "missing evidence"',
            "bash",
            str(ORCHESTRATOR_STEPS),
        ],
        cwd=llk,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert path.read_bytes() == before


@pytest.mark.parametrize("review_round", [False, True])
@pytest.mark.parametrize("receipt", ["absent", "missing", "numerical"])
def test_feedback_wrapper_preserves_explicit_pre_execution_compile_retry(
    tmp_path, receipt, review_round
):
    outcome = (
        {"selected": 2, "executed": 2, "passed": 1, "failed": 1}
        if receipt == "numerical"
        else None
    )
    _retry_context_fixture(tmp_path, [outcome])
    path = tmp_path / "run.json"
    run = json.loads(path.read_text())
    run["runner_pool"] = "audit"
    path.write_text(json.dumps(run))
    if receipt == "absent":
        (tmp_path / "verification_reduction.json").unlink()
    llk = tmp_path / "worktree/tt_metal/tt-llk"
    llk.mkdir(parents=True)
    (llk / ".codegen_run_state.json").write_text(json.dumps({"LOG_DIR": str(tmp_path)}))
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "ISSUE_NUMBER": "5",
                "PR_NUMBER": "9",
                "DEBUG_CYCLES": 0,
                "MAX_DEBUG_CYCLES": 3,
            }
        )
    )
    (tmp_path / "compile.log").write_text("error: unknown type name\n")
    # Same suite record emitted by the tester before entering the debug loop.
    _run(
        tmp_path,
        "metric",
        "--patch-json",
        json.dumps(
            {
                "arch_results": {
                    "blackhole": {
                        "suite_results": {
                            "llk": {
                                "status": "done",
                                "verdict": "COMPILE_FAILED",
                                "tests_total": 0,
                                "tests_passed": 0,
                                "queue_jobs": [],
                                "obstacle": str(tmp_path / "compile.log"),
                            }
                        }
                    }
                }
            }
        ),
    )
    command = (
        'execute_step_review_round_feedback tester "compile.log: unknown type" COMPILE_FAILED'
        if review_round
        else 'execute_step_debug_feedback "compile.log: unknown type" COMPILE_FAILED'
    )
    result = subprocess.run(
        ["bash", "-c", f'source "$1"; {command}', "bash", str(ORCHESTRATOR_STEPS)],
        cwd=llk,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["FAILURE_CLASS"] == (
        "TESTS_FAILED" if receipt == "numerical" else "COMPILE_ERROR"
    )
    assert bool(state["VERIFICATION_RETRY_CONTEXT"]) is (receipt == "numerical")
    run = json.loads(path.read_text())
    assert run["current_step"] == "fix_tests"
    assert run["status"] != "success"


# Real Git, host pytest and strict reducer; only silicon dispatch is faked.
def _functional_fixture(tmp_path, monkeypatch, requirements=None, host_body=None):
    import argparse
    import importlib.util

    tree, _, base = _host_worktree(
        tmp_path,
        host_body
        or """import pytest
pytestmark=pytest.mark.llk_host
@pytest.mark.parametrize("value", range(19))
def test_host(value): assert value >= 0
""",
    )
    root, logs = tree.parents[1], tmp_path / "logs"
    logs.mkdir()
    host = _requirement(
        backend="host",
        selector={"test": "test_host.py", "test_id": None, "k": None},
        minimum_selected=19,
        minimum_executed=19,
    )
    device = _requirement(
        index=2,
        selector={
            "test": "test_device.py",
            "test_id": "test_device.py::test_device",
            "k": None,
        },
        minimum_selected=35,
        minimum_executed=35,
    )
    manifest, path = _reducer_manifest(logs, requirements or [host, device])
    manifest.update(version=2, expected_base_sha=base)
    manifest["manifest_id"] = _content_id(manifest, {"manifest_id"})
    path.write_text(json.dumps(manifest))
    state = {
        "RUN_ID": manifest["run_id"],
        "LOG_DIR": str(logs),
        "WORKTREE_DIR": str(root),
        "GIT_COMMIT": base,
        "REQUIRED_VERIFICATION_MANIFEST": str(path),
        "REQUIRED_VERIFICATION_MANIFEST_ID": manifest["manifest_id"],
        "REQUIRED_VERIFICATION_ATTEMPT_ID": manifest["attempt_id"],
        "HW_TEST_DISPATCH_CMD": "sealed-dispatch-fixture",
    }
    (logs / "state.json").write_text(json.dumps(state))
    (tree / ".codegen_run_state.json").write_text(
        json.dumps({"RUN_ID": manifest["run_id"], "LOG_DIR": str(logs)})
    )
    (logs / "run.json").write_text(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "status": "running",
                "runner_pool": "audit",
                "base_commit": base,
                "functional_executor": "sealed-llk-v1",
                "required_verification": {
                    "manifest_id": manifest["manifest_id"],
                    "attempt_id": manifest["attempt_id"],
                },
            }
        )
    )
    spec = importlib.util.spec_from_file_location("sealed_functional_writer", SCRIPT)
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    monkeypatch.setenv("TT_LLK_LOCAL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("CODEGEN_HOST_TIMEOUT_SECS", "10")
    monkeypatch.delenv("CODEGEN_PATCH_SHA256", raising=False)
    return (
        writer,
        argparse.Namespace(log_dir=str(logs), worktree=str(root), timeout=30),
        manifest,
        tree,
    )


def _fake_functional_dispatch(
    monkeypatch,
    writer,
    args,
    manifest,
    *,
    outcomes=None,
    mutate=None,
    bad_description=None,
    missing=False,
    job_mismatch=False,
):
    real_run, calls = subprocess.run, []
    plan = writer._functional_execution_plan(Path(args.log_dir), Path(args.worktree))
    commands = {c["leaf"]["requirement_id"]: c for c in plan["commands"]}

    def fake(argv, **kwargs):
        if argv[0] != "sealed-dispatch-fixture":
            return real_run(argv, **kwargs)
        calls.append(list(argv))
        if "--help" in argv:
            return subprocess.CompletedProcess(
                argv, 0, "--requirement-id --describe --result-json-out", ""
            )
        identity = argv[argv.index("--requirement-id") + 1]
        command = commands[identity]
        leaf = command["leaf"]
        if "--describe" in argv:
            context = {
                "run_id": plan["run_id"],
                "attempt_id": plan["attempt_id"],
                "manifest_id": plan["manifest_id"],
                "requirement_id": identity,
                "arch": leaf["architecture"],
                "kind": "llk",
                "base": plan["base"],
                "worktree": args.worktree,
                "runner_pool": "audit",
                "copy_result_json": True,
                "test": leaf["selector"]["test_id"] or leaf["selector"]["test"],
                "test_filter": leaf["selector"]["k"],
                "result_json_out": command["result"],
            }
            context.update(bad_description or {})
            return subprocess.CompletedProcess(argv, 0, json.dumps(context), "")
        assert kwargs["env"].items() >= command["env"].items()
        assert kwargs["cwd"] == Path(args.worktree) / "tt_metal/tt-llk"
        assert argv[-2:] == ["--timeout", "30"]
        job_id = "job-device" if leaf["architecture"] == "blackhole" else "job-wormhole"
        kwargs["stdout"].write(
            f'HW_TEST_RESULT arch={leaf["architecture"]} ok=true ran=true passed=true job={job_id} failure_stage=build summary="fixture build output"\n'
        )
        if not missing:
            counts = {"selected": 35, "executed": 35, "passed": 35, **(outcomes or {})}
            result = _sealed_result(
                manifest,
                leaf,
                patch_sha256=plan["patch_sha256"],
                job_id="wrong-job" if job_mismatch else job_id,
                **counts,
            )
            output = Path(command["result"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result))
        if mutate:
            mutate()
        return subprocess.CompletedProcess(argv, 1 if missing else 0)

    monkeypatch.setattr(writer.subprocess, "run", fake)
    return calls


def _submitted(calls):
    return [c for c in calls if "--help" not in c and "--describe" not in c]


def test_sealed_functional_executes_real_host_and_exact_device_once(
    tmp_path, monkeypatch
):
    writer, args, manifest, _ = _functional_fixture(tmp_path, monkeypatch)
    calls = _fake_functional_dispatch(monkeypatch, writer, args, manifest)
    assert writer.cmd_execute_functional(args) == 0
    log = Path(args.log_dir)
    run = json.loads((log / "run.json").read_text())
    reduction = json.loads((log / "verification_reduction.json").read_text())
    assert reduction["classification"] == "success"
    assert reduction["tests_total"] == reduction["tests_passed"] == 54
    assert reduction["scope"] == "functional" and reduction["success_token"] is None
    assert run["status"] == "running" and run["current_step"] == "tester"
    assert [r["requirement_id"] for r in run["functional_execution"]["leaves"]] == [
        r["requirement_id"] for r in manifest["requirements"]
    ]
    assert len(_submitted(calls)) == 1
    before = list(calls)
    assert writer.cmd_execute_functional(args) == 0 and calls == before
    Path(run["functional_execution"]["leaves"][0]["result"]).unlink()
    assert writer.cmd_execute_functional(args) == 1 and calls == before


@pytest.mark.parametrize(
    "body",
    [
        "def test_unmarked(): pass\n",
        "import not_a_real_host_dependency\n",
        """import pytest
pytestmark=pytest.mark.llk_host
@pytest.fixture
def device(): raise AssertionError("device executed")
def test_device(device): pass
""",
    ],
)
def test_sealed_functional_unsupported_host_preflights_whole_route(
    tmp_path, monkeypatch, body, capsys
):
    device = _requirement(
        selector={"test": "test_device.py", "test_id": None, "k": None}
    )
    host = _requirement(
        index=2,
        backend="host",
        selector={"test": "test_host.py", "test_id": None, "k": None},
    )
    writer, args, manifest, _ = _functional_fixture(
        tmp_path, monkeypatch, [device, host], body
    )
    calls = _fake_functional_dispatch(monkeypatch, writer, args, manifest)
    assert writer.cmd_execute_functional(args) == 20 and not _submitted(calls)
    assert "preflight failed" in capsys.readouterr().out
    assert "functional_execution" not in json.loads(
        (Path(args.log_dir) / "run.json").read_text()
    )
    assert not (Path(args.log_dir) / "verification-results").exists()


@pytest.mark.parametrize(
    "defect", ["prod", "disabled", "metal", "simulator", "dispatch"]
)
def test_sealed_functional_unsupported_plan_never_executes(
    tmp_path, monkeypatch, defect
):
    requirements = [
        _requirement(selector={"test": "test_device.py", "test_id": None, "k": None})
    ]
    if defect == "metal":
        requirements.append(_requirement(suite="metal"))
    if defect == "simulator":
        requirements[0]["backend"] = "ttsim"
    writer, args, _, _ = _functional_fixture(tmp_path, monkeypatch, requirements)
    logs = Path(args.log_dir)
    if defect in {"prod", "disabled"}:
        run = json.loads((logs / "run.json").read_text())
        run["runner_pool" if defect == "prod" else "functional_executor"] = (
            "prod" if defect == "prod" else None
        )
        (logs / "run.json").write_text(json.dumps(run))
    if defect == "dispatch":
        state = json.loads((logs / "state.json").read_text())
        state["HW_TEST_DISPATCH_CMD"] = ""
        (logs / "state.json").write_text(json.dumps(state))
    calls = []
    monkeypatch.setattr(writer.subprocess, "run", lambda *a, **k: calls.append(a))
    assert writer.cmd_execute_functional(args) == 20 and calls == []


@pytest.mark.parametrize(
    "defect", ["run", "attempt", "worktree", "base", "description"]
)
def test_sealed_functional_rejects_foreign_identity_before_dispatch(
    tmp_path, monkeypatch, defect
):
    writer, args, manifest, tree = _functional_fixture(tmp_path, monkeypatch)
    calls = _fake_functional_dispatch(
        monkeypatch,
        writer,
        args,
        manifest,
        bad_description={"run_id": "wrong"} if defect == "description" else None,
    )
    state_path = Path(args.log_dir) / "state.json"
    state = json.loads(state_path.read_text())
    if defect == "run":
        state["RUN_ID"] = "wrong"
    if defect == "attempt":
        state["REQUIRED_VERIFICATION_ATTEMPT_ID"] = "wrong"
    if defect == "worktree":
        state["WORKTREE_DIR"] = str(tree)
    if defect == "base":
        state["GIT_COMMIT"] = "0" * 40
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="identity|current sealed"):
        writer.cmd_execute_functional(args)
    assert not _submitted(calls)


@pytest.mark.parametrize(
    "defect", ["skips", "numerical", "missing", "wrong_job", "mutation"]
)
def test_sealed_functional_preserves_partial_and_never_falls_back(
    tmp_path, monkeypatch, defect
):
    device = _requirement(
        selector={"test": "test_device.py", "test_id": None, "k": None}
    )
    host = _requirement(
        index=2,
        backend="host",
        selector={"test": "test_host.py", "test_id": None, "k": None},
    )
    writer, args, manifest, tree = _functional_fixture(
        tmp_path, monkeypatch, [device, host]
    )
    counts = (
        {"executed": 22, "passed": 22, "skipped": 13}
        if defect == "skips"
        else (
            {"passed": 34, "failed": 1, "returncode": 1}
            if defect == "numerical"
            else {}
        )
    )
    mutate = (
        (
            lambda: (tree / "tests/python_tests/test_device.py").write_text(
                "def test_device(): assert False\n"
            )
        )
        if defect == "mutation"
        else None
    )
    calls = _fake_functional_dispatch(
        monkeypatch,
        writer,
        args,
        manifest,
        outcomes=counts,
        missing=defect == "missing",
        job_mismatch=defect == "wrong_job",
        mutate=mutate,
    )
    assert writer.cmd_execute_functional(args) == 1
    logs = Path(args.log_dir)
    record = json.loads((logs / "run.json").read_text())["functional_execution"]
    assert record["status"] == "failed" and len(_submitted(calls)) == 1
    if defect in {"missing", "wrong_job", "mutation"}:
        assert record["leaves"][0]["queue_job_id"] == "job-device" and record["error"]
        assert record["leaves"][0]["failure_stage"] == "build"
        assert record["leaves"][0]["summary"] == "fixture build output"
        assert record["unstarted"] == [host["requirement_id"]]
    else:
        reduction = json.loads((logs / "verification_reduction.json").read_text())
        assert reduction["classification"] == (
            "coverage_error" if defect == "skips" else "candidate_failure"
        )
        assert len(record["leaves"]) == 2
    before = list(calls)
    if defect == "mutation":
        with pytest.raises(ValueError, match="different candidate"):
            writer.cmd_execute_functional(args)
    else:
        assert writer.cmd_execute_functional(args) == 1
    assert calls == before


def test_host_collect_only_never_executes_or_emits_receipt(tmp_path):
    proc, receipt, tree = _host_run(
        tmp_path,
        """from pathlib import Path
import pytest
pytestmark=pytest.mark.llk_host
@pytest.mark.parametrize("value", range(19))
def test_host(value): Path("BODY_RAN").write_text("bad")
""",
        extra=("--collect-only",),
        env_extra={"CONSUMER_RETURN_CODE": "0"},
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["selected"] == 19
    assert receipt is None and not (tree / "tests/python_tests/BODY_RAN").exists()
    assert "[RESULT]" not in proc.stderr
    assert not list(tmp_path.rglob("consumer-junit.xml"))


def test_host_collect_only_rejects_nonhost_modes(tmp_path):
    proc = subprocess.run(
        ["bash", str(RUN_TEST), "compile", "--collect-only"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 4 and "host-only" in proc.stderr


@pytest.mark.parametrize("defect", ["nodeids", "version", "mutation"])
def test_sealed_functional_collection_faults_cannot_admit_silicon(
    tmp_path, monkeypatch, defect
):
    writer, args, manifest, tree = _functional_fixture(tmp_path, monkeypatch)
    calls = _fake_functional_dispatch(monkeypatch, writer, args, manifest)
    real_run = writer.subprocess.run

    def collector(argv, **kwargs):
        if "--collect-only" in argv:
            if defect == "mutation":
                (tree / "tests/python_tests/test_device.py").write_text(
                    "def test_device(): assert False\n"
                )
                return subprocess.CompletedProcess(
                    argv, 1, "", "import failed after mutation"
                )
            doc = {
                "schema": "tt.issue-solver.pytest-collection",
                "version": 2,
                "selected": 19,
                "collected": 19,
                "errors": 0,
                "returncode": 0,
                "nodeids": [f"test_host.py::test_host[{n}]" for n in range(19)],
            }
            if defect == "version":
                doc["version"] = 1
            else:
                doc["nodeids"][1] = doc["nodeids"][0]
            return subprocess.CompletedProcess(argv, 0, json.dumps(doc), "")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(writer.subprocess, "run", collector)
    with pytest.raises(ValueError, match="schema|nodeids|identity changed"):
        writer.cmd_execute_functional(args)
    assert not _submitted(calls)


def test_sealed_functional_refuses_preexisting_attempt_receipts(tmp_path, monkeypatch):
    writer, args, manifest, _ = _functional_fixture(tmp_path, monkeypatch)
    calls = _fake_functional_dispatch(monkeypatch, writer, args, manifest)
    results = Path(args.log_dir) / "verification-results" / manifest["attempt_id"]
    results.mkdir(parents=True)
    (results / "existing.json").write_text(
        json.dumps(_sealed_result(manifest, manifest["requirements"][1]))
    )
    with pytest.raises(ValueError, match="already exist"):
        writer.cmd_execute_functional(args)
    assert calls == []


@pytest.mark.parametrize("mode", [None, "sealed-llk-v1", "unknown"])
def test_sealed_functional_optin_only_initialized_from_environment(
    tmp_path, monkeypatch, mode
):
    if mode:
        monkeypatch.setenv("CODEGEN_FUNCTIONAL_EXECUTOR", mode)
    else:
        monkeypatch.delenv("CODEGEN_FUNCTIONAL_EXECUTOR", raising=False)
    command = [
        sys.executable,
        str(SCRIPT),
        "init",
        "--log-dir",
        str(tmp_path),
        "--run-id",
        "run-test",
        "--kernel",
        "test",
        "--arch",
        "blackhole",
        "--first-step",
        "writer",
        "--first-message",
        "fix",
        "--patch-json",
        '{"functional_executor":"forged","functional_execution":{"status":"success"}}',
    ]
    proc = subprocess.run(command, capture_output=True, text=True)
    if mode == "unknown":
        assert proc.returncode != 0 and not (tmp_path / "run.json").exists()
        return
    assert proc.returncode == 0, proc.stderr
    doc = json.loads((tmp_path / "run.json").read_text())
    assert doc.get("functional_executor") == mode and "functional_execution" not in doc
    for key in (
        "functional_executor",
        "functional_execution.status",
        ".functional_execution.status",
    ):
        before = (tmp_path / "run.json").read_bytes()
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "metric",
                "--log-dir",
                str(tmp_path),
                "--patch-json",
                json.dumps({key: "success"}),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and (tmp_path / "run.json").read_bytes() == before


def test_sealed_functional_existing_shell_step_real_host_only(tmp_path, monkeypatch):
    host = _requirement(
        backend="host",
        selector={"test": "test_host.py", "test_id": None, "k": None},
        minimum_selected=19,
        minimum_executed=19,
    )
    _, args, _, tree = _functional_fixture(tmp_path, monkeypatch, [host])
    proc = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; execute_step_run_sealed_functional',
            "step",
            str(ORCHESTRATOR_STEPS.resolve()),
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run = json.loads((Path(args.log_dir) / "run.json").read_text())
    assert run["functional_execution"]["status"] == "success"
    assert run["tests_passed"] == 19 and run["status"] == "running"
    assert not (tree / "tests/sfpi").exists()


def test_sealed_functional_terminal_timeout_preserves_unexecuted_coverage(
    tmp_path, monkeypatch
):
    device = _requirement(
        selector={"test": "test_device.py", "test_id": None, "k": None}
    )
    host = _requirement(
        index=2,
        backend="host",
        selector={"test": "test_host.py", "test_id": None, "k": None},
    )
    writer, args, manifest, _ = _functional_fixture(
        tmp_path, monkeypatch, [device, host]
    )
    calls = _fake_functional_dispatch(
        monkeypatch,
        writer,
        args,
        manifest,
        outcomes={"executed": 0, "passed": 0, "timed_out": True, "returncode": 124},
    )
    assert writer.cmd_execute_functional(args) == 1
    run = json.loads((Path(args.log_dir) / "run.json").read_text())
    record = run["functional_execution"]
    assert record["leaves"][0]["status"] == "recorded"
    assert record["unstarted"] == [host["requirement_id"]]
    assert "execution_timed_out" in record["error"] and len(_submitted(calls)) == 1
    assert run["status"] == "running"


def test_sealed_functional_old_dispatch_interface_falls_back_before_any_leaf(
    tmp_path, monkeypatch
):
    writer, args, _, _ = _functional_fixture(tmp_path, monkeypatch)
    real_run, seen = subprocess.run, []

    def old_dispatch(argv, **kwargs):
        if argv[0] == "sealed-dispatch-fixture":
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 0, "--arch --test", "")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(writer.subprocess, "run", old_dispatch)
    assert writer.cmd_execute_functional(args) == 20
    assert seen == [["sealed-dispatch-fixture", "--help"]]


def test_sealed_functional_all_architectures_execute_in_manifest_order(
    tmp_path, monkeypatch
):
    requirements = [
        _requirement(
            arch,
            selector={
                "test": "test_device.py",
                "test_id": "test_device.py::test_device",
                "k": "device",
            },
        )
        for arch in ("wormhole", "blackhole")
    ]
    writer, args, manifest, _ = _functional_fixture(tmp_path, monkeypatch, requirements)
    calls = _fake_functional_dispatch(monkeypatch, writer, args, manifest)
    assert writer.cmd_execute_functional(args) == 0
    submitted = _submitted(calls)
    assert [argv[argv.index("--requirement-id") + 1] for argv in submitted] == [
        r["requirement_id"] for r in requirements
    ]
    run = json.loads((Path(args.log_dir) / "run.json").read_text())
    assert run["tests_passed"] == 70
    assert all(
        run["arch_results"][arch]["verdict"] == "SUCCESS"
        for arch in ("wormhole", "blackhole")
    )


def test_sealed_functional_host_only_checks_run_worktree_pointer(tmp_path, monkeypatch):
    host = _requirement(
        backend="host", selector={"test": "test_host.py", "test_id": None, "k": None}
    )
    writer, args, _, _ = _functional_fixture(tmp_path, monkeypatch, [host])
    run_path = Path(args.log_dir) / "run.json"
    run = json.loads(run_path.read_text())
    run["worktree_dir"] = str(tmp_path / "foreign")
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="worktree identity"):
        writer.cmd_execute_functional(args)


@pytest.mark.parametrize("foreign_attempt", [False, True])
def test_sealed_functional_preexisting_root_host_receipt_identity(
    tmp_path, monkeypatch, foreign_attempt
):
    host = _requirement(
        backend="host", selector={"test": "test_host.py", "test_id": None, "k": None}
    )
    writer, args, manifest, _ = _functional_fixture(tmp_path, monkeypatch, [host])
    assert writer.cmd_execute_functional(args) == 0
    logs = Path(args.log_dir)
    run = json.loads((logs / "run.json").read_text())
    receipt = Path(run.pop("functional_execution")["leaves"][0]["result"])
    root_receipt = logs / "verification-results/manual-host.json"
    receipt.rename(root_receipt)
    assert writer._load_verification_result(root_receipt)["version"] == 3
    if foreign_attempt:
        manifest["attempt_id"] = "attempt-002"
        manifest["manifest_id"] = _content_id(manifest, {"manifest_id"})
        (logs / "required_verification_manifest.json").write_text(json.dumps(manifest))
        run["required_verification"] = {
            key: manifest[key] for key in ("attempt_id", "manifest_id")
        }
        state = json.loads((logs / "state.json").read_text())
        state.update(
            REQUIRED_VERIFICATION_ATTEMPT_ID=manifest["attempt_id"],
            REQUIRED_VERIFICATION_MANIFEST_ID=manifest["manifest_id"],
        )
        (logs / "state.json").write_text(json.dumps(state))
    (logs / "run.json").write_text(json.dumps(run))
    if foreign_attempt:
        assert writer.cmd_execute_functional(args) == 0
        reduction = json.loads((logs / "verification_reduction.json").read_text())
        assert reduction["tests_passed"] == 19
        assert (
            reduction["excluded_results"][0]["reason"]
            == "superseded_or_foreign_attempt"
        )
    else:
        with pytest.raises(ValueError, match="already exist"):
            writer.cmd_execute_functional(args)
        assert len(list((logs / "verification-results").rglob("*.json"))) == 1


@pytest.mark.parametrize(
    ("outcomes", "missing", "expected_class", "repair"),
    [
        (
            {"selected": 48, "executed": 35, "passed": 35, "skipped": 13},
            False,
            "VERIFICATION_PLAN_ERROR",
            True,
        ),
        (
            {"selected": 35, "executed": 35, "passed": 34, "failed": 1},
            False,
            "TESTS_FAILED",
            True,
        ),
        (
            {"selected": 0, "executed": 0, "passed": 0},
            False,
            "MISSING_TEST_COVERAGE",
            True,
        ),
        ({}, True, "ENV_ERROR", False),
        ({}, False, "", False),
    ],
)
def test_sealed_executor_reduction_drives_existing_retry_wrapper(
    tmp_path, monkeypatch, outcomes, missing, expected_class, repair
):
    """The adapter's real reduction, not a hand-built hint, drives retry routing."""
    writer, args, manifest, tree = _functional_fixture(tmp_path, monkeypatch)
    calls = _fake_functional_dispatch(
        monkeypatch, writer, args, manifest, outcomes=outcomes, missing=missing
    )
    success = not missing and not outcomes
    assert writer.cmd_execute_functional(args) == (0 if success else 1)
    logs = Path(args.log_dir)
    reduction = json.loads((logs / "verification_reduction.json").read_text())
    assert reduction["scope"] == "functional"
    assert reduction["success_token"] is None
    state_path = logs / "state.json"
    state = json.loads(state_path.read_text())
    state.update(ISSUE_NUMBER="5", DEBUG_CYCLES=0, MAX_DEBUG_CYCLES=3)
    state_path.write_text(json.dumps(state))
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; execute_step_debug_feedback "sealed execution evidence"',
            "bash",
            str(ORCHESTRATOR_STEPS),
        ],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) is repair, result.stdout + result.stderr
    state = json.loads(state_path.read_text())
    assert state["FAILURE_CLASS"] == expected_class
    context = state["VERIFICATION_RETRY_CONTEXT"]
    assert context["reduction_id"] == reduction["reduction_id"]
    assert context["retry_allowed"] is repair
    assert context["manifest_id"] == manifest["manifest_id"]
    if not success:
        assert len(context["leaves"]) == 1
        assert context["leaves"][0]["backend"] == "silicon"
        assert (
            context["leaves"][0]["requirement_id"]
            == manifest["requirements"][1]["requirement_id"]
        )
    run = json.loads((logs / "run.json").read_text())
    assert run["status"] == "running"
    assert run["current_step"] == ("fix_tests" if repair else "tester")
    assert len(_submitted(calls)) == 1
    before = list(calls)
    assert writer.cmd_execute_functional(args) == (0 if success else 1)
    assert calls == before  # diagnostic routing cannot resubmit the same attempt
