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


def _required_manifest(tmp_path, analysis, plan, *extra, check=True):
    worktree = tmp_path / "worktree"
    llk_tests = worktree / "tt_metal" / "tt-llk" / "tests" / "python_tests"
    llk_tests.mkdir(parents=True, exist_ok=True)
    (llk_tests / "test_reduce.py").write_text("def test_reduce(): pass\n")
    (llk_tests / "perf_reduce.py").write_text("def test_reduce(): pass\n")
    metal = worktree / "tests" / "tt_metal" / "tt_metal" / "llk"
    metal.mkdir(parents=True, exist_ok=True)
    (metal / "test_reduce.cpp").write_text("// test\n")
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
            "a" * 40,
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


def test_required_verification_retains_perf_when_hypothesis_is_refuted(tmp_path):
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
    _, output = _required_manifest(tmp_path, analysis, plan, "--performance-only")
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


def test_audit_finalize_accepts_current_and_rejects_changed_patch(tmp_path):
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


def test_candidate_patch_digest_is_identical_from_llk_subdir_and_repo_root(tmp_path):
    worktree = tmp_path / "worktree"
    llk = worktree / "tt_metal" / "tt-llk"
    llk.mkdir(parents=True)
    (llk / "llk.txt").write_text("base llk\n")
    (worktree / "metal.txt").write_text("base metal\n")
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
    (worktree / "new-untracked.txt").write_text("new candidate input\n")

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

    assert digest(llk) == digest(worktree)
    run_test_source = RUN_TEST.read_text(encoding="utf-8")
    assert "candidate-patch-digest" in run_test_source
    assert "tt-llk-local-patch-v1" not in run_test_source


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


def test_setup_worktree_records_exact_base_before_bootstrap(tmp_path):
    repo = tmp_path / "repo"
    llk_tests = repo / "tt_metal" / "tt-llk" / "tests"
    llk_tests.mkdir(parents=True)
    setup_env = llk_tests / "setup_testing_env.sh"
    setup_env.write_text("#!/bin/bash\nexit 0\n")
    setup_env.chmod(0o755)
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
    patch = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", base, "--", "source.txt"],
        check=True,
        capture_output=True,
    ).stdout
    (repo / "source.txt").write_text("base\n")
    source_run_id = "2026-08-07_issue_5_source"
    source_attempt = "source-attempt"
    source_dir = tmp_path / source_run_id
    source_dir.mkdir()
    (source_dir / "generated.patch").write_bytes(patch)
    checkpoint = {
        "run_id": source_run_id,
        "attempt_id": source_attempt,
        "base_commit": base,
        "artifact_patch": "generated.patch",
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
                "timeout_classification": "outer_timeout",
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
    assert proc.returncode == 0, proc.stderr

    worktree = worktrees / "issue-5-v1"
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

    (source_dir / "generated.patch").write_bytes(patch + b"\nmutation\n")
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
    (source_dir / "generated.patch").write_bytes(patch)

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
