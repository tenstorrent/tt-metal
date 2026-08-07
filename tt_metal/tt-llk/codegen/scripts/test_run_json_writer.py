# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for run_json_writer.py — dashboard-compatibility schema."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "run_json_writer.py"
RUN_TEST = Path(__file__).parents[2] / ".claude" / "scripts" / "run_test.sh"


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


def test_run_test_isolates_artifacts_by_owner_and_full_source_content(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "artifact-roots.txt"
    pytest_bin = fake_bin / "pytest"
    pytest_bin.write_text(
        "#!/bin/bash\n"
        'collection=""; junit=""; producer=false\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  case "$1" in\n'
        '    --codegen-collection-json) collection="$2"; shift 2 ;;\n'
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
        'if [[ "$producer" == true ]]; then\n'
        '  printf \'%s\\n\' "$TT_LLK_ARTEFACTS_DIR" >> "$CAPTURE"\n'
        '  mkdir -p "$TT_LLK_ARTEFACTS_DIR"\n'
        '  touch "$TT_LLK_ARTEFACTS_DIR/fake-output"\n'
        "fi\n"
    )
    pytest_bin.chmod(0o755)

    def make_worktree(name):
        worktree = tmp_path / name
        test_dir = worktree / "tests" / "python_tests"
        compiler_dir = worktree / "tests" / "sfpi" / "compiler" / "bin"
        source_dir = worktree / "tt_llk_blackhole"
        writer_dir = worktree / "codegen" / "scripts"
        test_dir.mkdir(parents=True)
        compiler_dir.mkdir(parents=True)
        source_dir.mkdir()
        writer_dir.mkdir(parents=True)
        (writer_dir / "run_json_writer.py").symlink_to(SCRIPT)
        (test_dir / "test_fake.py").write_text("def test_fake(): pass\n")
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
