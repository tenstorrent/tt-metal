#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU-only fail-closed contract tests for the TTTv2 vLLM gate tools."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALIDATOR_PATH = HERE / "tttv2_validate_vllm_matrix.py"
RUNNER_PATH = HERE / "tttv2_vllm_hardware_gate_runner.sh"
SPEC = importlib.util.spec_from_file_location("tttv2_validate_vllm_matrix", VALIDATOR_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExpectationsContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.inputs = {}
        for name in ("required_capabilities", "characterization", "performance_floor"):
            path = root / f"{name}.json"
            path.write_text("{}\n")
            self.inputs[name] = {"path": str(path), "sha256": digest(path)}
        self.expectations = {
            "schema_version": 2,
            "model_id": "unit",
            "model": "org/model",
            "architecture": "UnitForCausalLM",
            "generator": "UnitGenerator",
            "canonical_row_ids": ["p150x4_dp1_all"],
            "rows": [
                {
                    "id": "p150x4_dp1_all",
                    "expected_traces": {"prefill": 1, "decode": 1},
                    "expected_program_logs": {"decode_compiles": 1, "sampling_compiles": 1},
                    "performance_thresholds": {
                        "output_throughput": {"direction": "minimum", "value": 10.0},
                        "median_ttft_ms": {"direction": "maximum", "value": 100.0},
                    },
                    "manifest": {
                        "model": "org/model",
                        "platform": "P150x4",
                        "dp": 1,
                        "trace_mode": "decode_only",
                        "trace_region_size": 1,
                        "sample_on_device_mode": "all",
                        "family_var": "LLAMA_VERSION",
                        "family_version": "llama3_8b",
                        "revision": "a" * 40,
                        "tokenizer_revision": "a" * 40,
                        "max_model_len": 4096,
                        "max_num_seqs_per_rank": 32,
                        "async_scheduling": True,
                        "prefix_caching": True,
                        "cache_root": "/tmp/unit-cache",
                        "visible_devices": [0, 1, 4, 5],
                        "context_subcases": [
                            {"kind": "long_prefill", "input_tokens": 4088, "output_tokens": 8},
                            {
                                "kind": "chunked_prefill",
                                "input_tokens": 2048,
                                "output_tokens": 8,
                                "expected_min_chunks": 2,
                            },
                            {
                                "kind": "cached_prefill",
                                "input_tokens": 2048,
                                "output_tokens": 8,
                                "common_prefix_tokens": 1024,
                                "expected_min_cache_hits": 1,
                            },
                        ],
                    },
                }
            ],
            "smoke": {"row_id": "p150x4_dp1_all", "tokens": 8},
            "quality": {
                "quality_tokens": 8,
                "require_pair_review": True,
                "semantic_term_groups": [["rain"], ["road", "street"]],
            },
            "common": {
                "backend": "openai",
                "endpoint": "/v1/completions",
                "input_tokens": 128,
                "output_tokens": 256,
                "num_prompts": 320,
                "max_concurrency": 320,
                "request_rate": "inf",
                "temperature": 0,
            },
            "hf_cache": {
                "hf_home": "/tmp/hf",
                "snapshot": "/tmp/hf/snapshots/" + "a" * 40,
                "ref_path": "/tmp/hf/refs/main",
                "revision": "a" * 40,
                "verified_files": ["config.json"],
            },
            "execution": {
                "python": "/usr/bin/python3",
                "vllm_dir": "/tmp/vllm",
                "server_script": "/tmp/server.py",
                "prompt": "rain on a road",
                "validator": {"path": str(VALIDATOR_PATH), "sha256": digest(VALIDATOR_PATH)},
            },
            "provenance": self.inputs,
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def validate(self, document=None):
        return validator.validate_expectations_document(document or self.expectations, Path("expectations.json"))

    def test_complete_expectations_pass(self) -> None:
        self.assertEqual(self.validate().errors, [])

    def test_quality_cannot_be_optional_or_untyped(self) -> None:
        for value in (False, 1, "true"):
            with self.subTest(value=value):
                document = copy.deepcopy(self.expectations)
                document["quality"]["require_pair_review"] = value
                self.assertTrue(any("require_pair_review" in error for error in self.validate(document).errors))

    def test_semantic_groups_are_strictly_nonempty_strings(self) -> None:
        for groups in ([], [[]], [[""]], [[1]]):
            with self.subTest(groups=groups):
                document = copy.deepcopy(self.expectations)
                document["quality"]["semantic_term_groups"] = groups
                self.assertTrue(any("semantic_term_groups" in error for error in self.validate(document).errors))

    def test_visible_devices_cache_and_context_cases_fail_closed(self) -> None:
        mutations = (
            ("visible_devices", []),
            ("cache_root", "relative/cache"),
            ("context_subcases", self.expectations["rows"][0]["manifest"]["context_subcases"][:-1]),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                document = copy.deepcopy(self.expectations)
                document["rows"][0]["manifest"][key] = value
                self.assertNotEqual(self.validate(document).errors, [])

    def test_unknown_execution_affecting_fields_fail_closed_at_every_level(self) -> None:
        mutations = (
            lambda document: document.__setitem__("surprise", True),
            lambda document: document["execution"].__setitem__("extra", True),
            lambda document: document["execution"]["validator"].__setitem__("extra", True),
            lambda document: document["provenance"]["characterization"].__setitem__("extra", True),
            lambda document: document["quality"].__setitem__("extra", True),
            lambda document: document["smoke"].__setitem__("extra", True),
            lambda document: document["hf_cache"].__setitem__("extra", True),
            lambda document: document["common"].__setitem__("extra", True),
            lambda document: document["rows"][0].__setitem__("extra", True),
            lambda document: document["rows"][0]["manifest"].__setitem__("extra", True),
            lambda document: document["rows"][0]["manifest"]["context_subcases"][0].__setitem__("extra", True),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                document = copy.deepcopy(self.expectations)
                mutate(document)
                self.assertTrue(any("unknown fields" in error for error in self.validate(document).errors))

    def test_provenance_hash_is_checked_against_live_file(self) -> None:
        Path(self.inputs["characterization"]["path"]).write_text('{"changed":true}\n')
        self.assertTrue(any("characterization live hash" in error for error in self.validate().errors))

    def test_performance_thresholds_are_required_and_directional(self) -> None:
        for thresholds in (
            {},
            {"output_throughput": {"direction": "maximum", "value": 10.0}},
            {"median_ttft_ms": {"direction": "minimum", "value": 100.0}},
            {
                "output_throughput": {"direction": "minimum", "value": 10.0},
                "median_ttft_ms": {"direction": "maximum", "value": 100.0},
                "invented": {"direction": "minimum", "value": 1.0},
            },
        ):
            with self.subTest(thresholds=thresholds):
                document = copy.deepcopy(self.expectations)
                document["rows"][0]["performance_thresholds"] = thresholds
                self.assertNotEqual(self.validate(document).errors, [])

    def test_check_expectations_cli(self) -> None:
        path = Path(self.temp.name) / "expectations.json"
        path.write_text(json.dumps(self.expectations))
        result = subprocess.run(
            ["python3", str(VALIDATOR_PATH), "--expectations", str(path), "--check-expectations"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS: expectations schema", result.stdout)

    def test_runner_dry_run_executes_schema_and_renders_frozen_launch_without_artifacts(self) -> None:
        root = Path(self.temp.name)
        vllm_dir = root / "vllm"
        vllm_dir.mkdir()
        subprocess.run(["git", "init", "-q", str(vllm_dir)], check=True)
        server = vllm_dir / "server.py"
        server.write_text("raise SystemExit('dry-run only')\n")

        revision = "a" * 40
        hf_home = root / "hf"
        snapshot = hf_home / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}\n")
        ref_path = hf_home / "refs" / "main"
        ref_path.parent.mkdir(parents=True)
        ref_path.write_text(revision + "\n")

        document = copy.deepcopy(self.expectations)
        document["execution"].update(
            {
                "python": "/usr/bin/python3",
                "vllm_dir": str(vllm_dir),
                "server_script": str(server),
            }
        )
        document["hf_cache"].update(
            {
                "hf_home": str(hf_home),
                "snapshot": str(snapshot),
                "ref_path": str(ref_path),
                "revision": revision,
                "ref": "main",
                "ref_revision": revision,
            }
        )
        document["rows"][0]["manifest"].update(
            {
                "revision": revision,
                "tokenizer_revision": revision,
                "cache_root": str(root / "model-cache"),
            }
        )
        expectations_path = root / "expectations.json"
        expectations_path.write_text(json.dumps(document))
        artifact_root = root / "artifacts"
        environment = dict(os.environ)
        environment.update({"PY": "/usr/bin/python3", "VLLM_DIR": str(vllm_dir)})
        result = subprocess.run(
            [
                "bash",
                str(RUNNER_PATH),
                "--tier",
                "benchmark",
                "--expectations",
                str(expectations_path),
                "--artifact-root",
                str(artifact_root),
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("immutable LaunchSpec", result.stdout)
        self.assertIn('"context_clients"', result.stdout)
        self.assertFalse(artifact_root.exists())


class EvidenceContractTest(unittest.TestCase):
    def test_benchmark_fails_closed_on_floor_or_ceiling_miss(self) -> None:
        count = 2
        result = {
            "backend": "openai",
            "model_id": "org/model",
            "num_prompts": count,
            "max_concurrency": count,
            "request_rate": "inf",
            "completed": count,
            "failed": 0,
            "total_output_tokens": count * 4,
            "errors": [None] * count,
            "output_lens": [4] * count,
            "input_lens": [2] * count,
            "generated_texts": [""] * count,
            "start_times": [0.0] * count,
            "ttfts": [0.1] * count,
            "itls": [[], []],
            "output_throughput": 9.0,
            "median_ttft_ms": 99.0,
        }
        expectations = {
            "model": "org/model",
            "common": {
                "backend": "openai",
                "num_prompts": count,
                "max_concurrency": count,
                "request_rate": "inf",
                "output_tokens": 4,
                "input_tokens": 2,
            },
        }
        row = {
            "id": "row",
            "performance_thresholds": {
                "output_throughput": {"direction": "minimum", "value": 10.0},
                "median_ttft_ms": {"direction": "maximum", "value": 100.0},
            },
        }
        with tempfile.TemporaryDirectory() as raw:
            Path(raw, "result.json").write_text(json.dumps(result))
            validation = validator.Validation()
            validator.validate_benchmark(Path(raw), expectations, row, validation)
            self.assertTrue(any("below required minimum" in error for error in validation.errors))

            result["output_throughput"] = 10.0
            result["median_ttft_ms"] = 101.0
            Path(raw, "result.json").write_text(json.dumps(result))
            validation = validator.Validation()
            validator.validate_benchmark(Path(raw), expectations, row, validation)
            self.assertTrue(any("exceeds required maximum" in error for error in validation.errors))

    def test_cache_fallback_warning_is_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "server.log"
            path.write_text(
                "WARNING Configured TT cache is not writable; using job-local tensor cache /tmp/tttv2_model_cache\n"
            )
            result = validator.Validation()
            validator.scan_logs((path,), result, "row")
            self.assertTrue(any("fallback" in error or "fall" in error for error in result.errors))

    def test_process_spec_rejects_extra_or_non_string_env(self) -> None:
        spec = {"kind": "process", "cwd": "/tmp", "argv": ["/bin/true"], "env": {"PATH": "/bin"}, "extra": True}
        result = validator.Validation()
        validator.validate_process_spec(spec, "process", result)
        self.assertTrue(any("fields must be exactly" in error for error in result.errors))
        spec.pop("extra")
        spec["env"]["BAD"] = 3
        result = validator.Validation()
        validator.validate_process_spec(spec, "process", result)
        self.assertTrue(any("complete string map" in error for error in result.errors))

    def test_live_process_proof_rejects_extra_fields_and_wrong_executable(self) -> None:
        spec = {"kind": "process", "cwd": "/tmp", "argv": ["/bin/true"], "env": {}}
        proof = {
            "pid": 1,
            "pgid": 1,
            "cwd": "/tmp",
            "argv": ["/bin/true"],
            "env": {},
            "executable": "/bin/false",
            "extra": True,
        }
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "proof.json"
            path.write_text(json.dumps(proof))
            result = validator.Validation()
            validator.validate_process_proof(path, spec, result, "client")
            self.assertTrue(any("unknown fields" in error for error in result.errors))
            self.assertTrue(any("live executable" in error and "expected" in error for error in result.errors))

    def test_runner_uses_frozen_specs_env_scrub_lock_and_journal(self) -> None:
        source = RUNNER_PATH.read_text()
        self.assertIn('exec_process_spec "$case_dir/launch.json" server', source)
        self.assertIn('run_process_spec_with_proof "$case_dir/launch.json" client', source)
        self.assertIn('flock -w "${TT_DEVICE_LOCK_TIMEOUT:-600}" 9', source)
        self.assertIn("attempt_journal.jsonl", source)
        self.assertNotIn('VALIDATOR="${VALIDATOR:-', source)
        self.assertNotIn("request_completion()", source)
        self.assertEqual(source.count('execute_http_client_spec "$case_dir/launch.json"'), 1)
        self.assertIn("reviewer must be a non-empty string", source)
        self.assertIn("note must be a non-empty string", source)
        proof_index = source.index('capture_live_process "$SERVER_PID" "$case_dir/live_process.json"')
        pgid_index = source.index('SERVER_PGID="$($PY - "$case_dir/live_process.json"')
        self.assertLess(proof_index, pgid_index)
        self.assertNotIn('ps -o pgid= -p "$SERVER_PID"', source)
        self.assertIn('[[ -z "$state" || "$state" == Z* ]]', source)
        self.assertIn("expected_launch_hash != actual_launch_hash", source)
        self.assertIn('find "$case_dir/context_subcases" -type f -name client.log', source)
        self.assertIn('reset_tt "${CURRENT_CASE:-$ROOT}/trap_reset_after.log"', source)
        self.assertNotIn('-m py_compile "$VALIDATOR"', source)

    def test_live_process_python_heredoc_is_executable_and_shell_helper_is_outside(self) -> None:
        source = RUNNER_PATH.read_text()
        function = source.split("capture_live_process() {", 1)[1]
        heredoc = function.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]

        compile(heredoc, str(RUNNER_PATH), "exec")
        self.assertNotIn("run_process_spec_with_proof()", heredoc)
        self.assertIn("\nrun_process_spec_with_proof() {", function.split("\nPY\n", 1)[1])

    def test_every_python_heredoc_compiles(self) -> None:
        source = RUNNER_PATH.read_text()
        heredocs = re.findall(r"<<'PY'\n(.*?)\nPY\n", source, re.DOTALL)

        self.assertGreater(len(heredocs), 10)
        for index, heredoc in enumerate(heredocs):
            with self.subTest(index=index):
                compile(heredoc, f"{RUNNER_PATH}:heredoc-{index}", "exec")

    def test_context_client_program_is_independently_pinned_by_validator(self) -> None:
        source = RUNNER_PATH.read_text()
        write_launch = source.split("write_launch() {", 1)[1].split("\n}\n", 1)[0]
        heredoc = write_launch.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        tree = ast.parse(heredoc)
        values = [
            node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "context_program" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ]
        self.assertEqual(values, [validator.CONTEXT_CLIENT_PROGRAM])


if __name__ == "__main__":
    unittest.main()
