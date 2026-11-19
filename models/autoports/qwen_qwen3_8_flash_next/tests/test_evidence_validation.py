# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only regression checks; run with python -m unittest (no device conftest)."""

import ast
import json
import lzma
import tempfile
import unittest
from pathlib import Path

from .summarize_functional_evidence import evaluate_gate
from .validate_trace_evidence import checked_archive, digest, source_closure


class EvidenceValidation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "tt").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "tt/model.py").write_text("runtime = 1\n")
        (self.root / "tests/capture.py").write_text("capture = 1\n")
        self.required = ["tt/model.py", "tests/capture.py"]
        self.record = {"source_sha256": {name: digest(self.root / name) for name in self.required}}

    def test_unrelated_tests_do_not_invalidate_runtime_closure(self):
        (self.root / "tests/unrelated.py").write_text("new_test = True\n")
        self.record["source_sha256"]["tests/historical_unrelated.py"] = "old hash"
        self.assertEqual(set(source_closure(self.root, self.record, self.required)), set(self.required))

    def test_runtime_addition_or_change_is_rejected(self):
        extra = self.root / "tt/new_kernel.cpp"
        extra.write_text("// new runtime\n")
        with self.assertRaisesRegex(ValueError, "inventory"):
            source_closure(self.root, self.record, self.required)
        extra.unlink()
        (self.root / "tt/model.py").write_text("runtime = 2\n")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            source_closure(self.root, self.record, self.required)

    def test_selected_harness_change_is_rejected(self):
        (self.root / "tests/capture.py").write_text("capture = 2\n")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            source_closure(self.root, self.record, self.required)

    def test_archive_tamper_and_original_disagreement_are_rejected(self):
        path = self.root / "report.csv.xz"
        content = b"native,report\n1,2\n"
        path.write_bytes(lzma.compress(content))
        sha = digest(path)
        self.assertEqual(checked_archive(path, sha), content)
        path.with_suffix("").write_bytes(b"different\n")
        with self.assertRaisesRegex(ValueError, "Archive/original mismatch"):
            checked_archive(path, sha)
        path.with_suffix("").unlink()
        path.write_bytes(lzma.compress(b"other valid archive"))
        with self.assertRaisesRegex(ValueError, "Archive identity changed"):
            checked_archive(path, sha)

    def test_additional_artifact_is_required_not_advisory(self):
        artifact = self.root / "profile.json"
        artifact.write_text('{"passed": true}\n')
        log = self.root / "run.log"
        log.write_text("RUN_RECORDED_FINISHED exit_code=0\n")
        record = {
            **self.record,
            "status": "finished",
            "exit_code": 0,
            "cwd": str(self.root),
            "argv": ["--output", str(artifact)],
            "log": str(log),
            "log_sha256": digest(log),
        }
        (self.root / "run.json").write_text(json.dumps(record))
        gate = {
            "id": "host_test",
            "scope": "unit fixture",
            "artifact": "profile.json",
            "provenance": "run.json",
            "required_sources": self.required,
            "checks": [{"pointer": "/passed", "equals": True}],
            "additional_required_artifacts": ["required.txt"],
        }
        for content in (None, ""):
            if content is not None:
                (self.root / "required.txt").write_text(content)
            result = evaluate_gate(self.root, gate)
            self.assertEqual(result["status"], "unmet")
            self.assertIn("Additional required artifact", result["reason"])
        (self.root / "required.txt").write_text("actual report\n")
        result = evaluate_gate(self.root, gate)
        self.assertEqual(result["status"], "declared_checks_satisfied_review_required")
        self.assertEqual(result["additional_required_artifacts"]["required.txt"], digest(self.root / "required.txt"))

    def test_public_capture_storage_is_not_read_before_replay(self):
        path = Path(__file__).with_name("probe_public_continuation.py")

        def pre_replay_reads(source):
            calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]

            def named(node, name):
                return isinstance(node.func, ast.Attribute) and node.func.attr == name

            capture_end = [node.lineno for node in calls if named(node, "end_trace_capture")]
            self.assertEqual(len(capture_end), 1)
            first_replay = min(
                node.lineno for node in calls if named(node, "execute_trace") and node.lineno > capture_end[0]
            )
            return [
                node.lineno for node in calls if named(node, "to_torch") and capture_end[0] < node.lineno < first_replay
            ]

        source = path.read_text()
        self.assertEqual(pre_replay_reads(source), [])
        # The prior defect must be detected even though its readback is discarded.
        mutated = source.replace(
            "acknowledge_corruptible(output)\n", "acknowledge_corruptible(output)\n            ttnn.to_torch(output)\n"
        )
        self.assertEqual(len(pre_replay_reads(mutated)), 1)

    def test_public_v7_bindings_reject_old_or_readback_schema(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "doc/functional_decoder/final_gate_bindings.json").read_text())
        public = [gate for gate in manifest["gates"] if "public_continuation_layer" in (gate.get("artifact") or "")]
        self.assertEqual(len(public), 6)
        expected = {"/capture_outputs_acknowledged": 2, "/capture_output_readbacks_before_first_replay": 0}
        for gate in public:
            self.assertTrue(gate["artifact"].endswith("_v7.json"))
            checks = {check["pointer"]: check.get("equals") for check in gate["checks"]}
            self.assertNotIn("/capture_outputs_consumed_before_replay", checks)
            for pointer, value in expected.items():
                self.assertEqual(checks[pointer], value)
        artifact = self.root / "profile.json"
        log = self.root / "run.log"
        log.write_text("RUN_RECORDED_FINISHED exit_code=0\n")
        record = {
            **self.record,
            "status": "finished",
            "exit_code": 0,
            "cwd": str(self.root),
            "argv": ["--output", str(artifact)],
            "log": str(log),
            "log_sha256": digest(log),
        }
        (self.root / "run.json").write_text(json.dumps(record))
        gate = {
            "id": "capture_lifetime",
            "scope": "host schema control",
            "artifact": "profile.json",
            "provenance": "run.json",
            "required_sources": self.required,
            "checks": [{"pointer": key, "equals": value} for key, value in expected.items()],
        }
        for document in (
            {"capture_outputs_consumed_before_replay": 2},
            {"capture_outputs_acknowledged": 2, "capture_output_readbacks_before_first_replay": 2},
            {"capture_outputs_acknowledged": 0, "capture_output_readbacks_before_first_replay": 0},
        ):
            artifact.write_text(json.dumps(document))
            self.assertEqual(evaluate_gate(self.root, gate)["status"], "unmet")
        artifact.write_text(json.dumps({key[1:]: value for key, value in expected.items()}))
        self.assertEqual(evaluate_gate(self.root, gate)["status"], "declared_checks_satisfied_review_required")


if __name__ == "__main__":
    unittest.main()
