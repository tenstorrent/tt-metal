# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU checks for admitting measured canonical-reference evidence."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location("trace_audit", Path(__file__).with_name("audit_vllm_trace_events.py"))
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ReferenceChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.args = SimpleNamespace(
            client=root / "failed.json",
            events=root / "events.jsonl",
            runtime=root / "runtime.json",
            canonical_reference=root / "reference.json",
        )
        self.client = dict(status="failed", exact_greedy_differences=[dict(label="mixed_513", first_difference=60)])
        self.args.client.write_text(json.dumps(self.client))
        self.args.events.write_text("synthetic events")
        self.args.runtime.write_text("synthetic runtime")
        self.reference = dict(
            status="verified",
            scope="Synthetic fixture only",
            provenance_limit="Synthetic fixture only",
            original_cross_shape_result=dict(
                status="failed", artifact=str(self.args.client), differences=self.client["exact_greedy_differences"]
            ),
            canonical_references=[dict(exact_output_tokens=96), dict(exact_output_tokens=96)],
            artifacts={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (self.args.client, self.args.events, self.args.runtime)
            },
        )
        self.save()

    def save(self):
        self.args.canonical_reference.write_text(json.dumps(self.reference))

    def test_original_pass(self):
        self.args.canonical_reference = None
        self.assertEqual(AUDIT.correctness_basis(self.args, dict(status="pass"))["original_status"], "pass")

    def test_failure_without_reference_rejected(self):
        self.args.canonical_reference = None
        with self.assertRaises(AssertionError):
            AUDIT.correctness_basis(self.args, self.client)

    def test_original_failure_preserved(self):
        result = AUDIT.correctness_basis(self.args, self.client)
        self.assertEqual(result["original_status"], "failed")
        self.assertEqual(json.loads(self.args.client.read_text()), self.client)

    def test_changed_artifact_rejected(self):
        self.args.events.write_text("changed")
        with self.assertRaises(AssertionError):
            AUDIT.correctness_basis(self.args, self.client)

    def test_unverified_reference_rejected(self):
        self.reference["status"] = "failed"
        self.save()
        with self.assertRaises(AssertionError):
            AUDIT.correctness_basis(self.args, self.client)

    def test_different_failure_rejected(self):
        self.reference["original_cross_shape_result"]["differences"] = []
        self.save()
        with self.assertRaises(AssertionError):
            AUDIT.correctness_basis(self.args, self.client)


if __name__ == "__main__":
    unittest.main()
