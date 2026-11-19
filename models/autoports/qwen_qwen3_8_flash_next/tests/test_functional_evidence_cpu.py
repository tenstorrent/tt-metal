# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only controls for inventory identity and explicit gate evaluation."""

import hashlib
import json
import lzma
import sys
import tempfile
import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests.summarize_functional_evidence import (
    evaluate_gate,
    evaluate_gates,
    file_digest,
    provenance_summary,
    verify_archive,
)


class EvidenceInventoryTests(unittest.TestCase):
    def test_archive_parts_and_stale_original_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = b'{"passed": false}\n'
            packed = lzma.compress(original)
            (root / "part0").write_bytes(packed[:16])
            (root / "part1").write_bytes(packed[16:])
            entry = {
                "source": "result.json",
                "archive_parts": ["part0", "part1"],
                "source_bytes": len(original),
                "archive_bytes": len(packed),
                "source_sha256": hashlib.sha256(original).hexdigest(),
                "archive_sha256": hashlib.sha256(packed).hexdigest(),
            }
            self.assertEqual(verify_archive(root, entry)["original_status"], "absent")
            (root / "result.json").write_bytes(b'{"passed": true}\n')
            verified = verify_archive(root, entry)
            self.assertEqual(verified["integrity"], "verified")
            self.assertEqual(verified["original_status"], "differs_from_archive")
            (root / "part0").write_bytes(b"bad")
            self.assertEqual(verify_archive(root, entry)["integrity"], "failed")

    def test_current_stale_failed_and_unbound_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tt").mkdir()
            source = root / "tt/decoder.py"
            source.write_text("version = 1\n")
            (root / "output.json").write_text('{"passed": true, "pcc": 0.996}')
            (root / "run.log").write_text("finished\n")
            record = {
                "status": "finished",
                "exit_code": 0,
                "cwd": str(root),
                "argv": ["probe.py", "--output", str(root / "output.json")],
                "log": "run.log",
                "log_sha256": file_digest(root / "run.log"),
                "source_sha256": {"tt/decoder.py": file_digest(source)},
            }
            provenance = root / "provenance.json"
            provenance.write_text(json.dumps(record))
            gate = {
                "id": "pcc_acceptance",
                "artifact": "output.json",
                "provenance": "provenance.json",
                "scope": "explicit single-row fixture",
                "required_sources": ["tt/decoder.py"],
                "checks": [{"pointer": "/passed", "equals": True}, {"pointer": "/pcc", "ge": 0.995}],
            }
            self.assertEqual(evaluate_gate(root, gate)["status"], "declared_checks_satisfied_review_required")
            self.assertEqual(evaluate_gate(root, {"id": "unbound"})["status"], "needs_review")
            (root / "output.json").write_text('{"passed": true, "pcc": 0.994}')
            self.assertEqual(evaluate_gate(root, gate)["status"], "unmet")
            source.write_text("version = 2\n")
            self.assertIn("stale", evaluate_gate(root, gate)["reason"])
            current = {"tt/decoder.py": file_digest(source)}
            summary = provenance_summary(root, provenance, current)
            self.assertEqual(summary["runtime_source_status"], "mismatch")
            self.assertEqual(summary["recorded_exit_code"], 0)
            self.assertFalse(summary["acceptance_inferred"])
        self.assertNotIn("ttnn", sys.modules)


class CompositeGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "source.py").write_text("version = 1\n")
        (self.root / "output.json").write_text('{"passed":true,"pcc":0.996}')
        (self.root / "run.log").write_text("finished\n")
        record = {
            "status": "finished",
            "exit_code": 0,
            "cwd": str(self.root),
            "argv": ["probe.py", "--output", "output.json"],
            "log": "run.log",
            "log_sha256": file_digest(self.root / "run.log"),
            "source_sha256": {"source.py": file_digest(self.root / "source.py")},
        }
        (self.root / "provenance.json").write_text(json.dumps(record))
        self.leaf = {
            "id": "leaf",
            "artifact": "output.json",
            "provenance": "provenance.json",
            "scope": "One real checked row, not all contexts",
            "required_sources": ["source.py"],
            "checks": [{"pointer": "/passed", "equals": True}, {"pointer": "/pcc", "ge": 0.995}],
            "additional_review": ["A controlled finding still requires review"],
        }

    @staticmethod
    def composite(name="combined", dependencies=None):
        return {
            "id": name,
            "kind": "all_of",
            "scope": "Explicit combined scope",
            "depends_on": ["leaf"] if dependencies is None else dependencies,
        }

    def test_nested_success_preserves_review_and_scope(self):
        first = self.composite()
        first["review_obligations"] = ["No capability reduction is authorized"]
        result = evaluate_gates(self.root, [self.composite("outer", ["combined"]), first, self.leaf])
        outer = result["outer"]
        self.assertEqual(outer["status"], "declared_checks_satisfied_review_required")
        self.assertFalse(outer["acceptance_inferred"])
        self.assertEqual(outer["stage_acceptance"], "not_evaluated_review_required")
        self.assertEqual(outer["dependency_scopes"]["leaf"], self.leaf["scope"])
        texts = {row["text"] for row in outer["review_obligations"]}
        self.assertIn("A controlled finding still requires review", texts)
        self.assertIn("No capability reduction is authorized", texts)
        self.assertEqual(result["leaf"]["checks"], evaluate_gate(self.root, self.leaf)["checks"])

    def test_unknown_empty_duplicate_and_malformed_dependencies(self):
        for dependencies in [[], ["missing"], ["leaf", "leaf"], "leaf", [None], [""]]:
            with self.subTest(dependencies=dependencies):
                result = evaluate_gates(self.root, [self.leaf, self.composite(dependencies=dependencies)])
                self.assertEqual(result["combined"]["status"], "unmet")

    def test_cycles_fail_without_hiding_other_leaf(self):
        for gates in [
            [self.composite("a", ["a"])],
            [self.composite("a", ["b"]), self.composite("b", ["a"])],
        ]:
            result = evaluate_gates(self.root, [self.leaf] + gates)
            self.assertEqual(result["leaf"]["status"], "declared_checks_satisfied_review_required")
            for gate in gates:
                self.assertEqual(result[gate["id"]]["status"], "unmet")

    def test_failed_checks_stale_source_and_unbound_leaf_propagate(self):
        for failure in ["pcc", "source", "unbound"]:
            with self.subTest(failure=failure):
                (self.root / "output.json").write_text('{"passed":true,"pcc":0.996}')
                (self.root / "source.py").write_text("version = 1\n")
                leaf = dict(self.leaf)
                if failure == "pcc":
                    (self.root / "output.json").write_text('{"passed":true,"pcc":0.994}')
                elif failure == "source":
                    (self.root / "source.py").write_text("version = 2\n")
                else:
                    leaf.pop("checks")
                result = evaluate_gates(self.root, [leaf, self.composite()])
                self.assertEqual(result["combined"]["status"], "unmet")
                self.assertEqual(result["combined"]["unmet_dependencies"], ["leaf"])

    def test_inspection_review_and_commit_stay_pending(self):
        for kind in ["inspection_closure", "post_review_record", "post_review_commit_record"]:
            with self.subTest(kind=kind):
                pending = {**self.leaf, "id": "pending", "kind": kind, "depends_on": ["leaf"]}
                result = evaluate_gates(self.root, [self.leaf, pending, self.composite(dependencies=["pending"])])
                self.assertEqual(result["pending"]["status"], "needs_review")
                self.assertEqual(result["combined"]["status"], "unmet")

    def test_unknown_kind_mixed_validation_and_duplicate_ids_reject(self):
        for kind in ["automatic_acceptance", []]:
            with self.subTest(kind=kind):
                result = evaluate_gates(self.root, [{**self.leaf, "kind": kind}])
                self.assertEqual(result["leaf"]["status"], "unmet")
        result = evaluate_gates(self.root, [self.leaf, {**self.composite(), "artifact": "output.json"}])
        self.assertEqual(result["combined"]["status"], "unmet")
        with self.assertRaises(ValueError):
            evaluate_gates(self.root, [self.leaf, dict(self.leaf)])

    def test_descriptive_leaf_dependencies_do_not_replace_checks(self):
        leaf = {**self.leaf, "depends_on": ["unbound_descriptive_context"]}
        self.assertEqual(
            evaluate_gates(self.root, [leaf])["leaf"]["status"], "declared_checks_satisfied_review_required"
        )


if __name__ == "__main__":
    unittest.main()
