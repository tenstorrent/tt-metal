# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only adversarial checks for the bounded source compatibility verifier."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import _verify_source


def digest(data):
    return hashlib.sha256(data).hexdigest()


class SourceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = b"def product(a, b):\n    return a*b\n"
        self.new = b"def product(a, b):\n    return a * b\n"
        self.source = self.root / "probe.py"
        self.archive = self.root / "probe.py.snapshot"
        self.source.write_bytes(self.new)
        self.archive.write_bytes(self.old)
        self.proof = self.root / "proof.json"
        self.proof.write_text(
            json.dumps(
                dict(
                    status="reviewed_inverse_delta_pass",
                    path="probe.py",
                    historical_sha256=digest(self.old),
                    current_sha256=digest(self.new),
                    arithmetic_ast_unchanged=True,
                    kernel_literals_unchanged=True,
                )
            )
        )
        self.manifest = self.root / "manifest.json"
        self.entry = dict(
            path="probe.py",
            historical_sha256=digest(self.old),
            current_sha256=digest(self.new),
            archive="probe.py.snapshot",
            proof="proof.json",
            proof_sha256=digest(self.proof.read_bytes()),
        )
        self.manifest.write_text(json.dumps(dict(schema_version=1, entries=[self.entry])))
        self.manifest_sha = digest(self.manifest.read_bytes())

    def verify(self):
        return _verify_source(
            self.source,
            digest(self.old),
            root=self.root,
            manifest_path=self.manifest,
            manifest_sha256=self.manifest_sha,
        )

    def reject(self, expected):
        try:
            self.verify()
        except ValueError as error:
            self.assertIn(expected, str(error))
        else:
            self.fail("Unapproved source was accepted")

    def test_exact_pair_reports_actual_bytes(self):
        binding = self.verify()
        self.assertEqual(binding["actual_sha256"], digest(self.new))
        self.assertNotEqual(binding["actual_sha256"], binding["historical_sha256"])

    def test_original_bytes_need_no_compatibility_manifest(self):
        self.source.write_bytes(self.old)
        self.manifest.unlink()
        self.assertEqual(self.verify()["compatibility"], "exact")

    def test_arithmetic_change_rejected(self):
        self.source.write_bytes(self.new.replace(b"a * b", b"a + b"))
        self.reject("Unapproved source bytes")

    def test_even_unreviewed_whitespace_rejected(self):
        self.source.write_bytes(self.new + b"\n")
        self.reject("Unapproved source bytes")

    def test_manifest_tamper_rejected(self):
        self.manifest.write_text(self.manifest.read_text() + "\n")
        self.reject("manifest hash mismatch")

    def test_proof_tamper_rejected(self):
        self.proof.write_text(self.proof.read_text() + "\n")
        self.reject("proof hash mismatch")

    def test_historical_snapshot_tamper_rejected(self):
        self.archive.write_bytes(self.new)
        self.reject("snapshot hash mismatch")

    def test_duplicate_approved_pair_rejected(self):
        self.manifest.write_text(json.dumps(dict(schema_version=1, entries=[self.entry, self.entry])))
        self.manifest_sha = digest(self.manifest.read_bytes())
        self.reject("Unapproved source bytes")


if __name__ == "__main__":
    unittest.main()
