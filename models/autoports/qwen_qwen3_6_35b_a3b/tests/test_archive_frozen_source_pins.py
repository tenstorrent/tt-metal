# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression checks for exact historical source export archival."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.archive_evidence import archive_evidence
from models.autoports.qwen_qwen3_6_35b_a3b.tests.restore_fused_evidence import restore


class FrozenSourcePinsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "endurance").mkdir()
        self.pin_map = self.root / "endurance/archive_frozen_source_pins.json"
        self.name = "endurance/historical.cpp"
        self.raw = b"int value(){return 3;}  \r\n// frozen without newline"
        (self.root / self.name).write_bytes(self.raw)
        self.sha = hashlib.sha256(self.raw).hexdigest()
        self.pins = {self.name: self.sha}
        self.pin_map.write_text(json.dumps(self.pins))
        self.assertNotIn("ttnn", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_exact_bytes_round_trip_and_only_explicit_sources_ignored(self):
        sibling = self.root / "endurance/active.cpp"
        sibling.write_bytes(self.raw)
        python_name = "endurance/historical.py"
        python_raw = b"value= 3\n"
        (self.root / python_name).write_bytes(python_raw)
        self.pins[python_name] = hashlib.sha256(python_raw).hexdigest()
        self.pin_map.write_text(json.dumps(self.pins))
        records = archive_evidence(self.root, fused=True)
        self.assertEqual(set(records), set(self.pins))
        for name, sha in self.pins.items():
            self.assertEqual(records[name]["immutable_source_pin"]["sha256"], sha)
            self.assertEqual(records[name]["sha256"], sha)
        ignore = (self.root / ".gitignore").read_text()
        self.assertIn("/" + self.name + "\n", ignore)
        self.assertNotIn("active.cpp", ignore)
        self.assertNotIn("*.cpp", ignore)
        self.assertNotIn("*.py", ignore)
        self.assertEqual(archive_evidence(self.root, fused=True), records)
        for name in self.pins:
            (self.root / name).unlink()
        self.assertEqual(restore(self.root)["restored_files"], 2)
        self.assertEqual((self.root / self.name).read_bytes(), self.raw)
        self.assertEqual((self.root / python_name).read_bytes(), python_raw)
        self.assertEqual(sibling.read_bytes(), self.raw)

    def test_mutated_source_rejects_before_any_archive_write(self):
        (self.root / "aaa.log").write_bytes(b"earlier ordinary archive input")
        (self.root / self.name).write_bytes(self.raw + b"\n")
        with self.assertRaisesRegex(ValueError, "Frozen source SHA256 mismatch"):
            archive_evidence(self.root, fused=True)
        self.assertFalse(list(self.root.rglob("*.gz")))
        self.assertFalse((self.root / "archive_manifest.json").exists())
        self.assertFalse((self.root / ".gitignore").exists())

    def test_missing_unsafe_duplicate_and_symlink_pins_reject(self):
        invalid = [
            {"endurance/missing.cpp": self.sha},
            {"../outside.cpp": self.sha},
            {"endurance/../outside.cpp": self.sha},
            {"tests/active.py": self.sha},
            {"endurance/raw/export.cpp": self.sha},
            {"endurance/historical.md": self.sha},
            {self.name: "not-a-sha"},
        ]
        for pins in invalid:
            with self.subTest(pins=pins):
                self.pin_map.write_text(json.dumps(pins))
                with self.assertRaises(ValueError):
                    archive_evidence(self.root, fused=True)
        self.pin_map.write_text('{"' + self.name + '":"' + self.sha + '","' + self.name + '":"' + self.sha + '"}')
        with self.assertRaisesRegex(ValueError, "Duplicate frozen source pin"):
            archive_evidence(self.root, fused=True)
        target = self.root / "endurance/link.cpp"
        target.symlink_to(self.root / self.name)
        self.pin_map.write_text(json.dumps({"endurance/link.cpp": self.sha}))
        with self.assertRaisesRegex(ValueError, "symlinked"):
            archive_evidence(self.root, fused=True)
        self.assertFalse(list(self.root.rglob("*.gz")))
        self.assertFalse((self.root / "archive_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
