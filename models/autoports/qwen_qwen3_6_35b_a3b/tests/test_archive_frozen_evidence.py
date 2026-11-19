# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only byte preservation checks; run directly with unittest, not root pytest."""

import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.archive_evidence import (
    PART_BYTES,
    archive_evidence,
    evidence_sources,
    frozen_text_format_risks,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tests.restore_fused_evidence import restore


class FrozenEvidenceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "endurance").mkdir()
        self.assertNotIn("ttnn", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def put(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def test_byte_exact_round_trip_and_scoped_ignores(self):
        frozen = {
            "endurance/no_newline.py.snapshot": b"def example():\n    return 3",
            "endurance/trailing.py.snapshot": b"kernel = 'payload'  \r\n\t \r\n",
            "endurance/extra_eof.snapshot": b"x = 1\n\n",
            "endurance/raw.diff": b" context line \n+new line\n",
            "endurance/kernel.disasm.txt": b"0000: op\t  \n",
            "endurance/odd[part] name.snapshot": b"frozen",
        }
        ordinary = {
            "README.md": b"ordinary docs  ",
            "endurance/live.py": b"x = 1  ",
            "endurance/ordinary.txt": b"ordinary report  ",
            "endurance/clean.py.snapshot": b"x = 1\n",
            "endurance/binary.snapshot": b"\x00\xff",
            "endurance/empty.snapshot": b"",
        }
        for name, raw in {**frozen, **ordinary}.items():
            self.put(name, raw)
        records = archive_evidence(self.root, fused=True)
        self.assertEqual(set(records), set(frozen))
        first_archives = {}
        for name, raw in frozen.items():
            entry = records[name]
            self.assertEqual(entry["bytes"], len(raw))
            self.assertEqual(entry["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertTrue(entry["immutable_text_format_risk"])
            compressed = (self.root / entry["archive"]).read_bytes()
            first_archives[name] = compressed
            self.assertEqual(gzip.decompress(compressed), raw)
            self.assertEqual((self.root / name).read_bytes(), raw)
        ignore = (self.root / ".gitignore").read_text()
        self.assertIn("/endurance/odd\\[part\\]\\ name.snapshot\n", ignore)
        self.assertNotIn("*.snapshot", ignore)
        for name in ordinary:
            self.assertNotIn("/" + name + "\n", ignore)
        self.assertEqual(archive_evidence(self.root, fused=True), records)
        self.assertEqual((self.root / ".gitignore").read_text(), ignore)
        for name, entry in records.items():
            self.assertEqual((self.root / entry["archive"]).read_bytes(), first_archives[name])
            (self.root / name).unlink()
        self.assertEqual(restore(self.root, verify_only=True)["restored_files"], 0)
        self.assertEqual(restore(self.root)["restored_files"], len(frozen))
        for name, raw in {**frozen, **ordinary}.items():
            self.assertEqual((self.root / name).read_bytes(), raw)
        self.assertEqual(restore(self.root)["restored_files"], 0)

    def test_conflicting_restoration_rejects_without_normalizing(self):
        path = self.put("endurance/exact.snapshot", b"original  ")
        archive_evidence(self.root, fused=True)
        path.write_bytes(b"original\n")
        with self.assertRaisesRegex(ValueError, "Existing evidence differs"):
            restore(self.root)
        self.assertEqual(path.read_bytes(), b"original\n")

    def test_large_json_and_raw_exclusions_remain(self):
        raw = json.dumps({"payload": "x" * PART_BYTES}).encode() + b"\n"
        large = self.put("large.json", raw)
        self.put("endurance/notes.json", b"{} ")
        self.put("raw/frozen.snapshot", b"not selected")
        self.put("oracle_cache/frozen.snapshot", b"not selected")
        self.put("pytest/frozen.snapshot", b"not selected")
        self.assertEqual(evidence_sources(self.root, fused=True), [large])
        records = archive_evidence(self.root, fused=True)
        self.assertEqual(set(records), {"large.json"})
        self.assertNotIn("immutable_text_format_risk", records["large.json"])
        self.assertIn("/large.json\n", (self.root / ".gitignore").read_text())
        large.unlink()
        self.assertEqual(restore(self.root)["restored_files"], 1)
        self.assertEqual(large.read_bytes(), raw)

    def test_default_hook_risk_detection(self):
        for i, (raw, expected) in enumerate(
            [
                (b"normal\n", {}),
                (b"normal\r\n", {}),
                (b"", {}),
                (b"\n", {"trailing_whitespace": False, "end_of_file": True}),
                (b"missing", {"trailing_whitespace": False, "end_of_file": True}),
                (b"tail\t\n", {"trailing_whitespace": True, "end_of_file": False}),
                (b"tail\n\n", {"trailing_whitespace": False, "end_of_file": True}),
            ]
        ):
            self.assertEqual(frozen_text_format_risks(self.put(f"case{i}.snapshot", raw)), expected)
        self.assertNotIn("ttnn", sys.modules)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
