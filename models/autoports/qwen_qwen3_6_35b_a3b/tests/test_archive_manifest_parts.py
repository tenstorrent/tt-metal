# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only manifest partition/validation tests; run with unittest, not root pytest."""

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
    write_archive_manifest,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tests.restore_fused_evidence import read_archive_manifest, restore


class ManifestPartsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "endurance").mkdir()
        self.raw = {}
        for i in range(24):
            name = f"endurance/frozen_generated_exact_source_{i:03d}.py.snapshot"
            raw = f"x = {i}  \n# no terminal newline".encode()
            (self.root / name).write_bytes(raw)
            self.raw[name] = raw
        self.records = archive_evidence(self.root, fused=True)
        self.write()
        self.assertNotIn("ttnn", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def write(self):
        write_archive_manifest(self.root, self.records, part_bytes=1800)
        self.index = json.loads((self.root / "archive_manifest.json").read_text())

    def save_index(self):
        (self.root / "archive_manifest.json").write_text(json.dumps(self.index) + "\n")

    def replace_part(self, number, payload, count):
        part = self.index["manifest_parts"][number]
        (self.root / part["path"]).write_bytes(payload)
        part.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), file_count=count)
        self.save_index()

    def test_bounded_deterministic_exact_restore(self):
        self.assertNotIn("files", self.index)
        self.assertEqual(self.index["file_count"], len(self.records))
        self.assertGreater(len(self.index["manifest_parts"]), 1)
        paths = [self.root / "archive_manifest.json"] + [self.root / p["path"] for p in self.index["manifest_parts"]]
        first = {p.name: p.read_bytes() for p in paths}
        self.assertTrue(all(len(payload) <= 1800 for payload in first.values()))
        write_archive_manifest(self.root, dict(reversed(list(self.records.items()))), part_bytes=1800)
        self.assertEqual({p.name: p.read_bytes() for p in paths}, first)
        self.assertEqual(read_archive_manifest(self.root), self.records)
        for name in self.records:
            (self.root / name).unlink()
        self.assertEqual(restore(self.root, verify_only=True)["restored_files"], 0)
        self.assertEqual(restore(self.root)["restored_files"], len(self.records))
        self.assertEqual(restore(self.root)["restored_files"], 0)
        for name, raw in self.raw.items():
            self.assertEqual((self.root / name).read_bytes(), raw)

    def test_tampered_missing_counts_and_part_order_reject_before_restore(self):
        missing_original = self.root / next(iter(self.records))
        missing_original.unlink()
        for change in ("tamper", "missing", "part_count", "total_count", "duplicate", "gap", "unsafe"):
            with self.subTest(change=change):
                self.write()
                part = self.index["manifest_parts"][0]
                path = self.root / part["path"]
                if change == "tamper":
                    path.write_bytes(path.read_bytes() + b" ")
                elif change == "missing":
                    path.unlink()
                elif change == "part_count":
                    part["file_count"] += 1
                    self.save_index()
                elif change == "total_count":
                    self.index["file_count"] += 1
                    self.save_index()
                else:
                    part["path"] = {
                        "duplicate": self.index["manifest_parts"][1]["path"],
                        "gap": "archive_manifest.part900.json",
                        "unsafe": "../archive_manifest.part000.json",
                    }[change]
                    self.save_index()
                with self.assertRaises(ValueError):
                    restore(self.root)
                self.assertFalse(missing_original.exists())

    def test_duplicate_json_keys_cross_part_paths_and_unsafe_evidence_reject(self):
        for change in ("json_duplicate", "cross_part_duplicate", "unsafe_evidence"):
            with self.subTest(change=change):
                self.write()
                first = json.loads((self.root / self.index["manifest_parts"][0]["path"]).read_text())["files"]
                name, entry = next(iter(first.items()))
                if change == "json_duplicate":
                    pair = json.dumps(name) + ":" + json.dumps(entry)
                    self.replace_part(0, ('{"files":{' + pair + "," + pair + "}}").encode(), 1)
                elif change == "cross_part_duplicate":
                    second = json.loads((self.root / self.index["manifest_parts"][1]["path"]).read_text())["files"]
                    second[name] = entry
                    self.replace_part(1, json.dumps({"files": second}).encode(), len(second))
                else:
                    del first[name]
                    first["../outside"] = entry
                    self.replace_part(0, json.dumps({"files": first}).encode(), len(first))
                with self.assertRaises(ValueError):
                    read_archive_manifest(self.root)

    def test_legacy_restore_and_safe_obsolete_part_cleanup(self):
        generated = self.root / "archive_manifest.part999.json"
        generated.write_bytes(b"x" * (PART_BYTES + 1))
        notes = self.root / "archive_manifest.part-notes.json"
        notes.write_text('{"ordinary": true}\n')
        nested = self.root / "endurance/archive_manifest.part999.json"
        nested.write_text('{"historical": true}\n')
        target = self.root / "untouched.txt"
        target.write_text("payload\n")
        symlink = self.root / "archive_manifest.part888.json"
        symlink.symlink_to(target)
        self.assertNotIn(generated, evidence_sources(self.root, fused=True))
        write_archive_manifest(self.root, self.records, part_bytes=100000)
        self.assertIn("files", json.loads((self.root / "archive_manifest.json").read_text()))
        self.assertFalse(generated.exists())
        self.assertTrue(notes.exists() and nested.exists() and symlink.is_symlink())
        self.assertEqual(target.read_text(), "payload\n")
        self.assertEqual(read_archive_manifest(self.root), self.records)
        for name in self.records:
            (self.root / name).unlink()
        self.assertEqual(restore(self.root)["restored_files"], len(self.records))
        for name, raw in self.raw.items():
            self.assertEqual((self.root / name).read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
