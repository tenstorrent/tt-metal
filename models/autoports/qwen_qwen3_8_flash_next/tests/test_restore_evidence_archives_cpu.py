# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only restore integrity/path tests using temporary synthetic archives."""

import copy
import hashlib
import importlib.util
import json
import lzma
import sys
import tempfile
import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests.archive_manifest import write_archive_manifest

spec = importlib.util.spec_from_file_location(
    "restore_under_test", Path(__file__).with_name("restore_evidence_archives.py")
)
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class RestoreEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.raw = b'{"cwd":"/original/recorded/repo","status":"finished"}\n' * 100
        self.packed = lzma.compress(self.raw, check=lzma.CHECK_CRC64)
        self.entry = {
            "source": "provenance/original.json",
            "source_sha256": sha(self.raw),
            "source_bytes": len(self.raw),
            "archive_sha256": sha(self.packed),
            "archive_bytes": len(self.packed),
            "archive_parts": [],
        }
        cut = len(self.packed) // 2
        for index, data in enumerate((self.packed[:cut], self.packed[cut:])):
            name = f"payload.part{index}"
            (self.root / name).write_bytes(data)
            self.entry["archive_parts"].append({"path": name, "sha256": sha(data), "bytes": len(data)})

    def tearDown(self):
        self.directory.cleanup()
        self.assertNotIn("ttnn", sys.modules)

    def test_multipart_preserves_provenance_bytes(self):
        result = restore.restore_entry(self.root, self.entry)
        self.assertEqual(result["status"], "restored")
        self.assertEqual((self.root / self.entry["source"]).read_bytes(), self.raw)

    def test_identical_existing_preserves_mtime(self):
        restore.restore_entry(self.root, self.entry)
        path = self.root / self.entry["source"]
        before = path.stat().st_mtime_ns
        self.assertEqual(restore.restore_entry(self.root, self.entry)["status"], "existing_identical")
        self.assertEqual(path.stat().st_mtime_ns, before)

    def test_conflict_never_overwritten(self):
        path = self.root / self.entry["source"]
        path.parent.mkdir()
        path.write_bytes(b"different")
        with self.assertRaises(ValueError):
            restore.restore_entry(self.root, self.entry)
        self.assertEqual(path.read_bytes(), b"different")

    def test_verify_only_writes_no_original(self):
        self.assertEqual(
            restore.restore_entry(self.root, self.entry, True)["status"],
            "verified_only",
        )
        self.assertFalse((self.root / "provenance").exists())

    def test_corrupt_part_or_source_hash_never_publishes(self):
        for key in ("source_sha256", "archive_sha256"):
            entry = copy.deepcopy(self.entry)
            entry[key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                restore.restore_entry(self.root, entry)
        (self.root / "payload.part1").write_bytes(b"corrupted")
        with self.assertRaises(ValueError):
            restore.restore_entry(self.root, self.entry)
        self.assertFalse((self.root / self.entry["source"]).exists())

    def test_path_escape_and_absolute_paths_rejected(self):
        for name in ("../escape", "/tmp/escape", "safe/../../escape", ".", ""):
            entry = copy.deepcopy(self.entry)
            entry["source"] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                restore.restore_entry(self.root, entry)
        entry = copy.deepcopy(self.entry)
        entry["archive_parts"][0]["path"] = "../outside"
        with self.assertRaises(ValueError):
            restore.restore_entry(self.root, entry)

    def test_symlink_target_or_parent_rejected(self):
        (self.root / "actual").mkdir()
        (self.root / "provenance").symlink_to(self.root / "actual", target_is_directory=True)
        with self.assertRaises(ValueError):
            restore.restore_entry(self.root, self.entry)
        self.assertEqual(list((self.root / "actual").iterdir()), [])

    def test_manifest_hashes_and_unknown_selection(self):
        path = self.root / "manifest.json"
        write_archive_manifest(self.root, path, {"entries": [self.entry]})
        result = restore.restore_manifest(self.root, "manifest.json", [self.entry["source"]])
        self.assertEqual(len(result), 1)
        with self.assertRaises(ValueError):
            restore.restore_manifest(self.root, "manifest.json", ["unknown"])
        document = json.loads(path.read_text())
        (self.root / document["entry_parts"][0]["path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            restore.restore_manifest(self.root, "manifest.json")

    def test_single_archive_and_size_validation(self):
        entry = copy.deepcopy(self.entry)
        del entry["archive_parts"]
        entry["archive"] = "single.xz"
        (self.root / "single.xz").write_bytes(self.packed)
        self.assertEqual(restore.restore_entry(self.root, entry, True)["status"], "verified_only")
        entry["source_bytes"] -= 1
        with self.assertRaises(ValueError):
            restore.restore_entry(self.root, entry)

    def test_duplicate_parts_rejected(self):
        entry = copy.deepcopy(self.entry)
        entry["archive_parts"] *= 2
        with self.assertRaises(ValueError):
            restore.restore_entry(self.root, entry)


if __name__ == "__main__":
    unittest.main()
