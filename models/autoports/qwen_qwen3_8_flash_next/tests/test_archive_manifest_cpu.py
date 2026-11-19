# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only losslessness and tamper checks for partitioned archive indexes."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from .archive_manifest import encoded, load_archive_manifest, sha256, write_archive_manifest


class ArchiveManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "evidence_archives.json"
        self.document = {
            "schema_version": 1,
            "scope": "test",
            "packaging_updates": [{"old": "retained"}],
            "entries": [{"source": f"source_{i}", "historical_identity": str(i) * 300} for i in range(7)],
        }
        self.path.write_bytes(encoded(self.document))

    def split(self):
        return write_archive_manifest(self.root, self.path, self.document, max_part_bytes=1000)

    def test_roundtrip_retains_every_entry_and_metadata_deterministically(self):
        self.assertEqual(load_archive_manifest(self.root, self.path), self.document)
        index = self.split()
        self.assertGreater(len(index["entry_parts"]), 1)
        self.assertNotIn("entries", index)  # Legacy readers must fail, not silently omit history.
        loaded = load_archive_manifest(self.root, self.path)
        self.assertEqual(loaded["entries"], self.document["entries"])
        self.assertEqual(loaded["packaging_updates"], self.document["packaging_updates"])
        original = self.path.read_bytes()
        write_archive_manifest(self.root, self.path, loaded, max_part_bytes=1000)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue(all(record["bytes"] <= 1000 for record in index["entry_parts"]))

    def test_missing_or_changed_part_fails_closed(self):
        index = self.split()
        part = self.root / index["entry_parts"][0]["path"]
        original = part.read_bytes()
        part.unlink()
        with self.assertRaises(FileNotFoundError):
            load_archive_manifest(self.root, self.path)
        part.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            load_archive_manifest(self.root, self.path)

    def test_omission_order_and_duplicate_cannot_hide_entries(self):
        index = self.split()
        for mutate in (
            lambda x: x["entry_parts"].pop(),
            lambda x: x["entry_parts"].reverse(),
            lambda x: x["entry_parts"].append(x["entry_parts"][0]),
        ):
            changed = copy.deepcopy(index)
            mutate(changed)
            self.path.write_bytes(encoded(changed))
            with self.assertRaises(ValueError):
                load_archive_manifest(self.root, self.path)

    def test_resigned_part_still_requires_combined_identity(self):
        index = self.split()
        record = index["entry_parts"][0]
        path = self.root / record["path"]
        part = json.loads(path.read_bytes())
        part["entries"][0]["historical_identity"] = "changed"
        payload = encoded(part)
        path.write_bytes(payload)
        record.update(bytes=len(payload), sha256=sha256(payload))
        self.path.write_bytes(encoded(index))
        with self.assertRaisesRegex(ValueError, "Combined archive identity"):
            load_archive_manifest(self.root, self.path)

    def test_unsafe_paths_and_duplicate_sources_rejected(self):
        index = self.split()
        index["entry_parts"][0]["path"] = "../outside.json"
        self.path.write_bytes(encoded(index))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            load_archive_manifest(self.root, self.path)
        self.document["entries"].append(self.document["entries"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate archive source"):
            self.split()

    def test_mixed_schema_and_oversized_entry_rejected(self):
        index = self.split()
        index["entries"] = []
        self.path.write_bytes(encoded(index))
        with self.assertRaisesRegex(ValueError, "mix inline"):
            load_archive_manifest(self.root, self.path)
        self.document["entries"][0]["historical_identity"] = "x" * 1100
        with self.assertRaisesRegex(ValueError, "exceeds part"):
            self.split()


if __name__ == "__main__":
    unittest.main()
