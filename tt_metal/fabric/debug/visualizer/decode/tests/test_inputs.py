# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tt_metal.fabric.debug.visualizer.decode.inputs import DecodeError, discover_inputs
from tt_metal.fabric.debug.visualizer.decode.tests.fixtures import write_input


class DecodeInputsTest(unittest.TestCase):
    def test_pairs_by_manifest_sha_and_verifies_raw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, snapshot, raw = write_input(root)
            result = discover_inputs([snapshot, manifest])

            self.assertEqual(len(result), 1)
            self.assertTrue(result[0].manifest_sha_verified)
            self.assertEqual(result[0].raw.path, raw.resolve())
            self.assertTrue(result[0].raw.verified)

    def test_manifest_mismatch_is_refused_or_explicitly_flagged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, snapshot, _ = write_input(root)
            data = json.loads(manifest.read_text())
            data["run"]["written_at"] = "2026-09-15T00:00:01Z"
            manifest.write_text(json.dumps(data))

            with self.assertRaisesRegex(DecodeError, "no supplied manifest"):
                discover_inputs([root])
            result = discover_inputs([root], allow_manifest_mismatch=True)
            self.assertFalse(result[0].manifest_sha_verified)

    def test_missing_raw_is_retained_for_partial_decode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, _, raw = write_input(root)
            raw.unlink()
            result = discover_inputs([root])
            self.assertIsNone(result[0].raw)

    def test_raw_size_and_hash_are_verified(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, _, raw = write_input(root)
            raw.write_bytes(b"wrong size")
            with self.assertRaisesRegex(DecodeError, "size"):
                discover_inputs([root])

            _, snapshot, raw = write_input(root)
            snapshot_data = json.loads(snapshot.read_text())
            bad = b"x" * snapshot_data["raw"]["size"]
            raw.write_bytes(bad)
            self.assertNotEqual(hashlib.sha256(bad).hexdigest(), snapshot_data["raw"]["sha256"])
            with self.assertRaisesRegex(DecodeError, "sha256"):
                discover_inputs([root])

    def test_mixed_architectures_are_refused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_input(root / "rank0", rank=0, local_chip=0, all_chip_ids=(0, 1))
            manifest, snapshot, _ = write_input(
                root / "rank1", rank=1, local_chip=1, all_chip_ids=(0, 1)
            )
            manifest_data = json.loads(manifest.read_text())
            manifest_data["run"]["arch"] = "BLACKHOLE"
            manifest.write_text(json.dumps(manifest_data))
            snapshot_data = json.loads(snapshot.read_text())
            snapshot_data["manifest"]["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            snapshot_data["manifest"]["run"]["arch"] = "BLACKHOLE"
            snapshot.write_text(json.dumps(snapshot_data))

            with self.assertRaisesRegex(DecodeError, "mix run architectures"):
                discover_inputs([root / "rank0", root / "rank1"])


if __name__ == "__main__":
    unittest.main()
