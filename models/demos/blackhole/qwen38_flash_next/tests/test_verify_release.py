# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for weekly read-only checkpoint verification."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.demos.blackhole.qwen38_flash_next.tools import verify_release


class VerifyReleaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        entries = []
        for name, payload in (("model.safetensors", b"weights"), ("config.json", b"{}"), ("tokenizer.json", b"tokens")):
            (self.checkpoint / name).write_bytes(payload)
            entries.append(
                {"Type": "blob", "Path": name, "Size": len(payload), "Sha256": hashlib.sha256(payload).hexdigest()}
            )
        self.manifest = self.root / "files.json"
        self.manifest.write_text(json.dumps({"Data": {"Files": entries}}))
        # A tiny synthetic release exercises the same pinned-manifest boundary.
        manifest_pin = patch.object(
            verify_release, "MANIFEST_SHA256", hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        )
        manifest_pin.start()
        self.addCleanup(manifest_pin.stop)

    def test_all_files_verified_without_writing_read_only_checkpoint(self):
        before = {path.name: path.read_bytes() for path in self.checkpoint.iterdir()}
        for path in self.checkpoint.iterdir():
            path.chmod(0o444)
        self.checkpoint.chmod(0o555)
        self.addCleanup(self.checkpoint.chmod, 0o755)
        result = verify_release.verify(self.checkpoint, self.manifest)
        self.assertEqual(result["verified_files"], 3)
        self.assertEqual(result["verified_bytes"], sum(map(len, before.values())))
        self.assertEqual({path.name: path.read_bytes() for path in self.checkpoint.iterdir()}, before)

    def test_corrupt_weight_config_and_tokenizer_are_each_fatal(self):
        for name in ("model.safetensors", "config.json", "tokenizer.json"):
            with self.subTest(name=name):
                path = self.checkpoint / name
                original = path.read_bytes()
                path.write_bytes(b"x" * len(original))
                with self.assertRaisesRegex(ValueError, name):
                    verify_release.verify(self.checkpoint, self.manifest)
                path.write_bytes(original)

    def test_missing_file_is_fatal(self):
        (self.checkpoint / "config.json").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_release.verify(self.checkpoint, self.manifest)

    def test_manifest_must_match_pinned_bytes(self):
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            verify_release.verify(self.checkpoint, self.manifest)

    def test_worker_count_is_bounded(self):
        for workers in (0, 5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                verify_release.verify(self.checkpoint, self.manifest, workers=workers)


if __name__ == "__main__":
    unittest.main()
