# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-content source hashing remains exact and fail-closed with eight readers."""
import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


class SourceInventoryTests(unittest.TestCase):
    # Parallel reads must reproduce every digest and preserve the caller's deterministic path order.
    def test_matches_serial_hashes_and_detects_content_change(self):
        from models.demos.llama_3p1_8b_d_p.tests.performance.source_inventory import hash_files

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(19):
                p = Path(tmp) / str(i)
                p.write_bytes(bytes([i]) * 100)
                paths.append(str(p))
            expected = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
            self.assertEqual(hash_files(paths), expected)
            self.assertEqual(list(hash_files(paths)), paths)
            Path(paths[-1]).write_bytes(b"changed")
            self.assertNotEqual(hash_files(paths), expected)

    # A missing file must raise rather than return a partial map that could look like successful verification.
    def test_missing_file_and_duplicates_refuse(self):
        from models.demos.llama_3p1_8b_d_p.tests.performance.source_inventory import hash_files

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "source"
            p.write_bytes(b"a")
            with self.assertRaises(FileNotFoundError) as error:
                hash_files([str(p), str(p.parent / "first-missing"), str(p.parent / "second-missing")])
            self.assertEqual(error.exception.filename, str(p.parent / "first-missing"))
            with self.assertRaises(ValueError):
                hash_files([str(p), str(p)])

    # Eight independent reads may overlap; a barrier makes an accidental serial implementation fail.
    def test_eight_worker_bound_and_deterministic_error(self):
        from models.demos.llama_3p1_8b_d_p.tests.performance import source_inventory

        barrier = threading.Barrier(8, timeout=5)
        active = 0
        peak = 0
        lock = threading.Lock()

        def fake(path):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            barrier.wait()
            with lock:
                active -= 1
            return str(path)

        with mock.patch.object(source_inventory, "file_sha", fake):
            result = source_inventory.hash_files([str(i) for i in range(8)])
        self.assertEqual(peak, 8)
        self.assertEqual(result, {str(i): str(i) for i in range(8)})
