# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import tempfile
import unittest
from pathlib import Path

from device_triage import index


class PublishTest(unittest.TestCase):
    def test_snapshot_opens_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = index.connect(Path(tmp) / "work" / "index.sqlite")
            conn.execute("INSERT INTO units (unit, source) VALUES ('u1', 'machine_hc')")
            conn.commit()
            snapshot = index.publish(conn, Path(tmp) / "shared")
            reader = index.connect(snapshot)
            self.assertEqual(reader.execute("SELECT unit FROM units").fetchall(), [("u1",)])
            with self.assertRaises(sqlite3.OperationalError):
                reader.execute("INSERT INTO units (unit, source) VALUES ('u2', 'machine_hc')")


if __name__ == "__main__":
    unittest.main()
