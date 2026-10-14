# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

from device_triage import live


class LiveTest(unittest.TestCase):
    def test_run_dirs_in_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for day, job in (("2026-10-10", "131697"), ("2026-10-12", "131717_post-reboot-1"), ("2026-09-01", "1")):
                run = root / "bh-glx-120-b02u08" / day / job
                run.mkdir(parents=True)
                (run / "diag_report.json").write_text("{}")
            dirs = live._run_dirs(root, "bh-glx-120-b02u08", "2026-10-01")
            self.assertEqual([Path(d).name for d in dirs], ["131697", "131717_post-reboot-1"])
            self.assertEqual(live._run_dirs(root, "bh-glx-110-a01u02", "2026-10-01"), [])

    def test_siblings_and_recover_dir_dates(self):
        self.assertEqual(live._siblings("bh-glx-120-b02u08"), ["bh-glx-110-b02u08"])
        self.assertEqual(live._siblings("qb2-120-p03t08"), [])
        self.assertEqual(live.RECOVER_DIR_DATE_RE.search("zcarver-20261003T191002Z.ZT2Pde").group(1), "20261003")

    def test_rarity_snapshot_loads(self):
        stats = live.fleet_stats()
        self.assertIn("post_reset_failed", stats.rarity)
        self.assertEqual(stats.sweeps, {})


if __name__ == "__main__":
    unittest.main()
