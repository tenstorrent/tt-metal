# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

from device_triage import collect


class CollectTest(unittest.TestCase):
    def test_run_dirs_in_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for day, job in (("2026-10-10", "131697"), ("2026-10-12", "131717_post-reboot-1"), ("2026-09-01", "1")):
                run = root / "bh-glx-120-b02u08" / day / job
                run.mkdir(parents=True)
                (run / "diag_report.json").write_text("{}")
            dirs = collect._run_dirs(root, "bh-glx-120-b02u08", "2026-10-01")
            self.assertEqual([Path(d).name for d in dirs], ["131697", "131717_post-reboot-1"])
            self.assertEqual(collect._run_dirs(root, "bh-glx-110-a01u02", "2026-10-01"), [])

    def test_recover_dir_dates(self):
        self.assertEqual(collect.RECOVER_DIR_DATE_RE.search("zcarver-20261003T191002Z.ZT2Pde").group(1), "20261003")

    def test_unparseable_unit_is_reported_not_raised(self):
        res = collect.parse_unit(("cluster_report", "/nonexistent/report.json"))
        self.assertTrue(res.error.startswith("FileNotFoundError"))


if __name__ == "__main__":
    unittest.main()
