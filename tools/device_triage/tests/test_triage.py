# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import unittest

from device_triage import triage
from device_triage.model import Event, HostData, Run, UnitResult

HOST = "bh-glx-120-b02u08"
NOW = 1_800_000_000.0


def run(i: int, status: str, age_days: float) -> Run:
    ts = NOW - age_days * triage.DAY
    return Run(f"mhc:{HOST}/{i}", "machine_hc", "medium", [HOST], ts, ts, status, status.lower(), f"/runs/{i}")


def missing_tray(i: int, age_days: float, repeats: int = 1) -> list[Event]:
    ts = NOW - age_days * triage.DAY
    return [
        Event(
            HOST,
            ts,
            "machine_hc",
            "check:snapshot/asic_location_per_ubb",
            "pcie",
            "asic_location_missing",
            "fail",
            "missing UBBs: [3]",
            chips="u3",
            run_id=f"mhc:{HOST}/{i}",
        )
        for _ in range(repeats)
    ]


class TriageTest(unittest.TestCase):
    def data(self, results: list[UnitResult]) -> HostData:
        data = HostData()
        for res in results:
            data.add(res)
        return data

    def test_failing_host_with_missing_tray(self):
        data = self.data(
            [
                UnitResult(f"/runs/{i}", runs=[run(i, "FAIL", age)], events=missing_tray(i, age, repeats=50))
                for i, age in ((1, 3.0), (2, 2.0), (3, 1.0))
            ]
        )
        report = triage.triage(HOST, data, [], NOW, rarity={"asic_location_missing": 0.9})
        self.assertEqual(report.status, "FAILING")
        self.assertEqual((report.likely, report.likely_chips), ("pcie", "u3"))
        # 50 identical lines in one run are one piece of evidence: 3 runs, not 150.
        self.assertEqual(report.scores[0].runs, 3)
        self.assertEqual(report.scores[0].evidence[0].runs, 3)

    def test_events_outside_window_or_other_hosts_are_ignored(self):
        old = missing_tray(1, 40.0)
        other = [Event("bh-glx-110-a01u02", NOW, "machine_hc", "x", "pcie", "asic_location_missing", "fail", "")]
        data = self.data([UnitResult("/runs/1", runs=[run(1, "PASS", 0.5)], events=old + other)])
        report = triage.triage(HOST, data, [], NOW, window_days=30, rarity={})
        self.assertEqual((report.status, report.likely), ("HEALTHY", "none"))

    def test_no_runs_is_unknown(self):
        report = triage.triage(HOST, HostData(), [], NOW, rarity={})
        self.assertEqual(report.status, "UNKNOWN")
        self.assertIn("UNKNOWN", triage.render(report))

    def test_pcie_and_reset_are_one_issue(self):
        self.assertTrue(triage.same_family("pcie", "reset"))
        self.assertFalse(triage.same_family("pcie", "gddr"))


if __name__ == "__main__":
    unittest.main()
