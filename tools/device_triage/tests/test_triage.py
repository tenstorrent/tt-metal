# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import unittest

from device_triage import triage


class SweepTest(unittest.TestCase):
    def test_sweep_hosts_counts_distinct_hosts_in_window(self):
        times = [1000.0, 1010.0, 1020.0, 1030.0, 5000.0]
        host_list = ["a", "b", "b", "c", "d"]
        stats = triage.FleetStats({}, {"eth_ports_down": (times, host_list)})
        self.assertEqual(stats.sweep_hosts("eth_ports_down", 1015.0), 3)
        self.assertEqual(stats.sweep_hosts("eth_ports_down", 5000.0), 1)
        self.assertEqual(stats.sweep_hosts("query_mappings_failed", 1015.0), 0)


if __name__ == "__main__":
    unittest.main()
