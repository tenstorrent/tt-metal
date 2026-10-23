# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import unittest
from noc_routing_model import predict

T = dict(
    workers=[[1, 3], [1, 4], [2, 3], [2, 4]],
    banks=[[0, 0], [0, 1]],
    banks_noc1=[[0, 0], [0, 1]],
    physical_grid=[4, 6],
    clock_mhz=1350,
)


def case(mode, banks="balanced"):
    return dict(cores=4, payload_bytes=4096, mode=mode, banks=banks, pages=32, batch=16, iterations=1)


class RoutingTests(unittest.TestCase):
    def test_equal_work_and_distinct_noc_resources(self):
        for mode in ("noc0", "noc1", "split"):
            r = predict(T, case(mode))
            self.assertEqual(sum(r["read_requests_by_noc"].values()), 128)
            self.assertEqual(r["read_bytes_per_iteration"], 128 * 4096)
            if mode == "split":
                self.assertEqual(r["read_requests_by_noc"], {"0": 64, "1": 64})

    def test_column_first_concentrates_response_traffic(self):
        a = predict(T, case("noc0"))
        b = predict(T, case("noc1"))
        self.assertGreater(b["hottest_response_links"][0]["flit_cycles"], a["hottest_response_links"][0]["flit_cycles"])
        self.assertEqual(b["hottest_response_links"][0]["resource"][2], 0)

    def test_two_nocs_do_not_duplicate_bank_bandwidth(self):
        bounds = [predict(T, case(m, "hotspot"))["bank_occupancy_lower_bound_us"] for m in ("noc0", "noc1", "split")]
        self.assertEqual(bounds, [bounds[0]] * 3)
        r = predict(T, case("split", "hotspot"))
        self.assertGreaterEqual(r["estimated_us"], r["bank_occupancy_lower_bound_us"])

    def test_contention_costs_time(self):
        r = predict(T, case("noc1"))
        isolated = predict(T, case("noc1"), contention=False)
        self.assertGreater(r["estimated_us"], isolated["estimated_us"])


if __name__ == "__main__":
    unittest.main()
