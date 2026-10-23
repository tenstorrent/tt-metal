# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import unittest
from noc_routing_model import predict
from read_service import ReadService, topology_signature
from test_noc_routing_model import T, case


class WriteTransportTests(unittest.TestCase):
    def test_write_bytes_and_ack_traffic(self):
        r = predict(T, dict(case("noc0"), traffic="write"))
        self.assertEqual(r["write_bytes_per_iteration"], 4 * 32 * 4096)
        self.assertEqual(r["read_bytes_per_iteration"], 0)
        self.assertEqual(r["write_requests_by_noc"], {"0": 128})
        self.assertGreater(r["hottest_request_links"][0]["flit_cycles"], r["hottest_response_links"][0]["flit_cycles"])

    def test_shared_bank_not_doubled_by_two_nocs(self):
        rows = [predict(T, dict(case(m, "hotspot"), traffic="write")) for m in ["noc0", "noc1", "split"]]
        self.assertEqual(len({r["bank_occupancy_lower_bound_us"] for r in rows}), 1)
        for r in rows:
            self.assertGreaterEqual(r["estimated_us"], r["bank_occupancy_lower_bound_us"])

    def test_read_admission_not_applied_to_write(self):
        profile = ReadService(
            dict(topology_signature=topology_signature(T), cycles_per_request_by_noc_bank=[10000, 10000])
        )
        c = dict(case("split"), traffic="write")
        self.assertEqual(predict(T, c)["estimated_us"], predict(T, c, read_service=profile)["estimated_us"])

    def test_mixed_bytes_and_shared_bank(self):
        c = dict(case("split", "hotspot"), traffic="mixed")
        r = predict(T, c)
        self.assertEqual(r["read_bytes_per_iteration"], r["write_bytes_per_iteration"])
        write = predict(T, dict(c, traffic="write"))
        self.assertEqual(r["bank_occupancy_lower_bound_us"], write["bank_occupancy_lower_bound_us"])
        self.assertEqual(r["read_requests_by_noc"], {"0": 64})
        self.assertEqual(r["write_requests_by_noc"], {"1": 64})

    def test_single_write_pays_issue_cost(self):
        c = dict(case("noc0"), traffic="write", cores=1, batch=1, pages=1, iterations=1)
        base = predict(T, c)["estimated_cycles"]
        issued = predict(T, c, issue_cycles=(40, 40))["estimated_cycles"]
        self.assertEqual(issued - base, 40)

    def test_source_flush_is_not_remote_completion(self):
        with self.assertRaisesRegex(ValueError, "source flush"):
            predict(T, dict(case("noc0"), traffic="write", completion="flushed"))


if __name__ == "__main__":
    unittest.main()
