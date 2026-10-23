# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import copy
import unittest
from pathlib import Path
from read_service import ReadService, characterize, topology_signature
from noc_routing_model import predict
from test_noc_routing_model import T, case


def service(intervals=(200, 200)):
    return ReadService(dict(topology_signature=topology_signature(T), cycles_per_request_by_noc_bank=intervals))


class ReadServiceTests(unittest.TestCase):
    def test_topology_guard(self):
        changed = copy.deepcopy(T)
        changed["clock_mhz"] += 1
        with self.assertRaisesRegex(ValueError, "mismatch"):
            service().validate(changed)

    def test_distinct_admission_shared_bytes(self):
        results = []
        for mode in ("noc0", "noc1", "split"):
            c = dict(case(mode, "hotspot"), payload_bytes=128)
            results.append(predict(T, c, read_service=service()))
        self.assertLess(results[2]["estimated_us"], results[0]["estimated_us"] * 0.7)
        self.assertEqual(len({r["bank_occupancy_lower_bound_us"] for r in results}), 1)
        self.assertEqual(len({r["read_bytes_per_iteration"] for r in results}), 1)

    def test_unused_noc_capacity_does_not_affect_reads(self):
        c = dict(case("noc0"), payload_bytes=128)
        a = predict(T, c, read_service=service((200, 1)))
        b = predict(T, c, read_service=service((200, 10000)))
        self.assertEqual(a["estimated_us"], b["estimated_us"])

    def test_reproduce_independent_profile(self):
        import json

        root = Path(__file__).parent
        topology = json.loads((root / "topology_bh.json").read_text())
        derived = characterize(root / "read_probe_bh.jsonl", topology)
        saved = ReadService(root / "read_service_bh.json")
        self.assertEqual(derived, saved.profile)
        self.assertEqual([r["mode"] for r in derived["training_cases"]], ["noc0", "noc1"])
        self.assertTrue(all(r["cores"] == 110 for r in derived["training_cases"]))


if __name__ == "__main__":
    unittest.main()
