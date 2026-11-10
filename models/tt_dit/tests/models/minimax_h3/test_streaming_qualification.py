# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Lightweight regression checks for the served-source qualification contract."""

import json
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parents[3] / "docs"


class ServedQualificationTest(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads((DOCS / "served_source_config_20261110.json").read_text())["qualification"]

    def test_current_acceptance_replaces_sequential_rtf(self):
        for obsolete in ("minimum_sequential_chunks", "final_capacity_rtf_max", "service_rtf_max"):
            self.assertNotIn(obsolete, self.contract)
        throughput = self.contract["sustained_caller_throughput"]
        self.assertEqual(throughput["operator"], ">")
        self.assertEqual(throughput["threshold"], 1)
        self.assertEqual(throughput["numerator"], "unique playable video seconds received by the API caller")
        self.assertEqual(throughput["denominator"], "elapsed wall seconds")
        self.assertEqual(set(throughput["included_costs"]), {"queue", "generation", "export", "delivery"})
        self.assertEqual(set(throughput["excluded_media"]), {"overlap", "padding"})
        self.assertTrue(throughput["concurrent_galaxies_count"])

    def test_sustained_protocol_and_reporting_survive(self):
        self.assertEqual(self.contract["minimum_chunks"], 60)
        self.assertEqual(self.contract["minimum_new_playable_seconds"], 300)
        self.assertEqual(self.contract["paced_playback_seconds"], 1800)
        self.assertEqual(self.contract["production"], "concurrent galaxies permitted in both protocols")
        self.assertEqual(
            set(self.contract["report_separately"]),
            {
                "per-clip latency",
                "startup buffer",
                "concurrency",
                "participating galaxies",
                "unpaced capacity",
                "deliberately paced playback",
            },
        )
        for evidence in (
            "bounded host/device memory",
            "A/V continuity and quality",
            "zero underruns after declared startup buffer",
        ):
            self.assertIn(evidence, self.contract["required_evidence"])

    def test_all_four_cells_require_power_evidence(self):
        self.assertEqual(self.contract["required_location"], "TYO2")
        self.assertEqual(self.contract["required_watts_per_chip"], 160)
        self.assertIsNone(self.contract["enforced_power_readbacks"])
        self.assertIn("per-chip enforced 160 W readback", self.contract["required_evidence"])
        self.assertEqual(
            {(c["model"], c["mode"]) for c in self.contract["cells"]},
            {(model, mode) for model in ("H3", "LTX 2.5 IC-LoRA") for mode in ("FL2V", "R2V")},
        )
        self.assertFalse(any(c["qualified"] for c in self.contract["cells"]))

    def test_document_marks_old_contract_historical(self):
        text = (DOCS / "served_source_reconciliation_20261110.md").read_text()
        self.assertIn("Historical contract (superseded):", text)
        self.assertNotIn("Final qualification requires unpaced capacity RTF <=0.8", text)
        self.assertNotIn("nor sustained RTF <=0.8", text)
        self.assertIn("unique playable video received by the API caller / elapsed wall time >1", text)


if __name__ == "__main__":
    unittest.main()
