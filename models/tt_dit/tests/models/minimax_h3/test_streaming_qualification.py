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
        self.assertEqual(
            set(throughput["included_costs"]),
            {"admission", "upload", "queue", "generation", "export", "delivery", "failures", "retries", "final drain"},
        )
        self.assertEqual(
            set(throughput["excluded_media"]), {"overlap", "padding", "conditioning-only frames", "duplicate attempts"}
        )
        self.assertTrue(throughput["concurrent_galaxies_count"])

    def test_sustained_protocol_and_reporting_survive(self):
        for obsolete in ("minimum_chunks", "minimum_new_playable_seconds", "paced_playback_seconds"):
            self.assertNotIn(obsolete, self.contract)
        self.assertEqual(self.contract["offered_load_seconds"], 60)
        self.assertEqual(self.contract["stop_new_submissions_after_seconds"], 60)
        self.assertTrue(self.contract["declare_finite_drain_bound_before_run"])
        self.assertEqual(self.contract["concurrency_policy"], "modest, existing owner-authorized concurrency")
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
                "actual tested playback duration",
                "prompt-sent timestamp",
                "generation-done timestamp",
                "prompt-to-generation-done roundtrip",
                "time to first clip",
                "per-stage times",
                "file-ready timestamp",
                "client-received timestamp",
                "decoded-playable timestamp",
                "failures and retries",
                "final drain duration and timeouts",
            },
        )
        for evidence in (
            "bounded host/device memory",
            "A/V continuity and quality",
            "zero underruns after declared startup buffer",
        ):
            self.assertIn(evidence, self.contract["required_evidence"])

    def test_current_docs_replace_long_duration_gates(self):
        for path in (
            DOCS / "served_source_reconciliation_20261110.md",
            DOCS / "streaming_integration_20261110.md",
            DOCS.parent / "pipelines/minimax_h3/STREAMING.md",
        ):
            text = path.read_text()
            self.assertIn("60 seconds of offered load", text)
            self.assertIn("bounded final drain", text)
            self.assertIn("prompt-sent", text)
            self.assertIn("generation-done", text)
            self.assertNotIn("Retain the initial >=60-chunk", text)
            self.assertNotIn("Then measure at least 60 chunks", text)

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
