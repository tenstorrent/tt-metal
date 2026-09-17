# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fault tests for complete performance evidence and observational token reporting."""

import copy
import unittest

from models.demos.llama_3p1_8b_d_p.tests.performance.long_context_performance_validation import report_reasons
from models.demos.llama_3p1_8b_d_p.tests.performance.synthetic_report import build_report


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    # A poor continuation rank remains readable evidence, not an invented semantic failure.
    def test_observation_without_semantic_success_gate(self):
        self.assertEqual(report_reasons(self.report, 4096), [])
        self.assertGreater(self.report["next_token_observations"][0]["expected_next_token_rank"], 5)

    # Every request must have one final observation from its exact real row and all TP slices.
    def test_missing_duplicate_wrong_row_and_tp_reject(self):
        for change in ("missing", "duplicate", "row", "tp", "position", "vocab"):
            report = copy.deepcopy(self.report)
            rows = report["next_token_observations"]
            if change == "missing":
                rows.pop()
            if change == "duplicate":
                rows[-1] = rows[0]
            if change == "row":
                rows[-1]["local_row"] = 254
            if change == "tp":
                rows[-1]["tp_order"][-1] = 6
            if change == "position":
                rows[-1]["final_prompt_position"] = 4094
            if change == "vocab":
                rows[-1]["vocab_size"] = 16032
            self.assertTrue(report_reasons(report, 4096), change)

    # Token IDs, readable metadata and repeated decoded observations must remain slot-bound.
    def test_slot_metadata_rank_and_probability_faults_reject(self):
        for field, value in (
            ("book_id", "other"),
            ("expected_next_token_id", 22),
            ("expected_next_token_rank", 0),
            ("expected_next_token_probability", float("nan")),
            ("argmax_token_id", 99),
            ("final_logits_float32_sha256", "0" * 64),
        ):
            report = copy.deepcopy(self.report)
            report["next_token_observations"][-1][field] = value
            self.assertTrue(report_reasons(report, 4096), field)

    # Sequential full-request intervals must not overlap even when individual chunk durations are valid.
    def test_full_request_overlap_rejects(self):
        report = copy.deepcopy(self.report)
        sample = report["samples"][1]
        delta = sample["clock_timestamps"]["prompt_start"] - report["samples"][0]["clock_timestamps"]["prompt_start"]
        for key in ("prompt_start", "prompt_end"):
            sample["clock_timestamps"][key] -= delta
        for chunk in sample["clock_timestamps"]["chunks"]:
            for key in ("chunk_start", "forward_start", "forward_end"):
                chunk[key] -= delta
        # Durations and summary remain unchanged; only the impossible overlap differs.
        self.assertIn("Full request timing intervals overlap or are out of order", report_reasons(report, 4096))

    # Readback/ranking cost is reported separately for every request and cannot disappear from the report.
    def test_readback_inventory_and_finite_cost_reject(self):
        for change in ("missing", "duplicate", "nan"):
            report = copy.deepcopy(self.report)
            rows = report["readback_hash_wall_seconds"]
            if change == "missing":
                rows.pop()
            if change == "duplicate":
                rows[-1] = rows[0]
            if change == "nan":
                rows[-1]["seconds"] = float("nan")
            self.assertTrue(report_reasons(report, 4096), change)


if __name__ == "__main__":
    unittest.main()
