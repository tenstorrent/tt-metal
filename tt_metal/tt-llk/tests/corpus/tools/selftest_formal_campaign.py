#!/usr/bin/env python3
"""Host-only tests for the current-tuple formal campaign runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import formal_campaign as campaign


class FormalCampaignTests(unittest.TestCase):
    def test_selection_uses_exact_per_operation_flags(self):
        data = {
            "operations": {
                "a": {"selection": {"flags": "-ma -mno-b"}},
                "b": {"selection": {"flags": "-mc"}},
                "unselected": {"status": "INCOMPLETE"},
            }
        }
        self.assertEqual(
            campaign._selection_from_json(data),
            {"a": "-ma -mno-b", "b": "-mc"},
        )

    def test_selected_tsv_without_flags_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected.tsv"
            path.write_text("op\ttoggles\na\tpass-a\n")
            with self.assertRaisesRegex(ValueError, "flags"):
                campaign.load_selection(path)

    def test_case_nodes_come_from_canonical_corpus_columns(self):
        case = {"sem_corr": "sem.py::test[a]", "hand_corr": "hand.py::test[b]"}
        self.assertEqual(
            campaign.case_nodes(case),
            ("sem.py::test[a]", "hand.py::test[b]"),
        )

    def test_domains_are_explicit_per_operation(self):
        domain = {"mul": [{"which": "all", "int_min": 1, "int_max": 40000}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "domains.json"
            path.write_text(json.dumps(domain))
            self.assertEqual(campaign.load_domains(path), domain)

    def test_formal_statuses_do_not_call_divergence_an_operational_failure(self):
        self.assertEqual(campaign.RESULT_STATUS["DIVERGENT"], "DIVERGENT")
        self.assertNotIn("DIVERGENT", campaign.OPERATIONAL_FAILURES)
        self.assertEqual(
            campaign.RESULT_STATUS["SEMANTICS-UNVALIDATED"],
            "TRACE_VALIDATION_FAILED",
        )

    def test_clean_environment_drops_caller_test_hooks(self):
        clean = campaign.clean_environment({"CHIP_ARCH": "blackhole"})
        self.assertEqual(clean["CHIP_ARCH"], "blackhole")
        self.assertNotIn("PYTEST_ADDOPTS", clean)
        self.assertNotIn("TT_LLK_EXTRA_COMPILER_OPTIONS", clean)

    def test_runner_has_no_historical_admission_inputs(self):
        source = Path(campaign.__file__).read_text()
        self.assertNotIn("prove_all_silicon_overlay", source)
        self.assertNotIn("prove_all_domain_overlay", source)
        self.assertNotIn("FINAL-BOARD", source)
        self.assertNotIn("EXPECT_JO_SHA", source)

    def test_artifact_operation_names_are_bounded(self):
        self.assertTrue(campaign.operation_slug("mulint32-fresh"))
        self.assertFalse(campaign.operation_slug("../outside"))


if __name__ == "__main__":
    unittest.main()
