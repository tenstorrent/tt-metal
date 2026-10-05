#!/usr/bin/env python3
"""Host-only tests for the current-tuple formal campaign runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

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

    def test_search_result_proposals_use_exact_per_operation_flags(self):
        data = {
            "operations": {
                "a": {"proposal": {"selection": {"flags": "-ma"}}},
                "b": {"proposal": None},
            }
        }
        self.assertEqual(campaign._selection_from_json(data), {"a": "-ma"})

    def test_search_profiles_use_one_explicit_frozen_baseline(self):
        data = {
            "settings": {"baseline_flags": "-mbase"},
            "operations": {
                "a": {
                    "proposal": {
                        "frozen_baseline_flags": "-mbase",
                        "selection": {"flags": "-ma"},
                    }
                }
            },
        }
        self.assertEqual(
            campaign._profiles_from_json(data),
            {"a": {"selected_flags": "-ma", "baseline_flags": "-mbase"}},
        )
        data["operations"]["a"]["proposal"]["frozen_baseline_flags"] = "-mother"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            campaign._profiles_from_json(data)

    def test_selection_without_frozen_baseline_is_rejected(self):
        data = {"operations": {"a": {"selection": {"flags": "-ma"}}}}
        with self.assertRaisesRegex(ValueError, "no frozen baseline"):
            campaign._profiles_from_json(data)

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
        self.assertEqual(campaign.RESULT_STATUS["SCOPE-REFUSED"], "UNSUPPORTED")
        self.assertNotIn("DIVERGENT", campaign.OPERATIONAL_FAILURES)
        self.assertNotIn("UNSUPPORTED", campaign.OPERATIONAL_FAILURES)
        self.assertEqual(
            campaign.RESULT_STATUS["SEMANTICS-UNVALIDATED"],
            "TRACE_VALIDATION_FAILED",
        )
        self.assertEqual(
            campaign.admitted_status("DIVERGENT", "PROVEN_EQUIVALENT_ON_DOMAIN"),
            "PROVEN_EQUIVALENT_ON_DOMAIN",
        )
        self.assertEqual(campaign.admitted_status("DIVERGENT", "DIVERGENT"), "DIVERGENT")

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

    def test_checkpoint_is_running_until_every_selected_row_finishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [{"op": "a", "status": "PROVEN_EQUIVALENT"}]
            campaign.write_results(root, records, {}, time.monotonic(), 2)
            checkpoint = json.loads((root / "formal-results.json").read_text())
            self.assertEqual(checkpoint["status"], "RUNNING")
            self.assertEqual(checkpoint["selected"], 1)
            self.assertEqual(checkpoint["expected_total"], 2)
            self.assertEqual(checkpoint["formal_admission"], "FOLLOWUP_REQUIRED")
            self.assertEqual(checkpoint["followup_required"], 1)

            records.append({"op": "b", "status": "DIVERGENT"})
            campaign.write_results(root, records, {}, time.monotonic(), 2)
            complete = json.loads((root / "formal-results.json").read_text())
            self.assertEqual(complete["status"], "COMPLETE")
            self.assertEqual(complete["selected"], complete["expected_total"])

    def test_checkpoint_files_are_atomically_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "formal-results.json"
            tsv_path = root / "formal-results.tsv"
            old_json = '{"generation": "old"}\n'
            old_tsv = "generation\nold\n"
            json_path.write_text(old_json)
            tsv_path.write_text(old_tsv)
            real_replace = os.replace
            replacements = []

            def inspect_then_replace(source, destination):
                source = Path(source)
                destination = Path(destination)
                self.assertEqual(source.parent, destination.parent)
                self.assertNotEqual(source, destination)
                if destination == json_path:
                    self.assertEqual(destination.read_text(), old_json)
                    checkpoint = json.loads(source.read_text())
                    self.assertEqual(checkpoint["selected"], 1)
                elif destination == tsv_path:
                    self.assertEqual(destination.read_text(), old_tsv)
                    rows = source.read_text().splitlines()
                    self.assertEqual(
                        rows[0].split("\t")[:3],
                        ["op", "status", "compiler_status"],
                    )
                    self.assertEqual(
                        rows[1].split("\t")[:2], ["a", "PROVEN_EQUIVALENT"]
                    )
                else:
                    self.fail(f"unexpected checkpoint destination: {destination}")
                replacements.append(destination)
                real_replace(source, destination)

            records = [{"op": "a", "status": "PROVEN_EQUIVALENT"}]
            with mock.patch.object(campaign.os, "replace", side_effect=inspect_then_replace):
                campaign.write_results(root, records, {}, time.monotonic(), 1)

            self.assertEqual(replacements, [tsv_path, json_path])
            self.assertEqual(json.loads(json_path.read_text())["status"], "COMPLETE")
            self.assertEqual(list(root.glob(".*.tmp")), [])

    @mock.patch.object(campaign, "prove_pair")
    @mock.patch.object(campaign, "run_leg")
    def test_compiler_gate_runs_without_a_handwritten_reference(self, run_leg, prove_pair):
        run_leg.side_effect = lambda **kw: {"node": kw["node"], "leg": kw["leg"]}
        prove_pair.return_value = {"status": "PROVEN_EQUIVALENT"}
        with tempfile.TemporaryDirectory() as temporary:
            result = campaign.run_case(
                op="a",
                case={"kind": "semantic", "sem_corr": "sem.py::test", "hand_corr": ""},
                selected_flags="-mselected",
                baseline_flags="-mbase",
                domain=[{"which": "all"}],
                tests=Path(temporary),
                python=Path("python"),
                sim=Path(temporary) / "libttsim.so",
                root=Path(temporary),
                timeout=1,
            )
        self.assertEqual(result["compiler_status"], "PROVEN_EQUIVALENT")
        self.assertEqual(result["semantic_status"], "NO_REFERENCE")
        self.assertEqual([call.kwargs["leg"] for call in run_leg.call_args_list],
                         ["selected-sem", "baseline-sem"])
        self.assertIsNone(prove_pair.call_args.kwargs["domain"])

    @mock.patch.object(campaign, "prove_pair")
    @mock.patch.object(campaign, "run_leg")
    def test_tri_arm_keeps_compiler_and_semantic_domains_separate(self, run_leg, prove_pair):
        run_leg.side_effect = lambda **kw: {"node": kw["node"], "leg": kw["leg"]}
        prove_pair.side_effect = [
            {"status": "PROVEN_EQUIVALENT"},
            {"status": "PROVEN_EQUIVALENT_ON_DOMAIN"},
        ]
        domain = [{"which": "all", "int_min": 1, "int_max": 7}]
        with tempfile.TemporaryDirectory() as temporary:
            result = campaign.run_case(
                op="a",
                case={
                    "kind": "full2x2",
                    "sem_corr": "sem.py::test",
                    "hand_corr": "hand.py::test",
                },
                selected_flags="-mselected",
                baseline_flags="-mbase",
                domain=domain,
                tests=Path(temporary),
                python=Path("python"),
                sim=Path(temporary) / "libttsim.so",
                root=Path(temporary),
                timeout=1,
            )
        self.assertEqual(
            [call.kwargs["leg"] for call in run_leg.call_args_list],
            ["selected-sem", "baseline-sem", "baseline-hand"],
        )
        self.assertIsNone(prove_pair.call_args_list[0].kwargs["domain"])
        self.assertEqual(prove_pair.call_args_list[1].kwargs["domain"], domain)
        self.assertEqual(result["compiler_gate"], "PASS")
        self.assertEqual(result["semantic_gate"], "PASS")
        self.assertEqual(result["deployment_gate"], "PASS")

    @mock.patch.object(campaign, "prove_pair")
    @mock.patch.object(campaign, "run_leg")
    def test_identical_selected_profile_reuses_semantic_arm(self, run_leg, prove_pair):
        run_leg.side_effect = lambda **kw: {"node": kw["node"], "leg": kw["leg"]}
        prove_pair.return_value = {"status": "PROVEN_EQUIVALENT"}
        with tempfile.TemporaryDirectory() as temporary:
            result = campaign.run_case(
                op="a",
                case={"kind": "full2x2", "sem_corr": "sem", "hand_corr": "hand"},
                selected_flags="-msame",
                baseline_flags="-msame",
                domain=None,
                tests=Path(temporary),
                python=Path("python"),
                sim=Path(temporary) / "libttsim.so",
                root=Path(temporary),
                timeout=1,
            )
        self.assertEqual(
            result["compiler_status"], "NOT_APPLICABLE_IDENTICAL_CONFIGURATION"
        )
        self.assertEqual([call.kwargs["leg"] for call in run_leg.call_args_list],
                         ["selected-sem", "baseline-hand"])
        self.assertEqual(len(prove_pair.call_args_list), 1)
        self.assertEqual(
            prove_pair.call_args.kwargs["trace_sem"].name,
            "trace-selected-sem.log",
        )


if __name__ == "__main__":
    unittest.main()
