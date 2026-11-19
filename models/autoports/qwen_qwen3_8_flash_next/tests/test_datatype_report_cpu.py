# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only safeguards against capability-failed precision selections."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "datatype_report_cpu", Path(__file__).with_name("report_datatype_sweep.py")
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class CapabilityEligibilityTest(unittest.TestCase):
    def test_missing_decision_is_ineligible_despite_accuracy_and_p1(self):
        decision = report.capability_decision("kv8", {}, Path("."))
        status, failures, pending = report.acceptance_status(
            accuracy=True, execution=True, p1=True, traced=True, capability=decision
        )
        self.assertEqual(status, "pending")
        self.assertEqual(failures, [])
        self.assertEqual(pending, ["capability"])
        self.assertIsNone(decision["passed"])

    def test_false_structured_gate_overrides_declared_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "comparison.json").write_text(json.dumps({"required_gate_passed": False}))
            decisions = {
                "kv8": {
                    "passed": True,
                    "reason": "test declaration",
                    "evidence": ["comparison.json"],
                    "checks": [{"artifact": "comparison.json", "field": "required_gate_passed"}],
                }
            }
            decision = report.capability_decision("kv8", decisions, root)
            self.assertFalse(decision["passed"])
            self.assertFalse(decision["checked_gates"][0]["passed"])
            self.assertEqual(
                report.acceptance_status(accuracy=True, execution=True, p1=True, traced=True, capability=decision)[0],
                "fail_capability",
            )

    def test_rejection_is_not_overridden_by_passing_other_gate(self):
        decisions = {"kv8": {"passed": False, "reason": "PCC below0.99", "evidence": ["comparison.json"]}}
        decision = report.capability_decision("kv8", decisions, Path("."))
        self.assertEqual(
            report.acceptance_status(accuracy=True, execution=True, p1=True, traced=True, capability=decision),
            ("fail_capability", ["capability"], []),
        )

    def test_all_five_gates_are_required(self):
        decision = {"passed": True}
        self.assertEqual(
            report.acceptance_status(accuracy=True, execution=True, p1=True, traced=True, capability=decision),
            ("pass", [], []),
        )
        for missing in ("accuracy", "execution", "p1", "traced"):
            args = dict(accuracy=True, execution=True, p1=True, traced=True, capability=decision)
            args[missing] = False
            status, failures, pending = report.acceptance_status(**args)
            self.assertNotEqual(status, "pass")
            self.assertEqual(len(failures), 1)

    def test_invalid_decision_cannot_use_truthy_string(self):
        with self.assertRaises(ValueError):
            report.capability_decision(
                "kv8", {"kv8": {"passed": "false", "reason": "bad", "evidence": ["x"]}}, Path(".")
            )


class UnscoredSelectionTest(unittest.TestCase):
    @staticmethod
    def row(cid, speed, p1=True, capability=True):
        decision = {"passed": p1, "status": "not_scored" if p1 is None else "pass", "reason": "test outcome"}
        status, failed, pending = report.acceptance_status(
            accuracy=True, execution=True, p1=p1, traced=True, capability={"passed": capability}
        )
        return {
            "config_id": cid,
            "teacher_forcing_decode_t_s_u": speed,
            "trace_verified": True,
            "p1": decision,
            "capability": {"passed": capability},
            "status": status,
            "failed_gates": failed,
            "pending_gates": pending,
            "eligible": status == "pass",
            "evidence": cid + ".json",
        }

    @staticmethod
    def prune(row, comparator):
        row["p1"].update(
            status="not_scored_dominated",
            dominated_by=comparator["config_id"],
            evidence=[row["evidence"], comparator["evidence"]],
        )

    def test_unscored_p1_retains_null_without_failure(self):
        decision = report.p1_decision("slow", {})
        self.assertIsNone(decision["passed"])
        self.assertEqual(decision["status"], "not_scored")
        self.assertTrue(decision["reason"])
        self.assertEqual(
            report.acceptance_status(accuracy=True, execution=True, p1=None, traced=True, capability={"passed": True}),
            ("pending_p1", [], ["p1"]),
        )

    def test_scored_p1_requires_evidence_and_strict_boolean(self):
        for value in ("true", 1, []):
            with self.assertRaises(ValueError):
                report.p1_decision("x", {"x": {"passed": value, "reason": "x", "evidence": ["x.json"]}})
        with self.assertRaises(ValueError):
            report.p1_decision("x", {"x": {"passed": False, "reason": "measured loss"}})
        self.assertIs(
            report.p1_decision("x", {"x": {"passed": False, "reason": "measured loss", "evidence": ["p1.json"]}})[
                "passed"
            ],
            False,
        )

    def test_unknown_capability_does_not_become_pass_from_passing_subcheck(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gate.json").write_text('{"passed": true}')
            decision = report.capability_decision(
                "x",
                {
                    "x": {
                        "passed": None,
                        "reason": "partial coverage",
                        "checks": [{"artifact": "gate.json", "field": "passed"}],
                    }
                },
                root,
            )
            self.assertIsNone(decision["passed"])
            (root / "gate.json").write_text('{"passed": false}')
            decision = report.capability_decision(
                "x",
                {
                    "x": {
                        "passed": None,
                        "reason": "partial coverage",
                        "checks": [{"artifact": "gate.json", "field": "passed"}],
                    }
                },
                root,
            )
            self.assertIs(decision["passed"], False)

    def test_valid_dominance_keeps_unknown_and_records_comparison(self):
        selected, slow = self.row("selected", 11), self.row("slow", 10, None)
        self.prune(slow, selected)
        report.validate_selection([selected, slow], "selected", 0.1)
        self.assertEqual(slow["status"], "not_selected_unscored")
        self.assertIsNone(slow["p1"]["passed"])
        self.assertFalse(slow["eligible"])
        self.assertEqual(slow["failed_gates"], [])
        self.assertEqual(slow["pending_gates"], ["p1"])
        self.assertEqual(slow["p1"]["dominance_verified"]["speed_gap_t_s_u"], 1)

    def test_faster_or_noise_tied_unscored_candidate_blocks_selection(self):
        for speed in (12, 11, 10.95):
            selected, other = self.row("selected", 11), self.row("other", speed, None)
            self.prune(other, selected)
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                report.validate_selection([selected, other], "selected", 0.1)

    def test_missing_pruning_comparator_or_exact_evidence_blocks_selection(self):
        for mutation in ("missing", "ineligible", "uncited"):
            selected, slow = self.row("selected", 11), self.row("slow", 10, None)
            self.prune(slow, selected)
            rows = [selected, slow]
            if mutation == "missing":
                slow["p1"]["dominated_by"] = "absent"
            if mutation == "ineligible":
                rejected = self.row("rejected", 12, False)
                rows.append(rejected)
                slow["p1"]["dominated_by"] = "rejected"
            if mutation == "uncited":
                slow["p1"]["evidence"] = ["unrelated.json"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report.validate_selection(rows, "selected", 0.1)

    def test_missing_decision_cannot_hide_faster_contender(self):
        selected = self.row("selected", 11)
        for p1, capability in ((None, True), (True, None), (None, None)):
            candidate = self.row("unknown", 12, p1, capability)
            with self.subTest(p1=p1, capability=capability), self.assertRaises(ValueError):
                report.validate_selection([selected, candidate], "selected", 0.1)

    def test_explicit_capability_rejection_needs_no_p1_fabrication(self):
        selected, incompatible = self.row("selected", 11), self.row("kv8", 12, None, False)
        report.validate_selection([selected, incompatible], "selected", 0.1)
        self.assertEqual(incompatible["status"], "fail_capability")
        self.assertEqual(incompatible["failed_gates"], ["capability"])
        self.assertEqual(incompatible["pending_gates"], ["p1"])

    def test_unknown_cannot_be_selected_and_duplicate_ids_are_rejected(self):
        selected = self.row("selected", 11, None)
        with self.assertRaises(ValueError):
            report.validate_selection([selected], "selected", 0.1)
        selected = self.row("selected", 11)
        with self.assertRaises(ValueError):
            report.validate_selection([selected, dict(selected)], "selected", 0.1)

    def test_numerically_passing_but_slower_selected_policy_is_rejected(self):
        selected, faster = self.row("selected", 11), self.row("faster", 12)
        with self.assertRaises(ValueError):
            report.validate_selection([selected, faster], "selected", 0.1)


class FinalArtifactsTest(unittest.TestCase):
    def test_cli_preserves_unscored_in_json_csv_and_builds_both_charts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "doc/datatype_sweep"
            output.mkdir(parents=True)
            base_runs = NumericalExecutionSeparationTest.runs()
            for run in base_runs:
                run.update(trace_verified=True, counter_delta={"captures": 0, "releases": 0})
                run["rows"][0].update({"decode_t/s/u": 11.0, "ttft_ms": 100.0})
            base = {
                "completed": True,
                "passed": True,
                "command": ["fixture"],
                "git_head": "fixture",
                "hardware": "CPU artifact fixture",
                "mesh": [1, 4],
                "measurement_regime": "fixture only",
                "build_kwargs": {"precision_config": "fixture_policy.json"},
                "runs": base_runs,
                "runtime_precision": {"requested": {"config_id": "inherited_e4_lofi", "compute_fidelities": {}}},
            }
            baseline_path, slow_path = root / "baseline.json", root / "slow.json"
            baseline_path.write_text(json.dumps(base))
            slow = copy.deepcopy(base)
            slow["runtime_precision"]["requested"]["config_id"] = "slow"
            for run in slow["runs"]:
                run["rows"][0]["decode_t/s/u"] = 10.0
            slow_path.write_text(json.dumps(slow))
            p1_path, capability_path = root / "p1.json", root / "capability.json"
            p1_path.write_text(
                json.dumps(
                    {
                        "inherited_e4_lofi": {"passed": True, "reason": "fixture score", "evidence": [str(p1_path)]},
                        "slow": {
                            "passed": None,
                            "status": "not_scored_dominated",
                            "reason": "fixture slower",
                            "dominated_by": "inherited_e4_lofi",
                            "evidence": [str(baseline_path), str(slow_path)],
                        },
                    }
                )
            )
            capability_path.write_text(
                json.dumps(
                    {
                        cid: {"passed": True, "reason": "fixture capability", "evidence": [str(path)]}
                        for cid, path in (("inherited_e4_lofi", baseline_path), ("slow", slow_path))
                    }
                )
            )
            argv = [
                "report",
                "--selected",
                "inherited_e4_lofi",
                "--inputs",
                str(baseline_path),
                str(slow_path),
                "--p1-decisions",
                str(p1_path),
                "--capability-decisions",
                str(capability_path),
            ]
            with patch.object(report, "__file__", str(root / "tests/report.py")), patch.object(sys, "argv", argv):
                report.main()
            summary = json.loads((output / "sweep_results.json").read_text())
            record = next(row for row in summary["results"] if row["config_id"] == "slow")
            self.assertIsNone(record["p1"]["passed"])
            self.assertEqual(record["status"], "not_selected_unscored")
            self.assertEqual(record["failed_gates"], [])
            self.assertEqual(record["pending_gates"], ["p1"])
            self.assertFalse(record["eligible"])
            self.assertIn("not_selected_unscored", (output / "sweep_results.csv").read_text())
            for metric in ("top1", "top5"):
                self.assertTrue((output / (metric + "_perf_pareto.png")).read_bytes().startswith(b"\x89PNG"))


class CrossProcessRepeatTest(unittest.TestCase):
    @staticmethod
    def measurement(path, speeds):
        runs = []
        for repeat, speed in enumerate(speeds):
            for run in NumericalExecutionSeparationTest.runs():
                run.update(
                    repeat=repeat,
                    passed=True,
                    trace_verified=True,
                    counter_delta={"captures": 0, "releases": 0, "model_replays": 99},
                )
                run["rows"][0].update({"decode_t/s/u": speed, "ttft_ms": speed * 10, "k": 100})
                runs.append(run)
        return {
            "completed": True,
            "passed": True,
            "command": ["fixture", "--output", str(path)],
            "git_head": "fixture",
            "hardware": "CPU artifact fixture",
            "mesh": [1, 4],
            "environment": {"TT_VISIBLE_DEVICES": "0,1,2,3"},
            "measurement_regime": "serial; traced teacher forcing; fixture only",
            "capacity": 262144,
            "build_kwargs": {"precision_config": "fixture_policy.json", "max_seq_len": 262144},
            "runtime_precision": {
                "requested": {
                    "config_id": "inherited_e4_lofi",
                    "compute_fidelities": {"shared": "LoFi"},
                    "weight_groups": {"shared": "bfloat8_b"},
                }
            },
            "runs": runs,
        }

    def test_cli_merges_processes_without_mean_of_means_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "doc/datatype_sweep"
            output.mkdir(parents=True)
            paths = [root / "baseline_first.json", root / "baseline_second.json"]
            originals = [self.measurement(path, speeds) for path, speeds in zip(paths, ([10.0, 12.0], [14.0]))]
            for path, raw in zip(paths, originals):
                path.write_text(json.dumps(raw))
                path.with_suffix(".provenance.json").write_text(json.dumps({"command": raw["command"]}))
            p1, capability = root / "p1.json", root / "capability.json"
            decision = {"inherited_e4_lofi": {"passed": True, "reason": "fixture only", "evidence": [str(paths[0])]}}
            p1.write_text(json.dumps(decision))
            capability.write_text(json.dumps(decision))
            argv = [
                "report",
                "--selected",
                "inherited_e4_lofi",
                "--inputs",
                *map(str, paths),
                "--p1-decisions",
                str(p1),
                "--capability-decisions",
                str(capability),
            ]
            with patch.object(report, "__file__", str(root / "tests/report.py")), patch.object(sys, "argv", argv):
                report.main()
            summary = json.loads((output / "sweep_results.json").read_text())
            self.assertEqual(len(summary["results"]), 1)
            row = summary["results"][0]
            self.assertEqual(row["teacher_forcing_decode_t_s_u"], 12.0)
            self.assertEqual(row["ttft_ms"], 120.0)
            self.assertEqual(row["aime_timed_sample_count"], 3)
            self.assertEqual(row["measurement_input_count"], 2)
            self.assertEqual(row["decode_repeat_range_t_s_u"], [10.0, 14.0])
            self.assertEqual(summary["baseline_repeat_spread_t_s_u"], 4.0)
            self.assertEqual(row["evidence_paths"], list(map(str, paths)))
            self.assertEqual(
                [source["command"] for source in row["source_measurements"]], [raw["command"] for raw in originals]
            )
            self.assertEqual(
                [source["provenance_artifact"] for source in row["source_measurements"]],
                [str(path.with_suffix(".provenance.json")) for path in paths],
            )
            self.assertEqual(len(row["references"]), 12)
            for source_index, raw in enumerate(originals):
                preserved = [r for r in row["references"] if r["source_artifact"] == str(paths[source_index])]
                self.assertEqual([r["source_run_index"] for r in preserved], list(range(len(raw["runs"]))))
                for actual, original in zip(preserved, raw["runs"]):
                    for key, value in original.items():
                        self.assertEqual(actual[key], value)
            # One row and one key entry per policy, despite multiple processes.
            self.assertEqual(len((output / "sweep_results.csv").read_text().splitlines()), 2)
            for metric in ("top1", "top5"):
                self.assertTrue((output / f"{metric}_perf_pareto.png").exists())

    def test_mismatched_policy_regime_environment_geometry_and_coverage_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("first.json", "second.json")]
            first = self.measurement(paths[0], [10.0])
            paths[0].write_text(json.dumps(first))
            for mismatch in (
                "policy",
                "regime",
                "environment",
                "hardware",
                "mesh",
                "capacity",
                "build",
                "coverage",
                "counts",
            ):
                second = self.measurement(paths[1], [11.0])
                if mismatch == "policy":
                    second["runtime_precision"]["requested"]["compute_fidelities"]["shared"] = "HiFi2"
                elif mismatch == "regime":
                    second["measurement_regime"] += "; profiler enabled"
                elif mismatch == "environment":
                    second["environment"]["TT_METAL_WATCHER"] = "1"
                elif mismatch == "hardware":
                    second["hardware"] = "different chips"
                elif mismatch == "mesh":
                    second["mesh"] = [1, 8]
                elif mismatch == "capacity":
                    second["capacity"] = 32768
                elif mismatch == "build":
                    second["build_kwargs"]["max_seq_len"] = 32768
                elif mismatch == "coverage":
                    second["runs"].pop()
                elif mismatch == "counts":
                    second["runs"][0]["rows"][0]["total"] = 99
                paths[1].write_text(json.dumps(second))
                with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                    report.measurement_groups(paths)

    def test_failed_incomplete_untraced_partial_and_duplicate_measurements_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("first.json", "second.json")]
            first = self.measurement(paths[0], [10.0])
            paths[0].write_text(json.dumps(first))
            for defect in (
                "failed",
                "incomplete",
                "row_failed",
                "untraced",
                "capture",
                "no_replays",
                "partial_repeat",
                "copied_report",
            ):
                second = self.measurement(paths[1], [11.0])
                if defect == "failed":
                    second["passed"] = False
                elif defect == "incomplete":
                    second["completed"] = False
                elif defect == "row_failed":
                    second["runs"][0]["passed"] = False
                elif defect == "untraced":
                    second["runs"][1]["trace_verified"] = False
                elif defect == "capture":
                    second["runs"][1]["counter_delta"]["captures"] = 1
                elif defect == "no_replays":
                    second["runs"][1]["counter_delta"]["model_replays"] = 0
                elif defect == "partial_repeat":
                    second = self.measurement(paths[1], [11.0, 12.0])
                    second["runs"].pop()
                elif defect == "copied_report":
                    second = first
                paths[1].write_text(json.dumps(second))
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    report.measurement_groups(paths)
            with self.assertRaises(ValueError):
                report.measurement_groups([paths[0], paths[0]])

    def test_dominance_requires_all_repeat_artifacts_and_resolves_decision_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = UnscoredSelectionTest.row("selected", 12.0)
            slow = UnscoredSelectionTest.row("slow", 10.0, None)
            for row in (selected, slow):
                row["evidence_paths"] = [str(root / f"{row['config_id']}_{i}.json") for i in range(2)]
                for path in row["evidence_paths"]:
                    Path(path).write_text("{}")
            UnscoredSelectionTest.prune(slow, selected)
            slow["p1"].update(evidence_base_dir=str(root), evidence=["slow_0.json", "selected_0.json"])
            with self.assertRaises(ValueError):
                report.validate_selection([selected, slow], "selected", 0.1)
            slow["p1"]["evidence"] = [Path(path).name for row in (selected, slow) for path in row["evidence_paths"]]
            report.validate_selection([selected, slow], "selected", 0.1)
            self.assertEqual(slow["status"], "not_selected_unscored")

    def test_failed_or_unfinished_process_provenance_cannot_hide_behind_completed_json(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("first.json", "second.json")]
            for path, speed in zip(paths, (10.0, 11.0)):
                path.write_text(json.dumps(self.measurement(path, [speed])))
            sidecar = paths[1].with_suffix(".provenance.json")
            for provenance in ({"status": "finished", "exit_code": 1}, {"status": "running", "exit_code": None}):
                sidecar.write_text(json.dumps(provenance))
                with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                    report.measurement_groups(paths)


class NumericalExecutionSeparationTest(unittest.TestCase):
    @staticmethod
    def runs():
        row = {
            "total": 100,
            "matches_top1": 95,
            "matches_top5": 99,
            "matches_top100": 100,
            "top1": 0.95,
            "top5": 0.99,
            "top100": 1.0,
        }
        return [
            {"reference": reference, "runner": runner, "rows": [dict(row)]}
            for reference in ("aime24.refpt", "natural.refpt")
            for runner in ("run_prefill_check", "run_teacher_forcing")
        ]

    def test_late_capture_failure_does_not_erase_numerical_pass(self):
        runs = self.runs()
        accuracy = report.numerical_accuracy(runs, runs)
        self.assertTrue(accuracy["passed"])
        status, failures, pending = report.acceptance_status(
            accuracy=accuracy["passed"], execution=False, p1=True, traced=True, capability={"passed": False}
        )
        self.assertEqual(status, "fail_capability")
        self.assertEqual(failures, ["execution", "capability"])

    def test_partial_rows_cannot_claim_full_accuracy(self):
        baseline = self.runs()
        for candidate in ([], baseline[:3]):
            accuracy = report.numerical_accuracy(candidate, baseline)
            self.assertFalse(accuracy["passed"])
            self.assertFalse(accuracy["coverage_complete"])
            self.assertTrue(accuracy["missing_reference_runners"])

    def test_shortened_reference_counts_fail_even_when_fractions_match(self):
        baseline, candidate = self.runs(), self.runs()
        row = candidate[0]["rows"][0]
        row.update(total=20, matches_top1=19, matches_top5=20, matches_top100=20, top5=1.0)
        accuracy = report.numerical_accuracy(candidate, baseline)
        self.assertFalse(accuracy["passed"])
        self.assertTrue(accuracy["coverage_complete"])
        self.assertTrue(accuracy["row_failures"])

    def test_claimed_fraction_cannot_override_bad_counts(self):
        baseline, candidate = self.runs(), self.runs()
        candidate[1]["rows"][0]["matches_top1"] = 89
        self.assertFalse(report.numerical_accuracy(candidate, baseline)["passed"])

    def test_repeated_baseline_does_not_require_candidate_repeats(self):
        baseline, candidate = self.runs(), self.runs()
        self.assertTrue(report.numerical_accuracy(candidate, baseline + baseline)["passed"])


if __name__ == "__main__":
    unittest.main()
