# SPDX-License-Identifier: Apache-2.0
"""Host-only final-report gate checks using temporary evidence and no servers."""

import importlib.util
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "doc/benchmark/finalize_report.py"
SPEC = importlib.util.spec_from_file_location("benchmark_finalizer_host_test", SOURCE)
FINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINAL)


class FinalizationTests(unittest.TestCase):
    def test_scored_documents_are_unique_across_upstream_filters(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            task = "gpqa_diamond_cot_zeroshot"
            folder = run / task
            folder.mkdir()
            rows = [dict(doc_id=i, filter=name) for name in ("strict-match", "flexible-extract") for i in range(128)]
            (folder / f"samples_{task}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            FINAL.write_json(folder / "results.json", {"results": {task: {"exact_match,flexible-extract": 0.5}}})
            manifest = dict(groups={task: dict(tasks=[task], sample_count=128, population=198)})
            counts = FINAL.response_counts(run, manifest)
            self.assertEqual(counts[task]["scored"], 128)
            self.assertEqual(counts[task]["sample_file_counts"][task], 256)
            self.assertEqual(counts[task]["sample_document_counts"][task], 128)
            self.assertEqual(counts[task]["sample_filter_counts"][task], {"strict-match": 128, "flexible-extract": 128})
            text = self.augment(
                run,
                counts=counts,
                manifest=manifest,
                tasks=[task],
                text=f"| {task} | 256 / 198 | metric | 50 | — | — | source |\n",
            )
            self.assertIn(f"| {task} | 128 / 198 |", text)
            self.assertNotIn("256 / 198", text)

    def augment(self, run, *, counts=None, manifest=None, tasks=None, text=""):
        return FINAL.augment_report(
            text,
            run,
            dict(tasks=tasks or []),
            manifest or dict(groups={}),
            dict(status="completed", performance={"1": {}, "32": {}}),
            counts or {},
            None,
            {},
            None,
            dict(evidence=dict(exit_code=0), context=dict(exit_code=0), context_bytes_unchanged=True),
        )

    def test_configuration_snapshots_and_current_accounting_scope_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for slots, version in ((1, "native_b1_gdn_payload_v2"), (32, "compact_view_prefix_broadcast_v3")):
                name = f"perf-b{slots}-server-configuration.json"
                FINAL.write_json(run / f"perf-b{slots}-server.json", dict(configuration_evidence=name))
                FINAL.write_json(run / name, dict(state=dict(max_num_seqs=slots, startup_seconds=slots + 0.25)))
                FINAL.write_json(
                    run / f"roofline-b{slots}-evidence.json",
                    dict(
                        work=dict(
                            details=dict(
                                execution_version=version,
                                gdn_traffic_scope="Other TTNN temporary/reload and collective staging traffic remains analytical uncertainty.",
                                pooled_gather_partitions={"decode": [dict(layer=3, workers=110)]},
                            ),
                            decode=dict(
                                estimate_kind="analytical_upper_routing"
                                if slots == 32
                                else "analytical_native_gdn_payload"
                            ),
                        )
                    ),
                )
            text = self.augment(run)
            for slots in (1, 32):
                self.assertIn(f"(perf-b{slots}-server-configuration.json)", text)
                self.assertIn(f"slots={slots}, startup={slots + 0.25} seconds", text)
            self.assertIn("native_b1_gdn_payload_v2", text)
            self.assertIn("compact_view_prefix_broadcast_v3", text)
            self.assertIn("not measured DRAM traffic or a bound on all hardware transactions", text)
            self.assertIn("collective staging traffic remains analytical uncertainty", text)
            self.assertIn("nominal hardware-peak percentage, not an available-grid utilization percentage", text)
            self.assertIn("110 available workers per ASIC", text)
            saved = FINAL.read_json(run / "accounting_summary.json")
            self.assertIn("collective staging", saved["roofline-b1-evidence.json"]["gdn_traffic_scope"])
            self.assertEqual(saved["roofline-b1-evidence.json"]["compute_peak_basis"]["nominal_cores_per_asic"], 120)
            self.assertEqual(
                saved["roofline-b1-evidence.json"]["compute_peak_basis"]["observed_gather_workers_per_asic"],
                {"decode": [110]},
            )

    def run_finalizer(self, *, elapsed=100, check_code=0, context_same=True, audit_code=0):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        run = root / "run"
        run.mkdir()
        FINAL.write_json(run / "summary.json", dict(status="completed", elapsed_seconds=80, performance={}))
        FINAL.write_json(run / "manifest.json", dict(groups={}))
        FINAL.write_json(run / "run_config.json", dict(tasks=[]))
        FINAL.write_json(run / "input_audit_status.json", dict(exit_code=audit_code))
        FINAL.write_json(run / "input_audit.json", dict(payloads=0))
        invocation = root / "invocation.json"
        FINAL.write_json(invocation, dict(invocation_monotonic=1000, invocation_wall_time=time.time() - 1))
        checks = dict(
            evidence=dict(exit_code=check_code), context=dict(exit_code=0), context_bytes_unchanged=context_same
        )
        report_module = types.ModuleType("benchmark_stage.report")

        def write_report(folder, config, summary):
            (folder / "REPORT.md").write_text(f"Status: {summary['status']}. Accuracy concurrency: 32.\n")

        report_module.write_report = write_report

        def completed_report_time():
            # Timing must be sampled after the final report body exists.
            self.assertIn("Retained counts and protocol", (run / "REPORT.md").read_text())
            return 1000 + elapsed

        with (
            patch.object(FINAL, "ROOT", root),
            patch.object(FINAL, "checks_for", return_value=checks),
            patch.object(FINAL.time, "monotonic", side_effect=completed_report_time),
            patch.dict(sys.modules, {"benchmark_stage.report": report_module}),
        ):
            code = 0
            try:
                FINAL.main(["--run-dir", str(run), "--invocation", str(invocation)])
            except SystemExit as exc:
                code = exc.code
        return run, code

    def test_success_clock_includes_written_report(self):
        run, code = self.run_finalizer()
        self.assertEqual(code, 0)
        completion = FINAL.read_json(run / "report_completion.json")
        self.assertTrue(completion["completion_requirements_passed"])
        self.assertEqual(completion["elapsed_seconds_through_checks_and_report"], 100)
        self.assertEqual(FINAL.read_json(run / "runner_summary.json")["elapsed_seconds"], 80)

    def test_check_context_and_audit_failures_cannot_pass(self):
        for settings in (dict(check_code=2), dict(context_same=False), dict(audit_code=1)):
            with self.subTest(settings=settings):
                run, code = self.run_finalizer(**settings)
                self.assertEqual(code, 2)
                self.assertEqual(FINAL.read_json(run / "summary.json")["status"], "failed")
                self.assertIn("Status: failed.", (run / "REPORT.md").read_text())
                self.assertFalse(FINAL.read_json(run / "report_completion.json")["completion_requirements_passed"])

    def test_hour_boundary_cannot_pass(self):
        run, code = self.run_finalizer(elapsed=3600)
        self.assertEqual(code, 2)
        completion = FINAL.read_json(run / "report_completion.json")
        self.assertFalse(completion["within_one_hour"])
        self.assertEqual(completion["status"], "failed")
        self.assertIn("Status: failed.", (run / "REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
