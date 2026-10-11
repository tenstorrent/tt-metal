#!/usr/bin/env python3
"""Host regression tests for the source-bound sweep claim filter."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sweep_admission_report as report


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.root.mkdir()
        self.ledger = Path(self.temp.name) / "ledger.tsv"
        self.manifest = {
            "tt_metal_head": {"returncode": 0, "stdout": "a" * 40 + "\n"},
            "tt_metal_status": {"returncode": 0, "stdout": ""},
            "tt_metal_diff": {"returncode": 0, "stdout": ""},
            "compiler": {
                name: {"sha256": value * 64}
                for name, value in (("driver", "b"), ("cc1plus", "c"))
            },
            "rows": [
                {"op": op, "kind": "full2x2"}
                for op in ("bounded", "stale", "unknown", "failed")
            ],
        }
        self.summary = {
            "selected_rows": 4,
            "completed_rows": 4,
            "counts": {"PASS": 3, "FAIL": 1},
            "status": "FAIL",
        }
        self.rows = [
            {
                "op": row["op"],
                "kind": row["kind"],
                "status": "FAIL" if row["op"] == "failed" else "PASS",
                "sem_off": "100",
                "sem_on": "80",
                "hand_off": "100",
                "hand_on": "100",
            }
            for row in self.manifest["rows"]
        ]
        self.metadata = {
            "campaign_source": "a" * 40,
            "compiler_driver_sha256": "b" * 64,
            "compiler_cc1plus_sha256": "c" * 64,
            "campaign_fully_admitted": "FALSE",
        }
        self.decisions = []
        for op, decision in (
            ("bounded", "BOUNDED_PARTIAL_ADMITTED"),
            ("stale", "REFUSED_STALE_SEMANTICS"),
            ("failed", "BOUNDED_PARTIAL_ADMITTED"),
        ):
            self.decisions.append(
                {
                    "row": op,
                    "raw_disposition": "RETAIN",
                    "comparison_admission": decision,
                    "replacement_scope": "partial fixture",
                    "validation_status": "ACTUAL_PARTIAL_PASS",
                    "actual_counts": "OFF:9_PASS/0_FAIL;ON:9_PASS/0_FAIL;sem_n_out=0;hand_n_out=0;sem_max_bf16_ulp=0;hand_max_bf16_ulp=0",
                    "unfulfilled_requirements": "NOT_EXHAUSTIVE",
                    "reason": "fixture",
                    "evidence_ref": "fixture/path",
                }
            )

    def write(self):
        for name, value in (
            ("manifest.json", self.manifest),
            ("summary.json", self.summary),
        ):
            (self.root / name).write_text(json.dumps(value))
        with (self.root / "results.tsv").open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(self.rows[0]), delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(self.rows)
        with self.ledger.open("w") as stream:
            self.metadata["terminal_results_sha256"] = report.sha256(
                self.root / "results.tsv"
            )
            self.metadata["terminal_summary_sha256"] = report.sha256(
                self.root / "summary.json"
            )
            self.metadata["campaign_manifest_sha256"] = report.sha256(
                self.root / "manifest.json"
            )
            for key, value in self.metadata.items():
                stream.write(f"# {key}\t{value}\n")
            writer = csv.DictWriter(
                stream, fieldnames=sorted(report.LEDGER_FIELDS), delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(self.decisions)

    def build(self):
        self.write()
        return report.build_report(self.root, self.ledger)

    def test_join_preserves_raw_and_never_promotes_partial(self):
        self.write()
        before = {path: path.read_bytes() for path in self.root.iterdir()}
        result = report.build_report(self.root, self.ledger)
        self.assertFalse(result["campaign_fully_admitted"])
        self.assertEqual(result["raw_counts"], self.summary["counts"])
        self.assertEqual(result["rows"]["unknown"]["effective_admission"], "UNREVIEWED")
        self.assertEqual(
            set(result["comparisons"]["bounded_partial_on_vs_hand_on"]), {"bounded"}
        )
        self.assertEqual(set(result["rows"]), {"bounded", "stale", "unknown", "failed"})
        self.assertEqual(
            result["rows"]["failed"]["effective_admission"], "RAW_NOT_PASS"
        )
        self.assertEqual(
            before, {path: path.read_bytes() for path in self.root.iterdir()}
        )
        self.assertEqual(
            result["artifacts_sha256"]["results.tsv"],
            report.sha256(self.root / "results.tsv"),
        )

    def test_partial_and_incoherent_campaigns_refused(self):
        for key, value in (
            ("status", "RUNNING"),
            ("completed_rows", 3),
            ("selected_rows", 5),
            ("counts", {"PASS": 4}),
        ):
            with self.subTest(key=key):
                saved = copy.deepcopy(self.summary)
                self.summary[key] = value
                with self.assertRaises(ValueError):
                    self.build()
                self.summary = saved

    def test_source_compiler_and_dirty_source_refused(self):
        original = copy.deepcopy(self.manifest)
        for key in ("tt_metal_head", "tt_metal_status", "tt_metal_diff"):
            with self.subTest(key=key):
                self.manifest[key]["stdout"] = "changed"
                with self.assertRaises(ValueError):
                    self.build()
                self.manifest = copy.deepcopy(original)

        for name in ("driver", "cc1plus"):
            with self.subTest(name=name):
                self.manifest["compiler"][name]["sha256"] = "d" * 64
                with self.assertRaises(ValueError):
                    self.build()
                self.manifest = copy.deepcopy(original)

    def test_admitted_ledger_cannot_report_failed_validation(self):
        original = copy.deepcopy(self.decisions)
        for key, value in (
            ("validation_status", "ACTUAL_FAIL"),
            ("actual_counts", "OFF:0_PASS/1_FAIL;ON:0_PASS/1_FAIL"),
            (
                "actual_counts",
                original[0]["actual_counts"].replace("sem_n_out=0", "sem_n_out=1"),
            ),
            (
                "actual_counts",
                original[0]["actual_counts"].replace(
                    "sem_max_bf16_ulp=0", "sem_max_bf16_ulp=1"
                ),
            ),
        ):
            with self.subTest(key=key, value=value):
                self.decisions[0][key] = value
                with self.assertRaises(ValueError):
                    self.build()
                self.decisions = copy.deepcopy(original)

    def test_all_skip_has_no_effective_admission(self):
        for row in self.rows:
            row["status"] = "SKIP"
        self.summary.update(status="PASS", counts={"SKIP": 4})
        result = self.build()
        self.assertEqual(result["completion"], "COMPLETE_NO_PASS")
        self.assertTrue(
            all(
                row["effective_admission"] == "RAW_NOT_PASS"
                for row in result["rows"].values()
            )
        )
        self.assertTrue(all(not rows for rows in result["comparisons"].values()))

    def test_manifest_hash_binds_flags_seed_device_and_population(self):
        for key, value in (
            ("flags", {"ON": "-DWRONG"}),
            ("seed", 999),
            ("device_index", 77),
            ("corpus_sha256", "d" * 64),
        ):
            with self.subTest(key=key):
                self.write()
                changed = dict(self.manifest, **{key: value})
                (self.root / "manifest.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "manifest does not match"):
                    report.build_report(self.root, self.ledger)

    def test_hashes_cover_parsed_snapshot_not_later_file_contents(self):
        self.write()
        target = self.root / "results.tsv"
        original = target.read_bytes()
        read_bytes = Path.read_bytes
        reads = []

        def mutate_after_read(path):
            data = read_bytes(path)
            reads.append(path)
            if path == target:
                target.write_bytes(b"changed after snapshot")
            return data

        with mock.patch.object(Path, "read_bytes", mutate_after_read):
            result = report.build_report(self.root, self.ledger)
        self.assertEqual(
            result["artifacts_sha256"]["results.tsv"],
            hashlib.sha256(original).hexdigest(),
        )
        self.assertEqual(len(reads), len(set(reads)))

    def test_terminal_metrics_and_summary_are_bound(self):
        for filename in ("results.tsv", "summary.json"):
            with self.subTest(filename=filename):
                self.write()
                target = self.root / filename
                if filename == "results.tsv":
                    target.write_text(target.read_text().replace("\t80\t", "\t1\t"))
                else:
                    target.write_text(json.dumps(dict(self.summary, formal="PASS")))
                with self.assertRaisesRegex(ValueError, "terminal .* does not match"):
                    report.build_report(self.root, self.ledger)

    def test_unknown_duplicate_and_extra_ledger_rows_refused(self):
        original = copy.deepcopy(self.decisions)
        self.decisions[0]["comparison_admission"] = "FULL_WIN"
        with self.assertRaises(ValueError):
            self.build()
        self.decisions = copy.deepcopy(original) + [copy.deepcopy(original[0])]
        with self.assertRaises(ValueError):
            self.build()
        self.decisions[-1]["row"] = "not-in-campaign"
        with self.assertRaises(ValueError):
            self.build()

    def test_raw_duplicates_and_missing_manifest_members_refused(self):
        self.rows[1]["op"] = "bounded"
        with self.assertRaises(ValueError):
            self.build()
        self.rows[1]["op"] = "not-in-manifest"
        with self.assertRaises(ValueError):
            self.build()

    def test_bad_metrics_and_missing_fourth_arm_refused(self):
        for value in ("", "nan", "inf", "0", "-1"):
            with self.subTest(value=value):
                self.rows[0]["hand_off"] = value
                with self.assertRaises(ValueError):
                    self.build()

    def test_causal_is_not_a_paired_or_numeric_admission(self):
        self.decisions[0]["comparison_admission"] = "CAUSAL_OFF_ON_ONLY"
        self.rows[0]["kind"] = self.manifest["rows"][0]["kind"] = "semantic"
        result = self.build()
        self.assertEqual(result["comparisons"]["bounded_partial_on_vs_hand_on"], {})
        self.assertEqual(
            set(result["comparisons"]["raw_causal_on_vs_off"]), {"bounded"}
        )

    def test_bounded_comparison_requires_full2x2(self):
        self.rows[0]["kind"] = self.manifest["rows"][0]["kind"] = "semantic"
        with self.assertRaises(ValueError):
            self.build()

    def test_output_cannot_overwrite_inputs_or_existing_file(self):
        self.write()
        for target in (self.root / "results.tsv", self.ledger):
            before = target.read_bytes()
            with self.assertRaises(SystemExit) as caught:
                report.main(
                    [
                        "--current",
                        str(self.root),
                        "--ledger",
                        str(self.ledger),
                        "--out",
                        str(target),
                    ]
                )
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
