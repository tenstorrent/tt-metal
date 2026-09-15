# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only standard-library tests. All record mutations are in memory."""

import copy
import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_final_smokes as F


def fixture(label):
    spec = F.SPECS[label]
    baseline, _ = F.read_record(F.HERE / spec["baseline"])
    record = copy.deepcopy(baseline)
    manifest = record[0] if isinstance(record, list) else record
    manifest.get("args", manifest)["label"] = label
    # Synthetic current-source record, never written or represented as measured.
    names = set(manifest["source_sha256"]) | F.required_sources(spec, manifest)
    manifest["source_sha256"] = {p: hashlib.sha256((F.ROOT / p).read_bytes()).hexdigest() for p in names}
    return spec, record, baseline


def evaluate(spec, record, baseline=None):
    e = F.V.Evidence("in-memory-only", spec["family"])
    F.validate_case(F.CurrentAudit(F.ROOT, F.HERE), e, spec, record, baseline)
    return e.export()


def first_row(record):
    return record[1] if isinstance(record, list) else record


class FinalSmokeTests(unittest.TestCase):
    def test_all_twenty_attention_baseline_schemas(self):
        rows = 0
        for label, spec in F.SPECS.items():
            if not spec["baseline"]:
                continue
            with self.subTest(label=label):
                spec, record, baseline = fixture(label)
                result = evaluate(spec, record, baseline)
                self.assertEqual(result["status"], "PASS", result["failures"])
                self.assertFalse(result["provenance"]["historical_fallback"])
                rows += result["summary"]["result_rows"]
        self.assertEqual(rows, 43)

    def test_complete_output_hash_mutation(self):
        spec, record, baseline = fixture("final-vaxis-D-v1")
        record[1]["output_sha256"] = "0" * 64
        self.assertEqual(evaluate(spec, record, baseline)["status"], "FAIL")

    def test_current_source_drift_never_uses_snapshot(self):
        spec, record, baseline = fixture("final-recipe-v1")
        record[0]["source_sha256"][F.PREFIX + spec["driver"]] = "0" * 64
        with mock.patch.object(
            F.V.Audit, "assertion_witness", side_effect=AssertionError("Historical fallback forbidden")
        ):
            result = evaluate(spec, record, baseline)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["provenance"]["status"], "FAIL")

    def test_missing_principal_source(self):
        spec, record, baseline = fixture("final-captured-v1")
        del record[0]["source_sha256"][F.PREFIX + spec["driver"]]
        result = evaluate(spec, record, baseline)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(F.PREFIX + spec["driver"], result["provenance"]["principal_omissions"])

    def test_finite_replay_prep_scope_and_flops_gates(self):
        for field, value in (
            ("finite", False),
            ("trace_equal", False),
            ("sampled_query_rows", [0]),
            ("useful_flops", 0),
            ("sources_unchanged", False),
        ):
            spec, record, baseline = fixture("final-recipe-v1")
            record[1][field] = value
            if field == "finite":
                record[1]["all_output_finite"] = False
            with self.subTest(field=field):
                self.assertEqual(evaluate(spec, record, baseline)["status"], "FAIL")
        spec, record, baseline = fixture("final-recipe-v1")
        record[1]["kernel"]["preprocessing_checks"][0]["mismatch"] = 1
        self.assertEqual(evaluate(spec, record, baseline)["status"], "FAIL")

    def test_disabled_assertion_only_preprocessing(self):
        spec, record, baseline = fixture("final-hi2-native-v1")
        record["check_preprocess"] = False
        self.assertEqual(evaluate(spec, record, baseline)["status"], "FAIL")

    def test_bf16_control_trace_hashes_and_inputs(self):
        for field, value in (("cpu_inputs_unchanged", False), ("replay_output_sha256", ["0" * 64] * 2)):
            spec, record, baseline = fixture("final-hi2-bf16-lut-v1")
            record[field] = value
            self.assertEqual(evaluate(spec, record, baseline)["status"], "FAIL")

    def test_no_timing_and_no_global_accuracy_cutoff(self):
        spec, record, baseline = fixture("final-hi2-native-v1")
        record["attention"]["median_ms"] = 1.0
        self.assertEqual(evaluate(spec, record, baseline)["status"], "FAIL")
        spec, record, baseline = fixture("final-hi2-native-v1")
        record["accuracy"]["l2_pct"] = 1000.0
        self.assertEqual(evaluate(spec, record, baseline)["status"], "PASS")

    def test_incomplete_and_missing_baseline_pending(self):
        spec, record, baseline = fixture("final-captured-v1")
        self.assertEqual(evaluate(spec, record[:-1], baseline)["status"], "PENDING")
        self.assertEqual(evaluate(spec, record, None)["status"], "PENDING")
        changed = record[:1] + record[2:]
        self.assertEqual(evaluate(spec, changed, baseline)["status"], "FAIL")

    def test_plan_and_missing_outputs(self):
        plan, _ = F.read_record(F.PLAN)
        self.assertEqual(F.plan_checks(F.ROOT, plan), [])
        changed = copy.deepcopy(plan)
        changed["cases"][0]["args"] += " --nonexistent-option"
        self.assertTrue(F.plan_checks(F.ROOT, changed))
        changed["cases"][0]["args"] = "--iters"
        self.assertTrue(F.plan_checks(F.ROOT, changed))
        # Mock absence only for expected final outputs; no filesystem writes.
        outputs = {F.HERE / Path(s["output"]).relative_to(F.PREFIX) for s in F.SPECS.values()}
        original = Path.is_file
        with mock.patch.object(Path, "is_file", lambda p: False if p in outputs else original(p)):
            result = F.run()
        self.assertEqual(result["final_status"], "PENDING")
        self.assertEqual(len(result["evidence"]), 24)
        self.assertTrue(all(e["status"] == "PENDING" for e in result["evidence"]))

    def test_b4_exact_oracle_and_explicit_limitations(self):
        for distribution in ("normal", "thresholds", "wide", "zeros"):
            spec = F.SPECS["final-b4-" + distribution + "-v1"]
            record = dict(
                label=spec["label"],
                length=1024,
                cores=4,
                actual_cores=4,
                batch=4,
                fp32_dst=False,
                output_format="b4",
                distribution=distribution,
                seed=1240,
                host_only=False,
                iters=0,
                mismatch=0,
                oracle_mismatch=0,
                numel=131072,
                median_ms=None,
                replay_ms=[],
                read_write_GBps=None,
                quantization_l2_pct=1.0,
            )
            record["source_sha256"] = {
                p: hashlib.sha256((F.ROOT / p).read_bytes()).hexdigest() for p in F.required_sources(spec, record)
            }
            result = evaluate(spec, record)
            self.assertEqual(result["status"], "PASS", result["failures"])
            self.assertEqual(result["summary"]["complete_output_hash_comparison"], "NOT_AVAILABLE")
            record["oracle_mismatch"] = 1
            self.assertEqual(evaluate(spec, record)["status"], "FAIL")

    def test_json_nan_and_duplicate_keys_rejected(self):
        for malformed in ('{"x": NaN}', '{"x": 1, "x": 2}'):
            with self.assertRaises(ValueError):
                F.V.decode(malformed)


if __name__ == "__main__":
    unittest.main()
