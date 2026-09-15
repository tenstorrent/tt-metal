# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standard-library record mutations only; no producers/devices or file writes."""

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_exp_stress as S


def record(route):
    label = f"exp-stress-{route}-normal-s1240-v1"
    data, _ = S.F.read_record(S.HERE / (label + ".json"))
    return S.SPECS[label], data


def evaluate(spec, data):
    e = S.V.Evidence("memory", "test")
    S.validate_case(S.F.CurrentAudit(S.ROOT, S.HERE), e, spec, data)
    return e.export()


class StressAuditTests(unittest.TestCase):
    def test_four_current_schema_controls(self):
        for route in S.ROUTES:
            spec, data = record(route)
            result = evaluate(spec, data)
            self.assertEqual(result["status"], "PASS", result["failures"])
            if route.startswith("lofi"):
                self.assertIn("NOT_RUN", result["summary"]["trace_replay"])
                self.assertIn("NOT_RECORDED", result["summary"]["input_integrity"])

    def test_undefined_pcc_and_large_finite_error_allowed(self):
        spec, data = record("hi2_lut")
        data["accuracy"].update(pcc=None, l2_pct=1000.0)
        self.assertEqual(evaluate(spec, data)["status"], "PASS")

    def test_exp_flag_prep_finite_hash_scope_failures(self):
        for field, value in (
            ("finite", False),
            ("sources_unchanged", False),
            ("sampled_query_rows", [0]),
            ("original_input_sha256", ["0" * 64] * 3),
            ("trace_equal", False),
        ):
            spec, data = record("hi2_lut")
            data[field] = value
            self.assertEqual(evaluate(spec, data)["status"], "FAIL")
        spec, data = record("lofi_lut")
        data["preprocessing_checks"][0]["mismatch"] = 1
        self.assertEqual(evaluate(spec, data)["status"], "FAIL")
        spec, data = record("lofi_lut")
        del data["defines"]["SDPA_LOFI_LUT_MACRO"]
        self.assertEqual(evaluate(spec, data)["status"], "FAIL")

    def test_source_drift_is_hard_failure(self):
        spec, data = record("lofi_native")
        data["source_sha256"][S.PREFIX + spec["driver"]] = "0" * 64
        self.assertEqual(evaluate(spec, data)["status"], "FAIL")

    def test_pair_and_repeat_hash_mutations(self):
        _, native = record("hi2_native")
        _, lut = record("hi2_lut")
        e = S.V.Evidence("memory", "pair")
        S.compare_pair(e, native, lut, True)
        self.assertFalse(e.failures)
        lut["preprocessing_checks"][0]["output_sha256"] = "0" * 64
        S.compare_pair(e, native, lut, True)
        self.assertTrue(e.failures)
        second = copy.deepcopy(native)
        second["label"] = second["label"][:-3] + "-v2"
        e = S.V.Evidence("memory", "repeat")
        S.compare_repeat(e, native, second)
        self.assertFalse(e.failures)
        second["output_sha256"] = "0" * 64
        S.compare_repeat(e, native, second)
        self.assertTrue(e.failures)

    def test_plan_coverage_and_long_scope(self):
        plan, _ = S.F.read_record(S.PLAN)
        self.assertEqual(S.check_plan(plan), [])
        self.assertEqual(len(S.plan_specs(plan)), 160)
        missing = copy.deepcopy(plan)
        missing["cases"].pop()
        with self.assertRaises(ValueError):
            S.plan_specs(missing)
        long_plan = dict(cases=[])
        for row in plan["cases"]:
            opt = S.options(S.shlex.split(row["args"]))
            if opt["--distribution"][0] not in S.LONG_DISTRIBUTIONS:
                continue
            opt.update({"--length": ["32768"], "--sample-rows": ["128"], "--cores": ["22"]})
            opt["--label"] = [row["label"].replace("exp-stress-", "exp-stress-long-")]
            argv = [v for key, vals in opt.items() for v in [key, *vals]]
            label = opt["--label"][0]
            long_plan["cases"].append(
                dict(
                    label=label,
                    driver=row["driver"],
                    args=S.shlex.join(argv),
                    output=S.PREFIX + label + ".json",
                    command=S.shlex.join(["python_env/bin/python", "-B", S.PREFIX + row["driver"], *argv]),
                )
            )
        self.assertEqual(S.check_plan(long_plan), [])
        specs = S.plan_specs(long_plan, "-repeat")
        self.assertEqual(len(specs), 96)
        self.assertTrue(all(s["length"] == 32768 and s["sample_rows"] == 128 for s in specs.values()))

    def test_all_missing_is_pending(self):
        original = Path.is_file
        outputs = {name + ".json" for name in S.SPECS}
        with mock.patch.object(Path, "is_file", lambda p: False if p.name in outputs else original(p)):
            result = S.run()
        self.assertEqual(result["status"], "PENDING")
        self.assertEqual(result["expected_results"], 160)
        self.assertTrue(all(e["status"] == "PENDING" for e in result["evidence"]))

    def test_256k_plan_exact_coverage_and_scope(self):
        plan, _ = S.F.read_record(S.HERE / "exp-stress-256k-plan.json")
        self.assertEqual(S.check_plan(plan), [])
        specs = S.plan_specs(plan)
        self.assertEqual(len(specs), 48)
        self.assertTrue(all(s["length"] == 262144 and s["sample_rows"] == 128 for s in specs.values()))
        self.assertEqual({s["distribution"] for s in specs.values()}, set(S.CONTEXT256K_DISTRIBUTIONS))
        changed = copy.deepcopy(plan)
        changed["cases"].pop()
        with self.assertRaises(ValueError):
            S.plan_specs(changed)
        changed = copy.deepcopy(plan)
        changed["cases"][0]["args"] = changed["cases"][0]["args"].replace("--sample-rows 128", "--sample-rows 1024")
        with self.assertRaises(ValueError):
            S.plan_specs(changed)
        changed = copy.deepcopy(plan)
        changed["cases"][0]["args"] = changed["cases"][0]["args"].replace(
            "--distribution normal", "--distribution common_v"
        )
        with self.assertRaises(ValueError):
            S.plan_specs(changed)


if __name__ == "__main__":
    unittest.main()
