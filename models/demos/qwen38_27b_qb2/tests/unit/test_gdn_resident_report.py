# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete/forged qualification receipts without device dependencies."""

import copy
import unittest

from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings
from models.demos.qwen38_27b_qb2.tests.gdn_resident import CASES, checkpoints, validate_report


def report():
    dense = dict(passed=True, finite=True, min_head_pcc=0.99999, max_head_relative_rms=0.0001, max_abs=0.0001)
    ranks = [
        dict(
            rank=i,
            state_bit_identical=True,
            output_bit_identical=True,
            state_dense=dense.copy(),
            output_dense=dense.copy(),
            state_sha256=str(i) * 64,
            output_sha256=str(i) * 64,
        )
        for i in range(4)
    ]
    cases = []
    for batch, cycles in CASES:
        case = dict(
            batch=batch,
            cycles=cycles,
            heads_per_rank=batch * 12,
            steps=cycles * 64,
            passed=True,
            input_and_address_stability=True,
            checks=[dict(phase=phase, steps=steps, ranks=copy.deepcopy(ranks)) for phase, steps in checkpoints(cycles)],
        )
        if batch != 1:
            timing = [
                dict(variant=variant, traced_call_us=[value] * 5)
                for variant, value in (("native", 100), ("fused", 90), ("native", 101))
            ]
            case.update(
                timings=timing,
                comparison=compare_timings(timing),
                identical_steps_per_arm=508,
                timing_final_states=[[str(i) * 64 for i in range(4)] for _ in range(3)],
                timing_final_outputs=[[str(i) * 64 for i in range(4)] for _ in range(3)],
            )
        cases.append(case)
    return dict(state="completed", passed=True, cleanup_completed=True, device_ids=[0, 1, 2, 3], cases=cases)


class ResidentReportTests(unittest.TestCase):
    def test_complete_receipt_is_kernel_evidence_not_model_promotion(self):
        result = validate_report(report())
        self.assertTrue(result["correctness_passed"])
        self.assertFalse(result["full_model_qualified"])
        self.assertFalse(result["promoted_to_serving"])

    def test_reject_missing_cleanup_and_duplicate_devices(self):
        for key, value in [
            ("state", "running"),
            ("passed", False),
            ("cleanup_completed", False),
            ("device_ids", [0, 1, 2, 2]),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                r = report()
                r[key] = value
                validate_report(r)

    def test_reject_omitted_long_horizon_or_intermediate_check(self):
        r = report()
        r["cases"].pop()
        with self.assertRaises(ValueError):
            validate_report(r)
        r = report()
        r["cases"][-1]["checks"].pop(3)
        with self.assertRaises(ValueError):
            validate_report(r)

    def test_missing_rank_and_wrong_output_are_failures(self):
        r = report()
        r["cases"][0]["checks"][0]["ranks"].pop()
        with self.assertRaises(ValueError):
            validate_report(r)
        r = report()
        r["cases"][0]["checks"][0]["ranks"][3]["output_bit_identical"] = False
        with self.assertRaises(ValueError):
            validate_report(r)

    def test_dense_accuracy_recomputed_even_if_passed_flag_true(self):
        for key, value in [
            ("min_head_pcc", 0.98),
            ("max_head_relative_rms", 0.02),
            ("max_abs", float("nan")),
            ("finite", False),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                r = report()
                r["cases"][-1]["checks"][-1]["ranks"][3]["state_dense"][key] = value
                validate_report(r)

    def test_reject_different_timing_trajectories(self):
        r = report()
        r["cases"][0]["timing_final_states"][1][3] = "f" * 64
        with self.assertRaises(ValueError):
            validate_report(r)
        r = report()
        r["cases"][0]["identical_steps_per_arm"] = 507
        with self.assertRaises(ValueError):
            validate_report(r)

    def test_invented_speedup_is_rejected(self):
        r = report()
        r["cases"][0]["comparison"]["qualified_speedup"] = 3
        with self.assertRaises(ValueError):
            validate_report(r)

    def test_drift_invalidates_performance_without_rewriting_correctness(self):
        r = report()
        case = r["cases"][0]
        case["timings"][2]["traced_call_us"] = [120] * 5
        case["comparison"] = compare_timings(case["timings"])
        result = validate_report(r)
        self.assertTrue(result["correctness_passed"])
        self.assertIsNone(result["comparisons"][0]["qualified_speedup"])


if __name__ == "__main__":
    unittest.main()
