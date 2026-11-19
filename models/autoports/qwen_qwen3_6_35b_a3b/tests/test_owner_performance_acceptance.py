# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression checks for the scoped stage-5 owner decision.

Run with python -m unittest; no TTNN import or hardware fixture is needed.
"""

import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.check_optimized_multichip_evidence import DOC, OptimizedGate
from models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_multichip_validation import Gate


class AlteredEvidence(OptimizedGate):
    def __init__(self, mutate):
        super().__init__(DOC)
        self.mutate = mutate

    def read(self, name):
        data = super().read(name)
        self.mutate(Path(name).name, data)
        return data


class OwnerPerformanceAcceptanceTests(unittest.TestCase):
    def test_recorded_results_keep_raw_outcomes(self):
        linear = OptimizedGate(DOC).functional(0)
        full = OptimizedGate(DOC).functional(3)
        self.assertFalse(linear["raw_floor_passed"])
        self.assertEqual(linear["disposition"], "owner-accepted-performance-limitation")
        self.assertTrue(full["raw_floor_passed"])
        self.assertEqual(full["disposition"], "measured-pass")

    def test_stage4_still_rejects_failed_result(self):
        with self.assertRaisesRegex(ValueError, "result not passed"):
            Gate(DOC).paired("restored_functional_l0.json", 0, 1)

    def test_missing_or_changed_decision_rejected(self):
        def missing(name, data):
            if name == "limitations.json":
                data["limitations"] = []

        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            AlteredEvidence(missing).functional(0)
        for field, value in [
            ("stage", "full-model"),
            ("category", "accuracy"),
            ("status", "proposed"),
            ("batch", 32),
            ("mesh", [1, 1]),
            ("raw_floor_passed", True),
            ("measured_decode_ms_per_token", 0.8),
            ("decoder_sha256", "different"),
        ]:
            with self.subTest(field=field):

                def change(name, data):
                    if name == "limitations.json":
                        data["limitations"][0][field] = value

                with self.assertRaises(ValueError):
                    AlteredEvidence(change).functional(0)

    def test_quality_and_trace_failures_still_rejected(self):
        changes = [
            lambda d: d.update(prefill_pcc=0.98),
            lambda d: d["decode_pcc"].__setitem__(0, 0.98),
            lambda d: d["state_checks"][0].update(pcc=0.98),
            lambda d: d["multichip"].update(runtime_fallback_audit="fail"),
            lambda d: d["multichip"].update(all_step_eager_trace_equal=False),
            lambda d: d["multichip"].update(output_replicas_equal=False),
            lambda d: d["multichip"]["functional_sequence"].update(repeated_bit_equal=False),
            lambda d: d.update(passed=True),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):

                def mutate(name, data):
                    if name == "restored_functional_l0.json":
                        change(data)

                with self.assertRaises(ValueError):
                    AlteredEvidence(mutate).functional(0)

    def test_full_attention_has_no_exception(self):
        def mutate(name, data):
            if name == "restored_functional_l3.json":
                data["passed"] = False

        with self.assertRaisesRegex(ValueError, "result not passed"):
            AlteredEvidence(mutate).functional(3)

    def test_different_native_build_rejected(self):
        def mutate(name, data):
            if name == "restored_functional_l0.json.provenance.json":
                next(iter(data["loaded_native_binaries"].values()))["sha256"] = "different"

        with self.assertRaisesRegex(ValueError, "different native build"):
            AlteredEvidence(mutate).functional(0)


if __name__ == "__main__":
    unittest.main()
