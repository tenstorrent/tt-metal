# SPDX-License-Identifier: Apache-2.0
"""Host-only composition guards: no additional arithmetic or launch change."""
import unittest

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_resident_combined_candidate as candidate


class CombinedTests(unittest.TestCase):
    def test_exact_qualified_component_composition(self):
        self.assertEqual(candidate.READER, candidate.narrow.READER)
        self.assertEqual(candidate.COMPUTE, candidate.tail.COMPUTE)
        self.assertEqual(candidate.LAUNCHER_SOURCE, candidate.resident.LAUNCHER_SOURCE)
        self.assertIs(candidate.gdn_method, candidate.resident.gdn_method)
        self.assertEqual(candidate.BODY_PATH, candidate.tail.BODY_PATH)

    def test_private_namespaces_and_preserved_controls(self):
        for module in (candidate.tail, candidate.narrow, candidate.resident):
            self.assertIsNot(candidate.RECURRENCE.__globals__, module.RECURRENCE.__globals__)
            self.assertEqual(module.RECURRENCE.__globals__["COMPUTE"], module.COMPUTE)
            self.assertEqual(module.RECURRENCE.__globals__["READER"], module.READER)
        self.assertIs(candidate.recurrence("current_rm"), candidate.baseline._rm_recurrence)
        self.assertIs(candidate.recurrence("resident_control"), candidate.resident.RECURRENCE)
        self.assertIs(candidate.recurrence("combined_nan"), candidate.RECURRENCE)

    def test_probe_keeps_exactness_poison_and_resident_controls(self):
        text = candidate.Path(__file__).with_name("benchmark_gdn_resident_combined_probe.py").read_text()
        self.assertIn('VARIANTS = ("current_rm", "resident_control", "combined_zero", "combined_nan")', text)
        self.assertIn('0x7FC12345 if name == "combined_nan" else 0', text)
        self.assertIn('default="128,256,512"', text)
        self.assertIn('require_exact(group["output"])', text)
        self.assertIn("relative_to_resident_speedup", text)
        self.assertIn("trace_released=True", text)


if __name__ == "__main__":
    unittest.main()
