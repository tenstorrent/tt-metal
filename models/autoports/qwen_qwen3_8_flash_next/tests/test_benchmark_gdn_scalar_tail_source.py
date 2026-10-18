# SPDX-License-Identifier: Apache-2.0
"""Host-only exact scalar-cache/tail composition and private-binding guards."""
import unittest
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_scalar_tail_candidate as candidate


class ScalarTailTests(unittest.TestCase):
    def test_exact_qualified_component_composition(self):
        self.assertEqual(candidate.READER, candidate.scalar.READER)
        self.assertEqual(candidate.COMPUTE, candidate.tail.COMPUTE)
        self.assertEqual(candidate.LAUNCHER_SOURCE, candidate.scalar.LAUNCHER_SOURCE)
        self.assertIs(candidate.gdn_method, candidate.resident.gdn_method)
        self.assertEqual(candidate.BODY_PATH, candidate.tail.BODY_PATH)

    def test_private_namespaces_and_preserved_controls(self):
        for module in (candidate.tail, candidate.scalar, candidate.resident):
            self.assertIsNot(candidate.RECURRENCE.__globals__, module.RECURRENCE.__globals__)
            self.assertEqual(module.RECURRENCE.__globals__["COMPUTE"], module.COMPUTE)
            self.assertEqual(module.RECURRENCE.__globals__["READER"], module.READER)
        self.assertIs(candidate.recurrence("current_rm"), candidate.baseline._rm_recurrence)
        self.assertIs(candidate.recurrence("resident_control"), candidate.resident.RECURRENCE)
        self.assertIs(candidate.recurrence("scalar_control"), candidate.scalar.RECURRENCE)
        self.assertIs(candidate.recurrence("combined_nan"), candidate.RECURRENCE)

    def test_probe_retains_exactness_poison_and_scalar_control(self):
        text = Path(__file__).with_name("benchmark_gdn_scalar_tail_probe.py").read_text()
        self.assertIn(
            'VARIANTS = ("current_rm", "resident_control", "scalar_control", "combined_zero", "combined_nan")', text
        )
        self.assertIn('0x7FC12345 if name == "combined_nan" else 0', text)
        self.assertIn('default="128,256,512"', text)
        self.assertIn('require_exact(group["output"])', text)
        self.assertIn("relative_to_scalar_speedup", text)
        self.assertIn("trace_released=True", text)
        self.assertIn("decay_beta_each_bytes_per_refresh=4096", text)


if __name__ == "__main__":
    unittest.main()
