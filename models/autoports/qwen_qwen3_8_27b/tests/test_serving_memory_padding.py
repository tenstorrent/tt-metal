# SPDX-License-Identifier: Apache-2.0
"""BH state allocation bank-padding regressions; no TTNN/vLLM imports."""
import importlib.util
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serving_memory_plan", root / "tests/plan_serving_memory.py")
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


class StatePaddingTest(unittest.TestCase):
    def setUp(self):
        self.constants = planner.constants(root / "tt/vllm_cache_specs.py")

    def test_2048_checkpoint_pool_has_separate_conv_bank_padding(self):
        state = planner.state_backing(self.constants, 771, 16)
        self.assertEqual(state, dict(payload_bytes=9890906112, physical_bytes=9891479552, bank_padding_bytes=573440))

    def test_aligned_measured_512_pool_and_single_active_row(self):
        aligned = planner.state_backing(self.constants, 1368, 16)
        self.assertEqual(aligned["physical_bytes"], 17549623296)
        self.assertEqual(aligned["bank_padding_bytes"], 0)
        active = planner.state_backing(self.constants, 1, 48)
        self.assertEqual(active["physical_bytes"], 39714816)
        self.assertEqual(active["bank_padding_bytes"], 1228800)

    def test_historical_1571_id_pool_has_same_seven_conv_pages_per_pool(self):
        state = planner.state_backing(self.constants, 1571, 16)
        self.assertEqual(state["bank_padding_bytes"], 573440)

    def test_payload_hook_does_not_hide_physical_budget_overrun(self):
        row = planner.forecast(self.constants, 30654464, 32, 2048)
        self.assertEqual(row["global_ids"], 1)
        self.assertEqual(row["physical_cache_over_budget_bytes"], 409600)


if __name__ == "__main__":
    unittest.main()
