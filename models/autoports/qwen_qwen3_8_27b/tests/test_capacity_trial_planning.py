# SPDX-License-Identifier: Apache-2.0
"""Pure decision guardrails, not simulated hardware or vLLM qualification."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("planner", Path(__file__).with_name("plan_capacity_trials.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PlanningTest(unittest.TestCase):
    def test_no_observations_only_proposes_reference_pair(self):
        rows = module.next_trials([])
        self.assertEqual(set(rows), {"0", "1"})
        self.assertTrue(all(row["status"] == "next_native_trial" and row["ids"] == 1368 for row in rows.values()))

    def test_unclosed_failure_blocks_both_arms(self):
        row = dict(
            shared="0",
            source_sha256={},
            runtime={},
            status="fail",
            stress_cycles=1,
            capacity_pool_ids=1368,
            close_mesh_device_returned=False,
            error="unknown",
        )
        result = module.next_trials([row])
        self.assertEqual(set(result), {"blocked"})
        self.assertTrue(result["blocked"]["no_device_trials_may_continue"])

    def test_physical_bytes_include_padding(self):
        self.assertEqual(module.pool_bytes(1368), 23646044160)
        self.assertEqual(module.pool_bytes(1556), 26895974400)


if __name__ == "__main__":
    unittest.main()
