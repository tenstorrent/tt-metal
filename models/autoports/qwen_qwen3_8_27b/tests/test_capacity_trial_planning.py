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

    def test_unknown_failure_cannot_be_erased_by_later_retry(self):
        failed = dict(
            shared="0",
            source_sha256={},
            runtime={},
            status="fail",
            stress_cycles=1,
            capacity_pool_ids=1368,
            close_mesh_device_returned=True,
            error="unexplained timeout",
        )
        retry = dict(failed, status="pass", stress_cycles=3)
        result = module.next_trials([failed, retry])
        self.assertEqual(set(result), {"blocked"})

    def test_completed_guard_failure_is_still_a_numeric_witness(self):
        base = dict(
            shared="0",
            source_sha256={},
            runtime={},
            status="fail",
            stress_cycles=1,
            close_mesh_device_returned=True,
            error="Transient headroom failed",
            error_type="AssertionError",
        )
        one = dict(
            base,
            capacity_pool_ids=1400,
            records=[dict(length=513, input_sha256="same", tokens=[[1]], logits=["a"])],
        )
        two = dict(
            base,
            capacity_pool_ids=1401,
            records=[dict(length=513, input_sha256="same", tokens=[[2]], logits=["a"])],
        )
        with self.assertRaisesRegex(AssertionError, "mismatch"):
            module.next_trials([one, two])

    def test_guard_failure_with_cleanup_error_is_unknown(self):
        row = dict(
            shared="0",
            source_sha256={},
            runtime={},
            status="fail",
            stress_cycles=1,
            capacity_pool_ids=1368,
            close_mesh_device_returned=True,
            error="Transient headroom failed",
            error_type="AssertionError",
            cleanup_error="failed release",
        )
        self.assertEqual(set(module.next_trials([row])), {"blocked"})

    def test_geometry_is_bound_into_identity(self):
        self.assertNotEqual(
            module.campaign_identity(dict(shared_mlp_readers=1)), module.campaign_identity(dict(shared_mlp_readers=2))
        )

    def test_physical_bytes_include_padding(self):
        self.assertEqual(module.pool_bytes(1368), 23646044160)
        self.assertEqual(module.pool_bytes(1556), 26895974400)


if __name__ == "__main__":
    unittest.main()
