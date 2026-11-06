# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for the pinned worker's model budget contract."""

import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tt" / "kv_pool_budget.py"
spec = importlib.util.spec_from_file_location("gemma4_kv_pool_budget_host_tests", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class KVPoolBudgetTests(unittest.TestCase):
    def test_default_supported_batches_preserve_previous_budgets(self):
        for batch, expected in [(1, 262144), (8, 262144), (16, 458752), (32, 917504)]:
            with self.subTest(batch=batch):
                self.assertEqual(module.get_pool_token_budget(max_model_len=262144, max_num_seqs=batch), expected)

    def test_default_retains_larger_context_floor(self):
        self.assertEqual(module.get_pool_token_budget(max_model_len=1048576, max_num_seqs=8), 1048576)

    def test_expanded_budget_yields_exact_normalized_pool_with_worker_allowances(self):
        budget = module.get_pool_token_budget(max_model_len=262144, max_num_seqs=8, b8_pool_ids="20480")
        self.assertEqual(budget, 589568)
        # Independent pinned worker contract: output reserve then eight sliding
        # groups' heuristic allowance; the one-request admission floor is lower.
        worker_tokens = budget + 8 * 32 + 8 * 8 * 1024
        self.assertEqual((worker_tokens + 31) // 32, 20480)
        self.assertGreaterEqual(20480, 18451 + 2000)

    def test_b8_option_does_not_expand_other_batches(self):
        for batch, expected in [(1, 262144), (16, 458752), (32, 917504)]:
            self.assertEqual(
                module.get_pool_token_budget(max_model_len=262144, max_num_seqs=batch, b8_pool_ids="20480"),
                expected,
            )

    def test_rejects_unmeasured_or_malformed_pool_values(self):
        for value in ["", "0", "10248", "18451", "20481", "36896", "-1", "20480.0", " 20480", 20480]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "supports only 20480"):
                module.get_pool_token_budget(max_model_len=262144, max_num_seqs=8, b8_pool_ids=value)

    def test_expanded_pool_requires_advertised_context(self):
        for context in [131072, 262143, 262145, 1048576]:
            with self.subTest(context=context), self.assertRaisesRegex(ValueError, "advertised"):
                module.get_pool_token_budget(max_model_len=context, max_num_seqs=8, b8_pool_ids="20480")


if __name__ == "__main__":
    unittest.main()
