# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for the pinned worker's model budget contract."""

import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "tt" / "kv_pool_budget.py"
GENERATOR_SOURCE = Path(__file__).resolve().parents[1] / "tt" / "generator_vllm.py"
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

    def test_explicit_pool_override_cancels_worker_reserves_for_supported_batches(self):
        expected_minimums = {1: 8449, 8: 10248, 16: 12304, 32: 16416}
        for batch, minimum in expected_minimums.items():
            with self.subTest(batch=batch):
                ids = minimum + 37
                info = module.get_pool_budget_info(max_model_len=262144, max_num_seqs=batch, pool_ids_override=str(ids))
                reserves = 32 * batch + 8 * 1024 * batch
                self.assertEqual(info["synthetic_token_budget"], ids * 32 - reserves)
                self.assertEqual(info["pool_ids_override"], ids)
                self.assertEqual(info["pool_override_source"], "GEMMA4_KV_POOL_IDS")
                self.assertEqual(info["pool_total_tokens_including_worker_reserves"], ids * 32)
                self.assertEqual(info["pool_minimum_ids_for_context"], minimum)
                self.assertEqual(info["worker_output_reserve_tokens"], 32 * batch)
                self.assertEqual(info["worker_sliding_reserve_tokens"], 8 * 1024 * batch)
                reconstructed = info["synthetic_token_budget"] + reserves
                self.assertEqual((reconstructed + 31) // 32, ids)

    def test_override_rejects_invalid_values_batches_context_and_subminimum_pools(self):
        for value in ("", "0", "01", "+10000", "10000.0", " 12000", "-1", "１２３", 12000):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "canonical positive decimal"):
                module.get_pool_token_budget(max_model_len=262144, max_num_seqs=8, pool_ids_override=value)
        for batch, minimum in {1: 8449, 8: 10248, 16: 12304, 32: 16416}.items():
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, "context-preserving minimum"):
                module.get_pool_token_budget(
                    max_model_len=262144, max_num_seqs=batch, pool_ids_override=str(minimum - 1)
                )
        with self.assertRaisesRegex(ValueError, "advertised 262144-token context"):
            module.get_pool_token_budget(max_model_len=131072, max_num_seqs=8, pool_ids_override="12000")
        with self.assertRaisesRegex(ValueError, "supports batches"):
            module.get_pool_token_budget(max_model_len=262144, max_num_seqs=4, pool_ids_override="12000")

    def test_new_and_legacy_overrides_conflict_but_legacy_behavior_is_preserved(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            module.get_pool_token_budget(
                max_model_len=262144,
                max_num_seqs=8,
                b8_pool_ids="20480",
                pool_ids_override="20480",
            )
        info = module.get_pool_budget_info(max_model_len=262144, max_num_seqs=8, b8_pool_ids="20480")
        self.assertEqual(info["synthetic_token_budget"], 589568)
        self.assertEqual(info["pool_ids_override"], 20480)
        self.assertEqual(info["pool_override_source"], "GEMMA4_B8_KV_POOL_IDS")
        self.assertEqual(info["b8_pool_ids_override"], 20480)
        for batch in (1, 16, 32):
            self.assertIsNone(
                module.get_pool_budget_info(max_model_len=262144, max_num_seqs=batch, b8_pool_ids="20480")[
                    "pool_override_source"
                ]
            )

    def test_serving_allocator_fit_guard_is_unchanged(self):
        source = GENERATOR_SOURCE.read_text()
        self.assertIn("memory_plan = validate_pool_memory(", source)
        self.assertIn('if not memory_plan["fits"]:', source)
        self.assertIn(
            'raise RuntimeError("Fixed serving KV pool exceeds launch free memory after recorded reserves")',
            source,
        )

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
