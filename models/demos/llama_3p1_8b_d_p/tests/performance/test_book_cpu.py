# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standard-library checks for the final native row and exact fixture binding."""

import ast
import copy
import math
import unittest
from pathlib import Path

from models.demos.llama_3p1_8b_d_p.tests.performance import book_observation as book


class BookTests(unittest.TestCase):
    # Vocabulary slices must be concatenated in TP order from the last real row only.
    def test_final_row_and_tp_order(self):
        self.assertEqual(book.final_position(4096), dict(position=4095, chunk_start=3072, sp=3, local_row=255))
        records = [
            dict(tp=tp, sp=3, local_row=255, position=4095, values=[tp * 2, tp * 2 + 1]) for tp in reversed(range(8))
        ]
        self.assertEqual(book.assemble_final_logits(records, 4096, shard_width=2), list(range(16)))
        for field, value in (("sp", 2), ("local_row", 254), ("position", 4094), ("tp", 1)):
            bad = copy.deepcopy(records)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                book.assemble_final_logits(bad, 4096, shard_width=2)
        with self.assertRaises(ValueError):
            book.assemble_final_logits(records[:-1], 4096, shard_width=2)
        with self.assertRaises(ValueError):
            book.final_position(4095)

    # Nonfinite or truncated vocabulary slices cannot produce a plausible top-five report.
    def test_finite_and_width_checks(self):
        records = [dict(tp=tp, sp=3, local_row=255, position=4095, values=[float(tp)] * 2) for tp in range(8)]
        for values in ([1], [1, math.nan], [1, math.inf]):
            bad = copy.deepcopy(records)
            bad[0]["values"] = values
            with self.assertRaises(ValueError):
                book.assemble_final_logits(bad, 4096, shard_width=2)

    # Expected continuation is observational: a low rank must be recorded without a success threshold.
    def test_ranking_probability_and_no_semantic_gate(self):
        row = book.rank_logits([0.0, 2.0, 2.0, -1.0, 1.0, -4.0], 5, lambda i: f"piece{i}")
        self.assertEqual([r["token_id"] for r in row["top5"]], [1, 2, 4, 0, 3])
        self.assertEqual(row["expected_next_token_rank"], 6)
        self.assertGreater(row["expected_next_token_probability"], 0)
        self.assertNotIn("passed", row)
        expected = math.exp(2) / sum(math.exp(x) for x in (0, 2, 2, -1, 1, -4))
        self.assertAlmostEqual(row["top5"][0]["probability"], expected)

    # Each slot must bind its own complete prompt and its final position, not just a matching length label.
    def test_prompt_binding(self):
        prompts = book.synthetic_prompts(4096)
        book.validate_prompts(prompts, 4096)
        for change in ("length", "slot", "same", "position", "bos", "expected", "token"):
            bad = copy.deepcopy(prompts)
            if change == "length":
                bad[0]["token_ids"].pop()
            if change == "slot":
                bad[1]["slot"] = 0
            if change == "same":
                bad[1]["token_ids"] = bad[0]["token_ids"][:]
            if change == "position":
                bad[0]["metadata"]["final_prompt_position"] = 4094
            if change == "bos":
                bad[0]["token_ids"][1] = 128000
            if change == "token":
                bad[0]["token_ids"][1] = 22
            if change == "expected":
                bad[0]["metadata"]["expected_next_token_id"] = 128256
            with self.assertRaises(ValueError):
                book.validate_prompts(bad, 4096)

    # The device entry uses the existing post-timer callback; no extra native pass or golden reference is added.
    def test_source_keeps_observation_in_existing_readback(self):
        tree = ast.parse((Path(__file__).parent / "test_long_context_performance.py").read_text())
        source = ast.unparse(tree)
        self.assertNotIn("chat_tokens", source)
        self.assertNotIn("reference_prefill", source)
        evidence = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "logits_evidence")
        calls = [ast.unparse(n.func) for n in ast.walk(evidence) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count("ttnn.to_torch"), 1)
        self.assertIn("assemble_final_logits", calls)
        self.assertIn("rank_logits", calls)
        self.assertNotIn("model.prefill_chunk", calls)


if __name__ == "__main__":
    unittest.main()
