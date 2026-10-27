# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch tests for opt-in grouped prefill; device correctness is separate."""

import unittest
from collections import Counter
from types import MethodType, SimpleNamespace

import torch

from models.demos.qwen38_27b_qb2.tests.unit.test_serving_prefill_trace import load_methods


class BatchedPrefillDispatchTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.ops = SimpleNamespace(uint32="u32", ROW_MAJOR_LAYOUT="rm")
        self.cache = SimpleNamespace(batch_size=4, capacity=16384)
        self.table = torch.zeros(4, 512, dtype=torch.int32)
        self.gen = SimpleNamespace(
            batched_prefill=True,
            prefill_prepared=None,
            cache=self.cache,
            prefill_signatures=set(),
            page_table=self.table,
            counters=Counter(),
            model=SimpleNamespace(
                config=SimpleNamespace(vocab_size=1000),
                upload=lambda x, **kw: x,
                prefill_batch=self.batch,
                prefill=lambda x, **kw: self.calls.append(("single", x.clone(), kw)) or "single",
            ),
            _release_traces=lambda **kw: None,
            _refresh_table=lambda table: None,
        )
        methods = load_methods("generator.py", "Qwen38Generator", ["prefill_forward"], self.ops)
        self.forward = MethodType(methods["prefill_forward"], self.gen)

    def batch(self, tokens, **kwargs):
        self.calls.append(("batch", tokens.clone(), kwargs))
        return ["first", "second"]

    def run_prefill(self, lengths, slots, starts=None):
        tokens = torch.stack([torch.full((max(lengths),), i + 1) for i in range(len(lengths))])
        return self.forward(
            tokens, page_table=self.table, kv_cache=self.cache, prompt_lens=lengths, slots=slots, start_pos=starts
        )

    def test_equal_lengths_group_without_mixing_rows(self):
        self.assertEqual(self.run_prefill([128, 128], [1, 2]), ["first", "second"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][2]["slots"], [1, 2])
        self.assertTrue(torch.all(self.calls[0][1][0] == 1))
        self.assertTrue(torch.all(self.calls[0][1][1] == 2))

    def test_chunks_preserve_prefix(self):
        self.run_prefill([4128, 4128], [0, 1], [32, 32])
        self.assertEqual([c[2]["length"] for c in self.calls], [4096, 32])
        self.assertEqual([c[2]["start_pos"] for c in self.calls], [32, 4128])

    def test_skip_only_intermediate_chunk_logits(self):
        self.gen.skip_intermediate_prefill_head = True
        self.run_prefill([4128, 4128], [0, 1], [32, 32])
        self.assertEqual([c[2].get("return_logits", True) for c in self.calls], [False, True])
        self.assertEqual([c[2]["start_pos"] for c in self.calls], [32, 4128])

    def test_ragged_reordered_unaligned_and_single_fall_back(self):
        for lengths, slots, starts in [
            ([64, 128], [0, 1], [0, 0]),
            ([64, 64], [2, 0], [0, 0]),
            ([64, 64], [0, 1], [1, 1]),
            ([64], [0], [0]),
        ]:
            with self.subTest(lengths=lengths, slots=slots, starts=starts):
                self.calls.clear()
                self.run_prefill(lengths, slots, starts)
                self.assertTrue(all(c[0] == "single" for c in self.calls))

    def test_default_path_stays_serial(self):
        self.gen.batched_prefill = False
        self.run_prefill([64, 64], [0, 1])
        self.assertEqual([c[0] for c in self.calls], ["single", "single"])

    def test_invalid_length_rejected_before_model(self):
        with self.assertRaises(ValueError):
            self.run_prefill([16400, 16400], [0, 1])
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
