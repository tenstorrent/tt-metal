# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch tests for opt-in grouped prefill; device correctness is separate."""

import unittest
from collections import Counter
from types import MethodType, SimpleNamespace

import torch

from models.demos.qwen38_27b_t3k.tests.unit.test_serving_prefill_trace import load_methods
from models.demos.qwen38_27b_t3k.tt.generator import batched_prefill_passes


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
            batched_prefill_shapes=set(),
            # Grouping only happens for a shape warmup already compiled, so these dispatch tests
            # stand in for a warmed server. The fallback has its own test below.
            _batched_prefill_compiling=True,
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
            _slot_prefix_len=[0] * self.cache.batch_size,
        )
        methods = load_methods(
            "generator.py", "Qwen38Generator", ["prefill_forward", "_begin_prefix", "_end_prefix"], self.ops
        )
        self.gen._begin_prefix = MethodType(methods["_begin_prefix"], self.gen)
        self.gen._end_prefix = MethodType(methods["_end_prefix"], self.gen)
        self.forward = MethodType(methods["prefill_forward"], self.gen)

    def batch(self, tokens, **kwargs):
        self.calls.append(("batch", tokens.clone(), kwargs))
        return ["first", "second"]

    def run_prefill(self, lengths, slots, starts=None):
        tokens = torch.stack([torch.full((max(lengths),), i + 1) for i in range(len(lengths))])
        # A nonzero start continues a request that already advanced this slot.
        for start, slot in zip(starts or [0] * len(slots), slots):
            if 0 <= slot < len(self.gen._slot_prefix_len):
                self.gen._slot_prefix_len[slot] = start
        return self.forward(
            tokens, page_table=self.table, kv_cache=self.cache, prompt_lens=lengths, slots=slots, start_pos=starts
        )

    def test_an_uncompiled_group_is_served_row_by_row(self):
        # Compiling a grouped prefill mid-request costs 8-15 s of TTFT. The per-row loop reaches
        # the same state through programs keyed on the length alone, which startup has built.
        self.gen._batched_prefill_compiling = False
        self.run_prefill([128, 128], [1, 2])
        self.assertEqual([call[0] for call in self.calls], ["single", "single"])
        self.assertEqual(self.gen.counters["batched_prefill_calls"], 0)

    def test_a_group_whose_shape_is_already_compiled_is_grouped(self):
        self.gen._batched_prefill_compiling = False
        self.gen.batched_prefill_shapes.add((2, 0, 128))
        self.run_prefill([128, 128], [1, 2])
        self.assertEqual([call[0] for call in self.calls], ["batch"])

    def test_compiling_a_group_records_its_shape_for_later(self):
        self.run_prefill([128, 128], [1, 2])
        self.assertIn((2, 0, 128), self.gen.batched_prefill_shapes)

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


class PrefillSignatureCoverageTests(unittest.TestCase):
    """Every chunk the per-row loop compiles has to appear in the signature set.

    Each chunk builds its own program and uploads its own selectors. Those are persistent device
    buffers, and the allocator refuses to place them safely beside a live trace, so a chunk the
    set does not name lands on addresses a decode trace already reserved.
    """

    def setUp(self):
        self.releases = []
        self.ops = SimpleNamespace(uint32="u32", ROW_MAJOR_LAYOUT="rm")
        self.cache = SimpleNamespace(batch_size=4, capacity=16384)
        self.table = torch.zeros(4, 512, dtype=torch.int32)
        self.gen = SimpleNamespace(
            batched_prefill=False,
            prefill_prepared=None,
            cache=self.cache,
            prefill_signatures=set(),
            batched_prefill_shapes=set(),
            page_table=self.table,
            counters=Counter(),
            model=SimpleNamespace(
                config=SimpleNamespace(vocab_size=1000),
                upload=lambda x, **kw: x,
                prefill=lambda x, **kw: "single",
            ),
            _release_traces=lambda **kw: self.releases.append(kw),
            _refresh_table=lambda table: None,
            _bucket_selectors=lambda bucket: SimpleNamespace(update=lambda count: None),
            _slot_prefix_len=[0] * self.cache.batch_size,
        )
        methods = load_methods(
            "generator.py", "Qwen38Generator", ["prefill_forward", "_begin_prefix", "_end_prefix"], self.ops
        )
        self.gen._begin_prefix = MethodType(methods["_begin_prefix"], self.gen)
        self.gen._end_prefix = MethodType(methods["_end_prefix"], self.gen)
        self.forward = MethodType(methods["prefill_forward"], self.gen)

    def run_prefill(self, length):
        tokens = torch.ones(1, length, dtype=torch.int64)
        self.gen._slot_prefix_len[0] = 0
        return self.forward(
            tokens, page_table=self.table, kv_cache=self.cache, prompt_lens=[length], slots=[0], start_pos=[0]
        )

    def test_a_tail_chunk_whose_shape_is_new_releases_before_it_compiles(self):
        # The head chunk of 4128 is the same 4096-token program a 4096 prompt already compiled,
        # so a set naming only the head would call this prompt entirely known.
        self.run_prefill(4096)
        self.releases.clear()
        self.run_prefill(4128)
        self.assertEqual(len(self.releases), 1)

    def test_a_prompt_whose_every_chunk_is_known_keeps_its_traces(self):
        self.run_prefill(4128)
        self.releases.clear()
        self.run_prefill(4128)
        self.assertEqual(self.releases, [])


class PassWidthTests(unittest.TestCase):
    """A grouped pass is capped by the activations it allocates, not by the server's occupancy."""

    def test_a_pass_inside_the_budget_is_not_split(self):
        self.assertEqual(batched_prefill_passes(8, 4096, 32768), [8])

    def test_the_serving_shape_splits_into_equal_halves(self):
        # 16 x 4096 allocates 640 MiB per projection, needing 53.3 MiB contiguous per bank
        # against a largest free block of 50.8 MiB. Two passes of 8 ask 26.7 MiB.
        self.assertEqual(batched_prefill_passes(16, 4096, 32768), [8, 8])

    def test_short_prompts_keep_the_full_occupancy(self):
        # The budget is a token count, so a quarter-length chunk carries four times the rows.
        self.assertEqual(batched_prefill_passes(16, 1024, 32768), [16])

    def test_widths_are_balanced_rather_than_greedy(self):
        # A greedy split would be [8, 1], and prefill_batch refuses a lone row.
        self.assertEqual(batched_prefill_passes(9, 4096, 32768), [5, 4])

    def test_no_width_is_ever_a_single_row(self):
        for rows in range(2, 33):
            for budget in (0, 2048, 4096, 32768, 65536):
                with self.subTest(rows=rows, budget=budget):
                    widths = batched_prefill_passes(rows, 4096, budget)
                    self.assertEqual(sum(widths), rows)
                    self.assertTrue(all(width >= 2 for width in widths))

    def test_a_zero_budget_removes_the_cap(self):
        self.assertEqual(batched_prefill_passes(16, 4096, 0), [16])


class PassSplitDispatchTests(unittest.TestCase):
    """The split has to reach prefill_batch as whole rows, in slot order, state untouched."""

    def setUp(self):
        self.calls = []
        self.releases = []
        self.ops = SimpleNamespace(uint32="u32", ROW_MAJOR_LAYOUT="rm")
        self.cache = SimpleNamespace(batch_size=16, capacity=16384)
        self.table = torch.zeros(16, 512, dtype=torch.int32)
        self.gen = SimpleNamespace(
            batched_prefill=True,
            prefill_prepared=None,
            cache=self.cache,
            prefill_signatures=set(),
            batched_prefill_shapes=set(),
            _batched_prefill_compiling=True,
            page_table=self.table,
            counters=Counter(),
            # One row of 4096 is half the budget, so four rows split into two passes of two.
            prefill_pass_tokens=8192,
            model=SimpleNamespace(
                config=SimpleNamespace(vocab_size=1000),
                upload=lambda x, **kw: x,
                prefill_batch=self.batch,
                prefill=lambda x, **kw: self.calls.append(("single", x.clone(), kw)) or "single",
            ),
            _release_traces=self.release,
            _refresh_table=lambda table: None,
            _slot_prefix_len=[0] * self.cache.batch_size,
        )
        methods = load_methods(
            "generator.py", "Qwen38Generator", ["prefill_forward", "_begin_prefix", "_end_prefix"], self.ops
        )
        self.gen._begin_prefix = MethodType(methods["_begin_prefix"], self.gen)
        self.gen._end_prefix = MethodType(methods["_end_prefix"], self.gen)
        self.forward = MethodType(methods["prefill_forward"], self.gen)

    def release(self, **kwargs):
        self.releases.append(kwargs)

    def batch(self, tokens, **kwargs):
        self.calls.append(("batch", tokens.clone(), kwargs))
        return [f"slot{slot}" for slot in kwargs["slots"]]

    def run_prefill(self, lengths, slots):
        tokens = torch.stack([torch.full((max(lengths),), i + 1) for i in range(len(lengths))])
        return self.forward(tokens, page_table=self.table, kv_cache=self.cache, prompt_lens=lengths, slots=slots)

    def test_rows_split_into_passes_over_consecutive_slots(self):
        self.run_prefill([4096] * 4, [0, 1, 2, 3])
        self.assertEqual([call[2]["slots"] for call in self.calls], [[0, 1], [2, 3]])

    def test_each_pass_carries_only_its_own_token_rows(self):
        self.run_prefill([4096] * 4, [0, 1, 2, 3])
        first, second = (call[1] for call in self.calls)
        self.assertTrue(torch.all(first[0] == 1) and torch.all(first[1] == 2))
        self.assertTrue(torch.all(second[0] == 3) and torch.all(second[1] == 4))

    def test_results_come_back_in_slot_order_across_passes(self):
        self.assertEqual(self.run_prefill([4096] * 4, [0, 1, 2, 3]), ["slot0", "slot1", "slot2", "slot3"])

    def test_a_split_group_still_records_the_shape_the_caller_asked_for(self):
        # The index is keyed on what the server requests; a later identical request must take
        # the grouped path rather than recompile, whatever width the pass ran at.
        self.run_prefill([4096] * 4, [0, 1, 2, 3])
        self.assertIn((4, 0, 4096), self.gen.batched_prefill_shapes)

    def test_every_chunk_of_every_pass_keeps_its_prefix(self):
        self.run_prefill([8192] * 4, [0, 1, 2, 3])
        self.assertEqual([c[2]["start_pos"] for c in self.calls], [0, 4096, 0, 4096])
        self.assertEqual([c[2]["slots"] for c in self.calls], [[0, 1], [0, 1], [2, 3], [2, 3]])

    def test_an_uncapped_generator_runs_one_pass(self):
        self.gen.prefill_pass_tokens = 0
        self.run_prefill([4096] * 4, [0, 1, 2, 3])
        self.assertEqual([call[2]["slots"] for call in self.calls], [[0, 1, 2, 3]])

    def test_a_changed_budget_releases_traces_before_it_compiles_new_widths(self):
        # The budget decides the pass widths, so the same caller shape at a new budget is a new
        # set of programs. Those own persistent buffers; compiling them while a trace still holds
        # the addresses puts them on reserved memory.
        self.run_prefill([4096] * 4, [0, 1, 2, 3])
        self.releases.clear()
        self.calls.clear()
        self.gen.prefill_pass_tokens = 16384
        self.run_prefill([4096] * 4, [0, 1, 2, 3])
        self.assertTrue(self.releases, "a new pass budget compiled new widths beside a live trace")
        self.assertEqual([call[2]["slots"] for call in self.calls], [[0, 1, 2, 3]])
