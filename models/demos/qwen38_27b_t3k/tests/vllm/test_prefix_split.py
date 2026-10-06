# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stopping a prefill on a block boundary so its recurrent state can be kept.

The 48 GDN layers summarise one slot's tokens in order, so the state at token 992 exists only
while the prefill is at 992. A caller that asks afterwards gets the state at the end of the
prompt, which no content hash stands for. The boundary therefore has to be a stopping point: the
prompt runs in two passes, and what the first one leaves behind is the snapshot.

Device effects are replaced; what is under test is which ranges run and what is kept.
"""

import unittest
from collections import Counter
from types import SimpleNamespace

import torch

from models.demos.qwen38_27b_t3k.tt.generator_vllm import Qwen38ForCausalLM

BLOCK = 32


class RecordingGenerator:
    """Records the prefill ranges asked for and hands out a fresh handle per save."""

    def __init__(self, batch_size=4):
        self.cache = SimpleNamespace(batch_size=batch_size, capacity=4096, num_pages=64)
        self.page_host = torch.zeros(batch_size, 4, dtype=torch.int32)
        self.page_table = torch.zeros_like(self.page_host)
        self.counters = Counter()
        self.prefills = []
        self.saved_slots = []
        self.next_handle = 100
        self.store_full = False

    def prefill_forward(self, tokens, *, page_table, kv_cache, prompt_lens, start_pos, slots):
        self.prefills.append((start_pos[0], start_pos[0] + prompt_lens[0], slots[0], tokens.shape[1]))
        return [object()]

    def save_slot_state(self, slot):
        self.saved_slots.append(slot)
        if self.store_full:
            return None
        self.next_handle += 1
        return self.next_handle


class BoundarySplitTests(unittest.TestCase):
    def setUp(self):
        self.gen = RecordingGenerator()
        self.adapter = Qwen38ForCausalLM(self.gen, 4, 4096)
        self.adapter.cache = self.gen.cache
        self.tokens = torch.arange(4 * 2048, dtype=torch.int32).reshape(4, 2048)

    def split(self, starts, ends, slots, rows, block=BLOCK):
        return self.adapter._snapshot_at_boundary(
            self.tokens, self.gen.page_table, self.gen.cache, starts, ends, slots, rows, block
        )

    def test_a_prompt_is_run_to_the_boundary_below_its_end_and_kept_there(self):
        starts = self.split([0], [1000], [2], [0])
        self.assertEqual(self.gen.prefills, [(0, 992, 2, 992)])
        self.assertEqual(self.gen.saved_slots, [2])
        # The sampling pass that follows continues from the snapshot, not from zero.
        self.assertEqual(starts, [992])
        self.assertEqual(self.adapter.take_prefix_snapshots(), {0: (992, 101)})

    def test_an_aligned_prompt_stops_one_block_back_so_the_next_pass_has_tokens(self):
        starts = self.split([0], [64], [0], [0])
        self.assertEqual(self.gen.prefills, [(0, 32, 0, 32)])
        self.assertEqual(starts, [32])

    def test_a_prompt_inside_one_block_is_left_alone(self):
        for end in (1, BLOCK, BLOCK - 1):
            with self.subTest(end=end):
                self.gen.prefills.clear()
                self.assertEqual(self.split([0], [end], [0], [0]), [0])
                self.assertEqual(self.gen.prefills, [])
                self.assertEqual(self.adapter.take_prefix_snapshots(), {})

    def test_a_continuation_with_no_whole_block_beyond_it_is_left_alone(self):
        # The slot already holds 992, which is the boundary below 1000.
        self.assertEqual(self.split([992], [1000], [1], [0]), [992])
        self.assertEqual(self.gen.prefills, [])

    def test_a_continuation_runs_only_the_part_the_slot_does_not_hold(self):
        self.assertEqual(self.split([992], [1100], [1], [0]), [1088])
        self.assertEqual(self.gen.prefills, [(992, 1088, 1, 96)])

    def test_a_full_store_still_leaves_the_prefill_able_to_finish(self):
        # Losing a snapshot costs a cache hit later; leaving the slot mid-prompt would corrupt
        # this request, so the start still advances to where the state actually is.
        self.gen.store_full = True
        self.assertEqual(self.split([0], [1000], [0], [0]), [992])
        self.assertEqual(self.adapter.take_prefix_snapshots(), {})

    def test_only_the_named_rows_are_split(self):
        starts = self.split([0, 0, 0], [1000, 500, 300], [0, 1, 2], [0, 2])
        self.assertEqual(starts, [992, 0, 288])
        self.assertEqual([p[2] for p in self.gen.prefills], [0, 2])
        self.assertEqual(self.adapter.take_prefix_snapshots(), {0: (992, 101), 2: (288, 102)})

    def test_snapshots_are_cleared_on_read_so_one_step_cannot_claim_another_s(self):
        self.split([0], [1000], [0], [0])
        self.assertEqual(len(self.adapter.take_prefix_snapshots()), 1)
        self.assertEqual(self.adapter.take_prefix_snapshots(), {})

    def test_a_larger_page_size_moves_the_boundary_with_it(self):
        self.assertEqual(self.split([0], [1000], [0], [0], block=64), [960])
        self.assertEqual(self.gen.prefills, [(0, 960, 0, 960)])


if __name__ == "__main__":
    unittest.main()
