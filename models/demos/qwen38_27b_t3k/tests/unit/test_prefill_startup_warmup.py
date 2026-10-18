# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Startup compilation must not leave synthetic request state in the server."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from models.demos.qwen38_27b_t3k.tt.generator_vllm import Qwen38ForCausalLM


class StartupWarmupTests(unittest.TestCase):
    def make_adapter(self, batch=4):
        adapter = object.__new__(Qwen38ForCausalLM)
        adapter.batch_size = batch
        adapter.context = 128
        adapter.prefill_startup_warmup = True
        adapter.cache = SimpleNamespace(num_pages=64, capacity=128)
        adapter.prefill_warmup_buckets = [32, 64, 96, 128]
        # Grouped prefill falls back to the per-row loop rather than compiling, so the width
        # ladder is throughput tuning and ships empty.
        adapter.prefill_warmup_widths = []
        adapter.generator = MagicMock()
        adapter.generator.cache = adapter.cache
        adapter.generator.batched_prefill = True
        adapter.generator.page_host = torch.zeros(batch, 4, dtype=torch.int32)
        adapter._decode_bound = True
        return adapter

    def test_disabled_warmup_touches_nothing(self):
        adapter = self.make_adapter()
        adapter.prefill_startup_warmup = False
        adapter.warmup_model_prefill(kv_cache=adapter.cache)
        self.assertEqual(adapter.generator.mock_calls, [])

    def test_a_single_capacity_server_still_warms_its_serving_buckets(self):
        # The occupancy sweep has nothing to do at capacity one, but a lone request still takes
        # the slot-zero path, and that path is the whole reason short prompts were slow.
        adapter = self.make_adapter(1)
        adapter.warmup_model_prefill(kv_cache=adapter.cache)
        self.assertEqual(adapter.generator.prefill_forward.call_count, 0)
        warmed = [call.kwargs["prompt_lens"][0] for call in adapter.generator.serving_prefill_tokens.call_args_list]
        self.assertEqual(warmed, [128, 96, 64, 32])
        self.assertTrue(adapter._prefill_startup_warmed)

    @patch("models.demos.qwen38_27b_t3k.tt.generator_vllm.ttnn.synchronize_device")
    def test_all_occupancies_then_reset_and_restore_table(self, synchronize):
        adapter = self.make_adapter()
        table = adapter.generator.page_host.clone()
        adapter.warmup_model_prefill(kv_cache=adapter.cache)
        occupancies = adapter.generator.prefill_forward.call_args_list[:4]
        # Largest first. The widest occupancy asks for the largest contiguous block of the whole
        # warmup, and running the narrow ones first leaves the heap in pieces too small for it:
        # the allocator reported 50.78 MiB per bank in one piece this way against 20.62 MiB the
        # other. A pass is capped at 8 rows now, so the largest ask is 26.67 MiB per bank.
        for batch, call in zip(range(4, 0, -1), occupancies):
            self.assertEqual(tuple(call.args[0].shape), (batch, 128))
            self.assertEqual(call.kwargs["slots"], list(range(batch)))
            self.assertIs(call.kwargs["kv_cache"], adapter.cache)
        # A lone request takes the slot-zero serving path, whose programs the occupancy sweep
        # above never builds, so each bucket is compiled through the call that will replay it.
        warmed = [call.kwargs["prompt_lens"][0] for call in adapter.generator.serving_prefill_tokens.call_args_list]
        # Largest first: only the last few buckets keep a trace, and a chat turn alternates
        # among the small ones.
        self.assertEqual(warmed, [128, 96, 64, 32])
        for call in adapter.generator.serving_prefill_tokens.call_args_list:
            self.assertEqual(call.kwargs["slots"], [0])
            self.assertEqual(call.kwargs["start_pos"], [0])
        # One reset ends the occupancy sweep, the other ends the bucket warmup. The width
        # sweep is off by default and resets nothing.
        self.assertEqual(adapter.generator.reset.call_count, 2)
        self.assertTrue(torch.equal(adapter.generator._refresh_table.call_args.args[0], table))
        self.assertFalse(adapter._decode_bound)
        self.assertTrue(adapter._prefill_startup_warmed)
        before = adapter.generator.prefill_forward.call_count
        adapter.warmup_model_prefill(kv_cache=adapter.cache)
        self.assertEqual(adapter.generator.prefill_forward.call_count, before)

    @patch("models.demos.qwen38_27b_t3k.tt.generator_vllm.ttnn.synchronize_device")
    def test_every_requested_width_is_compiled_at_the_full_occupancy(self, synchronize):
        # A grouped prefill is built around its row count, so this ladder is what lets a full
        # wave group at all; anything it misses is served row by row instead.
        adapter = self.make_adapter()
        adapter.prefill_warmup_widths = [32, 64, 96, 128]
        adapter.warmup_model_prefill(kv_cache=adapter.cache)
        widths = adapter.generator.prefill_forward.call_args_list[4:]
        self.assertEqual([call.kwargs["prompt_lens"][0] for call in widths], [32, 64, 96, 128])
        for call in widths:
            self.assertEqual(call.kwargs["slots"], [0, 1, 2, 3])
            self.assertEqual(tuple(call.args[0].shape)[0], 4)
            self.assertEqual(len(call.kwargs["prompt_lens"]), 4)
        # Rows must own disjoint pages, or a warm prompt writes over a page another row is using.
        table = widths[0].kwargs["page_table"]
        self.assertEqual(tuple(table.shape), tuple(adapter.generator.page_host.shape))
        owned = table[table > 0]
        self.assertEqual(owned.numel(), len(set(owned.tolist())))

    @patch("models.demos.qwen38_27b_t3k.tt.generator_vllm.ttnn.synchronize_device")
    def test_a_bucket_larger_than_the_rows_pages_is_left_out(self, synchronize):
        # Warming past a row's own pages would compile against a table the server cannot hand out.
        adapter = self.make_adapter()
        adapter.prefill_warmup_widths = [32, 64, 96, 128]
        adapter.cache = SimpleNamespace(num_pages=16, capacity=128)
        adapter.generator.cache = adapter.cache
        self.assertEqual(adapter._warm_bucket_widths(adapter.cache), [32, 64, 96])

    def test_the_smallest_buckets_are_warmed_last_so_they_stay_resident(self):
        # Only MAX_RESIDENT_PREFILL buckets keep their buffers and trace; the rest are evicted as
        # the ladder runs, so warm order decides which survive into serving.
        from models.demos.qwen38_27b_t3k.tt.generator import MAX_RESIDENT_PREFILL

        adapter = self.make_adapter(1)
        adapter.prefill_warmup_buckets = [32, 64, 96, 128, 160, 192, 224, 256]
        adapter.cache = SimpleNamespace(num_pages=64, capacity=256)
        adapter.generator.cache = adapter.cache
        adapter.warmup_model_prefill(kv_cache=adapter.cache)
        warmed = [call.kwargs["prompt_lens"][0] for call in adapter.generator.serving_prefill_tokens.call_args_list]
        self.assertEqual(warmed, sorted(warmed, reverse=True))
        self.assertEqual(sorted(warmed[-MAX_RESIDENT_PREFILL:]), [32, 64, 96, 128])

    def test_failure_restores_state_without_marking_ready(self):
        adapter = self.make_adapter()
        adapter.generator.prefill_forward.side_effect = RuntimeError("compile failed")
        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            adapter.warmup_model_prefill(kv_cache=adapter.cache)
        adapter.generator.reset.assert_called_once()
        adapter.generator._refresh_table.assert_called_once()
        self.assertFalse(getattr(adapter, "_prefill_startup_warmed", False))


if __name__ == "__main__":
    unittest.main()
