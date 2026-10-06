# SPDX-License-Identifier: Apache-2.0
"""Host-only regression for oversized physical chunks in a circular KV pool."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ..tt.multichip_decoder import MeshPolicy, MultichipDecoder
from ..tt.optimized_decoder import OptimizedDecoder


class RingGuard(unittest.TestCase):
    def setUp(self):
        self.model = MultichipDecoder()
        self.model.policy = MeshPolicy()
        self.model.sliding_cache_tokens = 4608

    def test_plan_bounds_chunks_without_reducing_logical_extent(self):
        for logical in (8192, 8193, 1048576):
            with patch.object(OptimizedDecoder, "prepare_prefill", return_value="planned") as planner:
                result = self.model.prepare_prefill(page_table_host=None, seq_len=logical, chunk_size=8192)
                self.assertEqual(result, "planned")
                self.assertEqual(planner.call_args.kwargs["seq_len"], logical)
                self.assertEqual(planner.call_args.kwargs["chunk_size"], 4096)

    def test_invalid_chunk_does_not_get_silently_rounded(self):
        for size in (0, -32, 8193):
            with self.assertRaises(ValueError):
                self.model.prepare_prefill(page_table_host=None, seq_len=8193, chunk_size=size)

    def test_non_circular_plan_keeps_requested_chunk(self):
        self.model.sliding_cache_tokens = None
        with patch.object(OptimizedDecoder, "prepare_prefill") as planner:
            self.model.prepare_prefill(page_table_host=None, seq_len=8193, chunk_size=8192)
            self.assertEqual(planner.call_args.kwargs["chunk_size"], 8192)

    def test_direct_chunk_rejected_before_any_device_operation(self):
        self.model._qkv = lambda *args: self.fail("Unsafe chunk reached device work")
        for size in (4128, 8192):
            with self.assertRaisesRegex(ValueError, "circular-cache capacity"):
                self.model.prefill_chunk_forward(
                    SimpleNamespace(shape=(1, 1, size, 2560)),
                    kv_cache=None,
                    page_table=None,
                    chunk_page_table=None,
                    chunk_start=None,
                )


if __name__ == "__main__":
    unittest.main()
