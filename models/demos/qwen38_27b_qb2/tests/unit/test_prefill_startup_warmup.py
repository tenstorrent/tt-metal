# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Startup compilation must not leave synthetic request state in the server."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from models.demos.qwen38_27b_qb2.tt.generator_vllm import Qwen38ForCausalLM


class StartupWarmupTests(unittest.TestCase):
    def make_adapter(self, batch=4):
        adapter = object.__new__(Qwen38ForCausalLM)
        adapter.batch_size = batch
        adapter.context = 128
        adapter.prefill_startup_warmup = True
        adapter.cache = SimpleNamespace(num_pages=64)
        adapter.generator = MagicMock()
        adapter.generator.cache = adapter.cache
        adapter.generator.batched_prefill = True
        adapter.generator.page_host = torch.zeros(batch, 4, dtype=torch.int32)
        adapter._decode_bound = True
        return adapter

    def test_default_off_and_b1_do_not_change_state(self):
        for batch, enabled in ((4, False), (1, True)):
            adapter = self.make_adapter(batch)
            adapter.prefill_startup_warmup = enabled
            adapter.warmup_model_prefill(kv_cache=adapter.cache)
            self.assertEqual(adapter.generator.mock_calls, [])

    @patch("models.demos.qwen38_27b_qb2.tt.generator_vllm.ttnn.synchronize_device")
    def test_all_occupancies_then_reset_and_restore_table(self, synchronize):
        adapter = self.make_adapter()
        table = adapter.generator.page_host.clone()
        adapter.warmup_model_prefill(kv_cache=adapter.cache)
        self.assertEqual(adapter.generator.prefill_forward.call_count, 4)
        for batch, call in enumerate(adapter.generator.prefill_forward.call_args_list, 1):
            self.assertEqual(tuple(call.args[0].shape), (batch, 128))
            self.assertEqual(call.kwargs["slots"], list(range(batch)))
            self.assertIs(call.kwargs["kv_cache"], adapter.cache)
        adapter.generator.reset.assert_called_once()
        self.assertTrue(torch.equal(adapter.generator._refresh_table.call_args.args[0], table))
        self.assertFalse(adapter._decode_bound)
        self.assertTrue(adapter._prefill_startup_warmed)
        adapter.warmup_model_prefill(kv_cache=adapter.cache)
        self.assertEqual(adapter.generator.prefill_forward.call_count, 4)

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
