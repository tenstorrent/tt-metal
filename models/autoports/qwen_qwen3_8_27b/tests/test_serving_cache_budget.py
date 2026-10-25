# SPDX-License-Identifier: Apache-2.0
"""Host-only cache admission/geometry checks; no TTNN import or device access."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from models.autoports.qwen_qwen3_8_27b.tt import vllm_cache_specs as specs


class CacheBudgetTest(unittest.TestCase):
    def environment(self, block=None, budget=None):
        values = {k: v for k, v in os.environ.items() if k not in (
            'QWEN38_CACHE_BLOCK_SIZE', 'QWEN38_CACHE_BUDGET_BYTES_PER_DEVICE')}
        if block is not None:
            values['QWEN38_CACHE_BLOCK_SIZE'] = str(block)
        if budget is not None:
            values['QWEN38_CACHE_BUDGET_BYTES_PER_DEVICE'] = str(budget)
        return patch.dict(os.environ, values, clear=True)

    def config(self, block):
        text = SimpleNamespace(layer_types=['linear_attention'] * 3 + ['full_attention'])
        text.layer_types *= 16
        text.num_key_value_heads, text.head_dim = 4, 256
        return SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(text_config=text)),
            cache_config=SimpleNamespace(block_size=block,mamba_block_size=block,mamba_cache_mode='align'))

    def test_default_unchanged(self):
        with self.environment():
            self.assertEqual(specs.selected_block_size(),4096)
            self.assertEqual(specs.bounded_token_budget(262144,32),262144)
            self.assertEqual((262144//4096)+32*7+1,289)

    def test_512_budget_preserves_native_admission(self):
        budget=23661297664
        with self.environment(512,budget):
            tokens=specs.bounded_token_budget(262144,32)
            self.assertEqual(tokens,166912)
            # Actual worker order: output tokens, ceil division, recurrent/null.
            blocks=(tokens+32*512+511)//512 +32*6+1
            self.assertEqual(blocks,551)
            self.assertEqual(blocks*specs.physical_bytes_per_block(512),budget)
            self.assertEqual(262144//512+6+1+32,blocks)
            actual=specs.get_kv_cache_spec(self.config(512))
            self.assertEqual(len(actual),64)
            self.assertEqual({s.block_size for s in actual.values()},{512})
            self.assertEqual({s.page_size_bytes for s in actual.values()},{2097152})
            recurrence=actual['model.layers.0.state']
            self.assertEqual(recurrence.prefill_checkpoint_alignment,512)
            self.assertEqual(recurrence.num_prefill_checkpoint_blocks,0)
            # Exercise installed upstream specs' admission formulas rather than
            # treating the token budget as a native-context capacity claim.
            sizing=self.config(512)
            sizing.model_config.max_model_len=262144
            sizing.parallel_config=SimpleNamespace(decode_context_parallel_size=1)
            attention=actual['model.layers.3.state']
            attention_pages=attention.max_memory_usage_bytes(sizing)//attention.page_size_bytes
            recurrent_pages=recurrence.max_memory_usage_bytes(sizing)//recurrence.page_size_bytes
            self.assertEqual(attention_pages,512)
            self.assertEqual(recurrent_pages,2)
            self.assertEqual(attention_pages+3*recurrent_pages+1+32,blocks)

    def test_undersized_budget_rejected(self):
        with self.environment(512,23661297664-1):
            with self.assertRaisesRegex(ValueError,'native context'):
                specs.bounded_token_budget(262144,32)

    def test_fine_checkpoints_require_explicit_budget(self):
        for block in (256, 512):
            with self.environment(block):
                with self.assertRaisesRegex(ValueError, 'explicit physical cache'):
                    specs.bounded_token_budget(262144,32)

    def test_mismatched_cli_or_geometry_rejected(self):
        with self.environment(512,23661297664):
            with self.assertRaisesRegex(ValueError,'must match'):
                specs.get_kv_cache_spec(self.config(4096))
            config=self.config(512)
            config.model_config.hf_config.text_config.num_key_value_heads=8
            with self.assertRaisesRegex(ValueError,'all64'):
                specs.get_kv_cache_spec(config)
            config=self.config(512)
            config.cache_config.mamba_block_size=4096
            with self.assertRaisesRegex(ValueError,'boundaries'):
                specs.get_kv_cache_spec(config)

    def test_unsupported_geometry_rejected(self):
        for block in (0,128,768,8192):
            with self.environment(block):
                with self.assertRaises(ValueError):
                    specs.selected_block_size()


if __name__ == '__main__':
    unittest.main()
