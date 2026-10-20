# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the actual vLLM prefill consumer without constructing a device or model."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.demos.qwen38_27b_qb2.tt.generator_vllm import Qwen38ForCausalLM


class FakePrefillGenerator:
    def __init__(self):
        self.cache = object()
        self.page_host = torch.arange(16, dtype=torch.int32).reshape(4, 4)
        self.calls = []
        self.resets = []
        self.sampled = []

    def reset_recurrent_slots(self, slots):
        self.resets.append(list(slots))

    def set_batch_sampling_params(self, **kwargs):
        self.sampling = kwargs

    def prefill_forward(self, tokens, **kwargs):
        self.calls.append((tokens.clone(), kwargs))
        logits = torch.zeros(1, 1, 1, 256)
        logits[..., int(tokens[0, -1])] = 10.0
        return [logits]

    def sample_prefill(self, outputs):
        self.sampled = list(outputs)
        tokens = torch.full((32, 1), 255, dtype=torch.int32)
        for row, logits in enumerate(outputs):
            tokens[row, 0] = logits.argmax()
        return tokens

    def serving_prefill_tokens(self, tokens, *, page_table, kv_cache, prompt_lens, start_pos, slots):
        outputs = []
        for row, (start, end, slot) in enumerate(zip(start_pos, prompt_lens, slots)):
            outputs.extend(
                self.prefill_forward(
                    tokens[row : row + 1, start:end],
                    page_table=page_table,
                    kv_cache=kv_cache,
                    prompt_lens=[end - start],
                    start_pos=[start],
                    slots=[slot],
                )
            )
        return self.sample_prefill(outputs)

    def _host_logits(self, output):
        return output


class PrefillHostTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"QWEN_VLLM_HOST_COMPATIBILITY": "0"})
        env.start()
        self.addCleanup(env.stop)
        self.gen = FakePrefillGenerator()
        self.adapter = Qwen38ForCausalLM(self.gen, 4, 128)
        self.adapter.cache = self.gen.cache
        self.adapter._decode_bound = True
        read = patch.object(self.adapter, "read_decode_output", side_effect=lambda tensor: tensor.clone())
        read.start()
        self.addCleanup(read.stop)
        convert = patch(
            "models.demos.qwen38_27b_qb2.tt.generator_vllm.ttnn.to_torch", side_effect=lambda tensor: tensor
        )
        convert.start()
        self.addCleanup(convert.stop)

    def inputs(self, n=2):
        return SimpleNamespace(
            input_tokens=torch.stack([torch.arange(40) + 100 * row for row in range(n)]),
            input_positions=torch.tensor([0, 32][:n]),
            prompt_lens=[3, 35][:n],
            block_tables=torch.tensor([[29, 31], [37, 41]], dtype=torch.int32)[:n],
            prefill_empty_slots=[3, 1][:n],
            sampling_params=SimpleNamespace(
                temperature=torch.zeros(n),
                top_k=torch.ones(n, dtype=torch.int32),
                top_p=torch.ones(n),
                seed=torch.tensor([7, 11][:n]),
            ),
        )

    def test_device_sampling_preserves_fixed_slots_and_packed_outputs(self):
        mi = self.inputs()
        before = self.gen.page_host.clone()
        result, deltas = self.adapter.prefill_forward(
            tokens=mi.input_tokens,
            page_table=mi.block_tables,
            kv_cache=self.gen.cache,
            prompt_lens=mi.prompt_lens,
            start_pos=mi.input_positions,
            sampling_params=mi.sampling_params,
            empty_slots=mi.prefill_empty_slots,
        )
        expected_table = before.clone()
        expected_table[[3, 1]] = 0
        expected_table[[3, 1], :2] = mi.block_tables
        self.assertEqual(self.gen.resets, [[3]])
        self.assertEqual(len(self.gen.calls), 2)
        for row, (tokens, kwargs) in enumerate(self.gen.calls):
            start, end = int(mi.input_positions[row]), mi.prompt_lens[row]
            self.assertTrue(torch.equal(tokens, mi.input_tokens[row : row + 1, start:end]))
            self.assertEqual(kwargs["start_pos"], [start])
            self.assertEqual(kwargs["prompt_lens"], [end - start])
            self.assertEqual(kwargs["slots"], [mi.prefill_empty_slots[row]])
            self.assertIs(kwargs["kv_cache"], self.gen.cache)
            self.assertTrue(torch.equal(kwargs["page_table"], expected_table))
        self.assertTrue(torch.equal(self.gen.page_host, before))
        self.assertEqual(self.gen.sampling["seed"][:2], [10, 46])
        self.assertFalse(self.adapter._decode_bound)
        self.assertEqual(result.flatten().tolist(), [2, 134])
        self.assertTrue(torch.equal(deltas, torch.zeros(2, dtype=torch.int64)))

    def test_host_logits_and_default_rows_return_tensor_rope_deltas(self):
        self.adapter.host_compatibility = True
        mi = self.inputs()
        result, deltas = self.adapter.prefill_forward(
            tokens=mi.input_tokens,
            page_table=mi.block_tables,
            kv_cache=self.gen.cache,
            prompt_lens=[3, 5],
        )
        self.assertEqual(result.shape, (2, 1, 256))
        self.assertTrue(result.is_floating_point())
        self.assertEqual(result[:, -1, :].argmax(dim=-1).tolist(), [2, 104])
        self.assertEqual(deltas.shape, (2,))
        self.assertEqual(deltas.dtype, torch.int64)
        self.assertEqual(deltas.device.type, "cpu")
        self.assertEqual([delta.item() for delta in deltas], [0, 0])
        self.assertEqual(self.gen.resets, [[0, 1]])
        self.assertEqual([kwargs["slots"] for _, kwargs in self.gen.calls], [[0], [1]])
        self.assertEqual(self.gen.sampled, [])

    def test_wrong_cache_is_rejected_before_prefill_effects(self):
        mi = self.inputs(1)
        with self.assertRaisesRegex(ValueError, "exact vLLM allocated cache"):
            self.adapter.prefill_forward(
                mi.input_tokens,
                mi.block_tables,
                object(),
                mi.prompt_lens,
                mi.input_positions,
                mi.sampling_params,
                mi.prefill_empty_slots,
            )
        self.gen.cache = object()
        with self.assertRaisesRegex(ValueError, "exact vLLM allocated cache"):
            self.adapter.prefill_forward(
                mi.input_tokens,
                mi.block_tables,
                self.adapter.cache,
                mi.prompt_lens,
                mi.input_positions,
                mi.sampling_params,
                mi.prefill_empty_slots,
            )
        self.assertEqual(self.gen.resets, [])
        self.assertEqual(self.gen.calls, [])
        self.assertFalse(hasattr(self.gen, "sampling"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
