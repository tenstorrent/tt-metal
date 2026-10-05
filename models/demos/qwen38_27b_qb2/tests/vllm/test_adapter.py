# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only adapter state tests; no model construction or device execution."""

import os
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from models.demos.qwen38_27b_qb2.tt import generator_vllm as adapter_module
from models.demos.qwen38_27b_qb2.tt.generator import Qwen38Generator
from models.demos.qwen38_27b_qb2.tt.generator_vllm import Qwen38ForCausalLM


class KVPoolConfigurationTests(unittest.TestCase):
    def test_default_preserves_single_request_pool(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Qwen38ForCausalLM.get_max_tokens_all_users(262144), 262144)

    def test_concurrent_pool_fits_eight_long_requests(self):
        with patch.dict(os.environ, {"QWEN_VLLM_KV_POOL_TOKENS": "1050592"}):
            self.assertEqual(
                Qwen38ForCausalLM.get_max_tokens_all_users(262144, max_num_seqs=8),
                8 * (131072 + 252),
            )

    def test_invalid_or_oversized_pool_is_rejected(self):
        for value in ("-1", "0", "262145", "2097152", "1e6", "１２３"):
            with self.subTest(value=value), patch.dict(os.environ, {"QWEN_VLLM_KV_POOL_TOKENS": value}):
                with self.assertRaises(ValueError):
                    Qwen38ForCausalLM.get_max_tokens_all_users(262144)


class FakeGenerator:
    """Use the real page refresh decision and replace only device effects."""

    _refresh_table = Qwen38Generator._refresh_table

    def __init__(self, batch_size=2, width=4):
        self.cache = SimpleNamespace(batch_size=batch_size, capacity=width * 32, num_pages=64)
        self.page_host = torch.zeros(batch_size, width, dtype=torch.int32)
        self.page_table = torch.zeros_like(self.page_host)
        self.counters = Counter()
        self.events = []
        self.calls = []
        self.tokens = torch.tensor([[701], [702]], dtype=torch.int32)[:batch_size]
        self.positions = torch.tensor([64, 96], dtype=torch.int32)[:batch_size]
        self.output = object()

    def _copy(self, host, target, counter):
        target.copy_(host)
        self.counters[counter] += 1
        self.events.append(counter)

    def set_batch_sampling_params(self, **kwargs):
        self.events.append("sampling")
        self.sampling = kwargs

    def remap_recurrent_slots(self, remap):
        self.events.append("remap")
        self.remap = list(remap)

    def decode_forward(self, **kwargs):
        self.calls.append(kwargs)
        self._refresh_table(kwargs["page_table"])
        if kwargs["tokens"] is not None:
            self._copy(kwargs["tokens"], self.tokens, "token_refreshes")
        if kwargs["start_pos"] is not None:
            self._copy(kwargs["start_pos"], self.positions, "position_refreshes")
        self.events.append("decode")
        return self.output


class AdapterHostTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"QWEN_VLLM_HOST_COMPATIBILITY": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.gen = FakeGenerator()
        self.adapter = Qwen38ForCausalLM(self.gen, 2, 128)
        self.adapter.cache = self.gen.cache
        self.params = SimpleNamespace(temperature=[0.0, 0.0], top_k=[1, 1], top_p=[1.0, 1.0], seed=[3, 5])
        # Enter steady decode without constructing hardware or staging host inputs.
        self.adapter._decode_bound = True
        self.adapter._sampling(self.params)
        self.gen.events.clear()
        self.table = torch.tensor([[4, 7, 0, 0], [11, 13, 17, 0]], dtype=torch.int32)
        self.gen._refresh_table(self.table)
        self.gen.events.clear()
        self.gen.counters.clear()

    def decode(self, **kwargs):
        args = dict(
            tokens=torch.tensor([[601], [602]], dtype=torch.int32),
            start_pos=torch.tensor([63, 95], dtype=torch.int32),
            page_table=self.table,
            kv_cache=self.gen.cache,
            sampling_params=self.params,
            reset_batch=False,
            read_from_device=False,
        )
        args.update(kwargs)
        return self.adapter.decode_forward(**args)

    def test_growth_refreshes_pages_and_preserves_device_token_position(self):
        grown = self.table.clone()
        grown[0, 2] = 23
        token_before, position_before = self.gen.tokens.clone(), self.gen.positions.clone()
        device_table = self.gen.page_table

        self.assertIs(self.decode(page_table=grown), self.gen.output)

        self.assertIs(self.gen.page_table, device_table)
        self.assertTrue(torch.equal(self.gen.page_table, grown))
        self.assertTrue(torch.equal(self.gen.tokens, token_before))
        self.assertTrue(torch.equal(self.gen.positions, position_before))
        self.assertIsNone(self.gen.calls[-1]["tokens"])
        self.assertIsNone(self.gen.calls[-1]["start_pos"])
        self.assertIsNone(self.gen.calls[-1]["active_slots"])
        self.assertEqual(self.gen.counters, {"page_table_refreshes": 1})
        self.assertEqual(self.gen.events, ["page_table_refreshes", "decode"])

    def test_value_identical_tables_do_not_refresh_each_decode(self):
        for _ in range(3):
            self.decode(page_table=self.table.clone())
        self.assertEqual(self.gen.counters, {})
        self.assertEqual(self.gen.events, ["decode"] * 3)

    def test_narrow_table_is_padded_without_false_growth(self):
        self.gen._refresh_table(torch.tensor([[4, 7, 0, 0], [11, 13, 0, 0]], dtype=torch.int32))
        self.gen.counters.clear()
        self.decode(page_table=torch.tensor([[4, 7], [11, 13]], dtype=torch.int32))
        self.assertEqual(self.gen.counters, {})
        self.assertEqual(tuple(self.gen.calls[-1]["page_table"].shape), (2, 4))

    def test_caller_mutation_is_detected_without_aliasing_saved_table(self):
        incoming = self.table.clone()
        self.decode(page_table=incoming)
        incoming[0, 2] = 29
        self.assertEqual(self.gen.page_host[0, 2].item(), 0)
        self.decode(page_table=incoming)
        self.assertEqual(self.gen.page_host[0, 2].item(), 29)
        self.assertEqual(self.gen.counters, {"page_table_refreshes": 1})

    def test_first_decode_binds_authoritative_inputs_even_without_reset_flag(self):
        self.adapter._decode_bound = False
        self.decode()
        self.assertTrue(torch.equal(self.gen.tokens, torch.tensor([[601], [602]], dtype=torch.int32)))
        self.assertTrue(torch.equal(self.gen.positions, torch.tensor([63, 95], dtype=torch.int32)))
        self.assertEqual(self.gen.calls[-1]["active_slots"], [0, 1])
        self.assertEqual(self.gen.counters, {"token_refreshes": 1, "position_refreshes": 1})
        self.assertTrue(self.adapter._decode_bound)

    def test_real_reset_reloads_host_inputs_and_recomputes_active_slots(self):
        fresh_tokens = torch.tensor([[801], [0]], dtype=torch.int32)
        fresh_positions = torch.tensor([65, -1], dtype=torch.int32)
        self.decode(reset_batch=True, tokens=fresh_tokens, start_pos=fresh_positions)
        self.assertTrue(torch.equal(self.gen.tokens, fresh_tokens))
        self.assertTrue(torch.equal(self.gen.positions, fresh_positions))
        self.assertEqual(self.gen.calls[-1]["active_slots"], [0])
        self.assertEqual(self.gen.counters, {"token_refreshes": 1, "position_refreshes": 1})
        self.assertEqual(self.gen.events, ["sampling", "token_refreshes", "position_refreshes", "decode"])

    def test_wrong_cache_is_rejected_before_remap_or_sampling(self):
        with self.assertRaisesRegex(ValueError, "exact vLLM allocated cache"):
            self.decode(kv_cache=SimpleNamespace(**vars(self.gen.cache)), reset_batch=True, slot_remap=[1, 0])
        self.assertEqual(self.gen.events, [])
        self.assertEqual(self.gen.calls, [])

    def test_generator_cache_identity_must_also_match(self):
        bound_cache = self.adapter.cache
        self.gen.cache = SimpleNamespace(**vars(bound_cache))
        with self.assertRaisesRegex(ValueError, "exact vLLM allocated cache"):
            self.decode(kv_cache=bound_cache, reset_batch=True, slot_remap=[1, 0])
        self.assertEqual(self.gen.events, [])

    def test_slot_remap_precedes_reset_inputs_and_decode(self):
        remapped_table = self.table[[1, 0]]
        new_tokens = torch.tensor([[901], [902]], dtype=torch.int32)
        new_positions = torch.tensor([97, 65], dtype=torch.int32)
        self.decode(
            slot_remap=[1, 0],
            reset_batch=True,
            tokens=new_tokens,
            start_pos=new_positions,
            page_table=remapped_table,
        )
        self.assertEqual(self.gen.remap, [1, 0])
        self.assertEqual(
            self.gen.events,
            ["remap", "sampling", "page_table_refreshes", "token_refreshes", "position_refreshes", "decode"],
        )
        self.assertTrue(torch.equal(self.gen.page_table, remapped_table))
        self.assertTrue(torch.equal(self.gen.tokens, new_tokens))
        self.assertTrue(torch.equal(self.gen.positions, new_positions))

    def test_out_of_pool_growth_is_rejected_by_real_table_validator(self):
        grown = self.table.clone()
        grown[0, 2] = self.gen.cache.num_pages
        with self.assertRaisesRegex(ValueError, "outside the bound physical cache"):
            self.decode(page_table=grown)
        self.assertEqual(self.gen.counters, {})
        self.assertNotIn("decode", self.gen.events)

    def test_raw_device_tokens_read_one_replica_before_host_conversion(self):
        raw_device, host = object(), object()
        replicas = [Mock(), Mock(), Mock(), Mock()]
        replicas[0].cpu.return_value = host
        padded_tokens = torch.tensor([73, 91] + [0] * 30, dtype=torch.int32)

        def convert(value):
            self.assertIs(value, host, "Raw multi-device tokens must not reach to_torch")
            return padded_tokens

        with (
            patch.object(adapter_module.ttnn, "is_tensor_storage_on_device", side_effect=lambda x: x is raw_device),
            patch.object(adapter_module.ttnn, "get_device_tensors", return_value=replicas) as get_shards,
            patch.object(adapter_module.ttnn, "to_torch", side_effect=convert),
            patch.object(adapter_module.ttnn, "record_event") as record_event,
        ):
            result = self.adapter.process_decode_output_host(raw_device, is_tokens=True)

        get_shards.assert_called_once_with(raw_device)
        replicas[0].cpu.assert_called_once_with(blocking=True)
        for replica in replicas[1:]:
            replica.cpu.assert_not_called()
        record_event.assert_not_called()
        self.assertEqual(self.gen.counters, {"token_readbacks": 1})
        self.assertTrue(torch.equal(result, torch.tensor([[73], [91]], dtype=torch.int64)))

    def test_already_read_host_tokens_are_not_read_again(self):
        host = object()
        with (
            patch.object(adapter_module.ttnn, "is_tensor_storage_on_device", return_value=False),
            patch.object(adapter_module.ttnn, "to_torch", return_value=torch.tensor([43, 61, 0])),
            patch.object(self.adapter, "read_decode_output") as read,
        ):
            result = self.adapter.process_decode_output_host(host, is_tokens=True)
        read.assert_not_called()
        self.assertEqual(self.gen.counters, {})
        self.assertTrue(torch.equal(result, torch.tensor([[43], [61]], dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
