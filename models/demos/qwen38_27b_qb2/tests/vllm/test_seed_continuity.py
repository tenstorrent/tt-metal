# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real adapter seed decisions without opening devices."""

import os
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.demos.qwen38_27b_qb2.tt.generator import Qwen38Generator
from models.demos.qwen38_27b_qb2.tt.generator_vllm import Qwen38ForCausalLM as Adapter


class FakeGenerator:
    set_batch_sampling_params = Qwen38Generator.set_batch_sampling_params

    def __init__(self):
        self.cache = object()
        self.page_host = torch.zeros(2, 4, dtype=torch.int32)
        self.counters = Counter()
        self.parameter_updates = []
        self.draws = []
        self.sampler = SimpleNamespace(
            seeds_tt_tensor=torch.zeros(32, dtype=torch.int64),
            reset_params=lambda *args: self.parameter_updates.append(args),
        )

    def _copy(self, source, target, counter):
        target.copy_(source)
        self.counters[counter] += 1

    def _draw(self):
        self.draws.append(self.sampler.seeds_tt_tensor.clone())
        self.sampler.seeds_tt_tensor.add_(1)
        return torch.zeros(32, 1, dtype=torch.long)

    def decode_forward(self, **kwargs):
        if kwargs["host_sampling"]:
            return torch.zeros(2, 16)
        return self._draw()

    def reset_recurrent_slots(self, slots):
        pass

    def remap_recurrent_slots(self, remap):
        pass

    def prefill_forward(self, *args, **kwargs):
        return [object()]

    def sample_prefill(self, outputs):
        return self._draw()

    def serving_prefill_tokens(self, tokens, **kwargs):
        return self.sample_prefill(self.prefill_forward(tokens, **kwargs))


def params(seeds=(42, 99), top_k=(5, 1)):
    return SimpleNamespace(
        temperature=[2.0] * len(seeds), top_k=list(top_k), top_p=[1.0] * len(seeds), seed=list(seeds)
    )


class SamplingModeTests(unittest.TestCase):
    def test_explicit_all_mode_uses_existing_host_boundary(self):
        with patch.dict(os.environ, {"QWEN_VLLM_HOST_COMPATIBILITY": "all"}):
            adapter = Adapter(FakeGenerator(), 2, 262144)
            self.assertTrue(adapter.host_compatibility)
            for decode in (False, True):
                self.assertFalse(Adapter.supports_device_sampling(None, is_decode=decode))
            self.assertFalse(adapter._sampling(None))

    def test_native_and_fallback_modes_keep_supported_requests_on_device(self):
        policy = SimpleNamespace(
            temperature=torch.tensor([0.0, 0.7]),
            top_k=torch.tensor([248320, 5]),
            presence_penalty=torch.zeros(2),
            frequency_penalty=torch.zeros(2),
            repetition_penalty=torch.ones(2),
        )
        for mode in ("", "1"):
            with patch.dict(os.environ, {"QWEN_VLLM_HOST_COMPATIBILITY": mode}):
                self.assertTrue(Adapter.supports_device_sampling(policy, is_decode=True))
        policy.top_k[1] = 100
        with patch.dict(os.environ, {"QWEN_VLLM_HOST_COMPATIBILITY": ""}):
            with self.assertRaisesRegex(ValueError, "explicit"):
                Adapter.supports_device_sampling(policy, is_decode=True)
        with patch.dict(os.environ, {"QWEN_VLLM_HOST_COMPATIBILITY": "1"}):
            self.assertFalse(Adapter.supports_device_sampling(policy, is_decode=True))


class SeedContinuityTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"QWEN_VLLM_HOST_COMPATIBILITY": "1"})
        env.start()
        self.addCleanup(env.stop)

    def adapter(self):
        generator = FakeGenerator()
        adapter = Adapter(generator, 2, 262144)
        adapter.cache = generator.cache
        adapter.read_decode_output = lambda output: output
        adapter.process_decode_output_host = lambda output, **kwargs: output
        return adapter

    def decode(self, adapter, *, positions=(4, 8), reset=False, sampling=None, remap=None, host=False):
        adapter.decode_forward(
            tokens=torch.tensor([[3], [7]]),
            start_pos=torch.tensor(positions),
            page_table=torch.zeros(2, 4, dtype=torch.int32),
            kv_cache=adapter.cache,
            sampling_params=None if host else sampling or params(),
            reset_batch=reset,
            slot_remap=remap,
            read_from_device=False,
        )
        return adapter.generator.draws[-1] if adapter.generator.draws else None

    def prefill(self, adapter, *, ends=(4, 8), sampling=None):
        adapter.prefill_forward(
            tokens=torch.ones(2, max(ends), dtype=torch.long),
            page_table=torch.zeros(2, 4, dtype=torch.int32),
            kv_cache=adapter.cache,
            prompt_lens=ends,
            sampling_params=sampling or params(),
        )
        return adapter.generator.draws[-1]

    def test_layout_reset_preserves_survivor_next_draw(self):
        control, changed = self.adapter(), self.adapter()
        for adapter in (control, changed):
            for _ in range(3):
                self.decode(adapter)
        expected = self.decode(control)
        actual = self.decode(changed, positions=(7, -1), reset=True)
        self.assertEqual(int(actual[0]), int(expected[0]))

    def test_row_remap_preserves_survivor_next_draw(self):
        control, changed = self.adapter(), self.adapter()
        for adapter in (control, changed):
            for _ in range(3):
                self.decode(adapter)
        expected = self.decode(control)
        actual = self.decode(changed, positions=(11, -1), reset=True, sampling=params((99, 42), (1, 5)), remap=[1, 0])
        self.assertEqual(int(actual[0]), int(expected[1]))

    def test_companion_parameter_change_preserves_seeds_with_stale_host_positions(self):
        control, changed = self.adapter(), self.adapter()
        for adapter in (control, changed):
            for _ in range(3):
                self.decode(adapter)
        before = changed.generator.counters["seed_refreshes"]
        expected = self.decode(control)
        actual = self.decode(changed, sampling=params(top_k=(5, 2)))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(changed.generator.counters["seed_refreshes"], before)
        self.assertEqual(changed.generator.parameter_updates[-1][0][:2], [5, 2])

    def test_prefill_and_first_decode_use_successive_output_positions(self):
        adapter = self.adapter()
        first = self.prefill(adapter)
        second = self.decode(adapter)
        self.assertEqual(first[:2].tolist(), [46, 107])
        self.assertEqual(second[:2].tolist(), [47, 108])

    def test_steady_decode_ignores_lagging_host_positions_and_does_not_reseed(self):
        adapter = self.adapter()
        first = self.decode(adapter)
        before = adapter.generator.counters.copy()
        second = self.decode(adapter, positions=(0, 0))
        self.assertTrue(torch.equal(second, first + 1))
        self.assertEqual(adapter.generator.counters, before)

    def test_seed_change_requires_authoritative_positions_before_mutation(self):
        adapter = self.adapter()
        self.decode(adapter)
        before = adapter.generator.sampler.seeds_tt_tensor.clone()
        updates = len(adapter.generator.parameter_updates)
        with self.assertRaisesRegex(ValueError, "authoritative batch reset"):
            self.decode(adapter, sampling=params((123, 99)))
        self.assertTrue(torch.equal(adapter.generator.sampler.seeds_tt_tensor, before))
        self.assertEqual(len(adapter.generator.parameter_updates), updates)
        actual = self.decode(adapter, sampling=params((123, 99)), positions=(5, 9), reset=True)
        self.assertEqual(actual[:2].tolist(), [129, 109])

    def test_intervening_prefill_does_not_rewind_existing_request(self):
        control, changed = self.adapter(), self.adapter()
        for adapter in (control, changed):
            for _ in range(3):
                self.decode(adapter)
        self.prefill(changed, sampling=params((11, 13)))
        expected = self.decode(control)
        actual = self.decode(changed, positions=(7, 11))
        self.assertTrue(torch.equal(actual[:2], expected[:2]))

    def test_host_to_device_transition_reanchors_before_sampling(self):
        control, changed = self.adapter(), self.adapter()
        for adapter in (control, changed):
            self.decode(adapter)
        self.decode(changed, positions=(5, 9), host=True)
        self.decode(control)
        expected = self.decode(control)
        actual = self.decode(changed, positions=(6, 10))
        self.assertTrue(torch.equal(actual[:2], expected[:2]))

    def test_large_seeds_do_not_wrap_across_authoritative_refresh(self):
        modulus = (1 << 31) - 262144 - 1
        for seed in (modulus - 1, 2**31 - 2, 2**31 - 1, 2**32 - 1, 2**63 - 1, -(2**63)):
            with self.subTest(seed=seed):
                control, changed = self.adapter(), self.adapter()
                sampling = params((seed, 7))
                for adapter in (control, changed):
                    self.decode(adapter, positions=(262142, 8), sampling=sampling)
                expected = self.decode(control, sampling=sampling)
                actual = self.decode(changed, positions=(262143, -1), reset=True, sampling=sampling)
                self.assertEqual(int(actual[0]), int(expected[0]))
                self.assertGreaterEqual(int(actual[0]), 0)
                self.assertLess(int(actual[0]), 2**31 - 1)
                self.assertLessEqual(int(changed.generator.sampler.seeds_tt_tensor[0]), 2**31 - 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
