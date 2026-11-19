# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only compatibility semantics; runnable with unittest, no TT fixtures."""

import unittest
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_8_flash_next.tt.host_sampling import HostSampler, HostSamplingParams


class HostSamplingTest(unittest.TestCase):
    def test_penalties_use_correct_histories_and_sign(self):
        sampler = HostSampler(
            HostSamplingParams(repetition_penalty=2.0, frequency_penalty=0.25, presence_penalty=0.5),
            prompt_token_ids=[0, 1],
            output_token_ids=[1, 2, 2],
        )
        logits = torch.tensor([4.0, -2.0, 3.0, 0.0])
        _, processed = sampler._processed_logits(logits)
        torch.testing.assert_close(processed, torch.tensor([2.0, -4.75, 0.5, 0.0]))
        torch.testing.assert_close(logits, torch.tensor([4.0, -2.0, 3.0, 0.0]))

    def test_bias_precedes_repetition_and_preserves_bans(self):
        sampler = HostSampler(
            HostSamplingParams(repetition_penalty=2.0, logit_bias={0: 4.0, 2: 1000.0}, allowed_token_ids=(0, 1)),
            prompt_token_ids=[0],
        )
        _, processed = sampler._processed_logits(torch.tensor([4.0, 3.0, 0.0]))
        self.assertEqual(processed[0], 4.0)
        self.assertTrue(torch.isneginf(processed[2]))
        self.assertEqual(sampler.sample(torch.tensor([4.0, 3.0, 0.0])).token_id, 0)

    def test_bad_words_match_generated_suffix_not_prompt(self):
        sampler = HostSampler(HostSamplingParams(bad_words_token_ids=((1, 2), (3,))), prompt_token_ids=[1])
        logits = torch.tensor([0.0, 0.0, 5.0, 6.0])
        self.assertEqual(sampler.sample(logits, commit=False).token_id, 2)
        sampler.append_token(1)
        self.assertEqual(sampler.sample(logits).token_id, 0)

    def test_min_tokens_and_grammar_mask(self):
        sampler = HostSampler(HostSamplingParams(min_tokens=2, eos_token_ids=(0,)))
        logits = torch.tensor([5.0, 4.0, 3.0])
        self.assertEqual(sampler.sample(logits).token_id, 1)
        self.assertEqual(sampler.sample(logits).token_id, 1)
        self.assertEqual(sampler.sample(logits).token_id, 0)
        mask = torch.tensor([False, False, True])
        self.assertEqual(sampler.sample(logits, allowed_token_mask=mask).token_id, 2)
        with self.assertRaisesRegex(ValueError, "no valid"):
            sampler.sample(logits, allowed_token_mask=torch.zeros(3, dtype=torch.bool))

    def test_top_p_top_k_and_temperature_then_min_p(self):
        logits = torch.tensor([0.6, 0.25, 0.1, 0.05]).log()
        sampler = HostSampler(HostSamplingParams(temperature=1.0, top_k=3, top_p=0.8))
        _, processed = sampler._processed_logits(logits)
        self.assertEqual(torch.isfinite(processed).tolist(), [True, True, False, False])
        sampler = HostSampler(HostSamplingParams(temperature=2.0, min_p=0.5))
        _, processed = sampler._processed_logits(logits)
        self.assertEqual(torch.isfinite(processed).tolist(), [True, True, False, False])
        torch.testing.assert_close(processed[:2], logits[:2] / 2)

    def test_seeded_request_is_independent_of_other_requests_and_reset(self):
        params = HostSamplingParams(temperature=0.8, top_k=4, seed=-7)
        first, second = HostSampler(params), HostSampler(params)
        other = HostSampler(HostSamplingParams(temperature=1.5, seed=42))
        logits = torch.tensor([0.1, 0.2, 0.3, 0.4])
        a, b = [], []
        for _ in range(30):
            a.append(first.sample(logits).token_id)
            for _ in range(3):
                other.sample(logits)
            b.append(second.sample(logits).token_id)
        self.assertEqual(a, b)
        self.assertGreater(len(set(a)), 1)
        reset = HostSampler(params)
        self.assertEqual(a, [reset.sample(logits).token_id for _ in a])

    def test_greedy_ties_do_not_consume_rng_and_teacher_forcing_commits_actual(self):
        sampler = HostSampler(HostSamplingParams(temperature=0, seed=5, top_k=1, min_p=1))
        before = sampler.generator.get_state().clone()
        result = sampler.sample(torch.tensor([3.0, 3.0, 1.0]), commit=False)
        self.assertEqual(result.token_id, 0)
        self.assertEqual(sampler.output_token_ids, [])
        sampler.append_token(2)
        self.assertEqual(sampler.output_token_ids, [2])
        torch.testing.assert_close(before, sampler.generator.get_state())

    def test_raw_logprobs_rank_ties_and_all_vocab(self):
        logits = torch.tensor([4.0, 4.0, 2.0])
        sampler = HostSampler(HostSamplingParams(logprobs=1, logit_bias={2: 100.0}))
        result = sampler.sample(logits)
        self.assertEqual(result.token_id, 2)
        self.assertAlmostEqual(result.sampled_logprob, float(logits.log_softmax(-1)[2]))
        self.assertEqual(result.selected_token_rank, 3)
        self.assertEqual(len(result.top_logprobs), 2)
        tied = HostSampler(HostSamplingParams(logprobs=0)).sample(logits)
        self.assertEqual(tied.selected_token_rank, 2)
        full = HostSampler(HostSamplingParams(logprobs=-1)).sample(logits)
        torch.testing.assert_close(full.full_logprobs, logits.log_softmax(-1))

    def test_common_per_row_params_and_unseeded_list(self):
        params = SimpleNamespace(
            temperature=[0.0, 0.7],
            top_k=[1, 20],
            top_p=[1.0, 0.9],
            seed=[None, -9],
            enable_log_probs=[False, True],
            num_logprobs=[0, 5],
        )
        self.assertIsNone(HostSamplingParams.from_object(params, row=0).seed)
        selected = HostSamplingParams.from_object(params, row=1)
        self.assertEqual((selected.temperature, selected.seed, selected.logprobs), (0.7, -9, 5))
        with self.assertRaises(ValueError):
            HostSamplingParams.from_object(params, row=2)

    def test_unbounded_top_k_and_invalid_input_do_not_silently_fallback(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        for k in (-1, 0, 100):
            _, processed = HostSampler(HostSamplingParams(temperature=1.0, top_k=k))._processed_logits(logits)
            torch.testing.assert_close(logits, processed)
        for bad in (torch.tensor([float("nan"), 0.0]), torch.full((3,), -torch.inf), torch.ones(2, 3)):
            with self.assertRaises(ValueError):
                HostSampler().sample(bad)
        with self.assertRaises(ValueError):
            HostSampler(HostSamplingParams(allowed_token_ids=(99,))).sample(logits)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
