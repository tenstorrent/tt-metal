# SPDX-License-Identifier: Apache-2.0
"""Focused host-only categorical implementation tests."""
import os

os.environ["VLLM_PLUGINS"] = ""
import sys
import unittest
from unittest.mock import patch

import torch
from benchmark_multinomial_sampler import ExperimentalMultinomialTopKTopPSampler, cpu_multinomial_sample
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample


class MultinomialTests(unittest.TestCase):
    def test_zero_bins_single_support_and_extreme_weights(self):
        for row in [[0.0, 0.0, 1.0, 0.0], [0.0, 1e-38, 0.0, 0.0], [1.0, 1e-30, 0.0, 0.0], [1e-30, 1.0, 0.0, 0.0]]:
            weights = torch.tensor([row]).expand(10000, -1)
            tokens = cpu_multinomial_sample(weights, {})
            self.assertTrue((weights.gather(1, tokens[:, None]) > 0).all())

    def test_invalid_weights_fallback_preserves_rng(self):
        for weights in [
            torch.tensor([[float("nan"), 1.0]]),
            torch.tensor([[float("inf"), 1.0]]),
            torch.zeros(1, 3),
            torch.tensor([[-0.1, 1.0]]),
        ]:
            state = torch.random.get_rng_state()
            expected = random_sample(weights.clone(), {})
            after = torch.random.get_rng_state()
            torch.random.set_rng_state(state)
            actual = cpu_multinomial_sample(weights.clone(), {})
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(after, torch.random.get_rng_state()))

    def test_fp64_request_fallback(self):
        a = {0: torch.Generator().manual_seed(42)}
        b = {0: torch.Generator().manual_seed(42)}
        probs = torch.tensor([[0.1, 0.2, 0.7]])
        self.assertTrue(
            torch.equal(random_sample(probs.clone(), a, True), cpu_multinomial_sample(probs.clone(), b, True))
        )
        self.assertTrue(torch.equal(a[0].get_state(), b[0].get_state()))

    def test_generator_ownership_repeatability_and_two_draw_contract(self):
        weights = torch.tensor([[0.125, 0.25, 0.625]]).expand(32, -1)
        for count in [0, 7, 32]:
            actual_gens = {i: torch.Generator().manual_seed(i) for i in range(count)}
            expected_gens = {i: torch.Generator().manual_seed(i) for i in range(count)}
            before = torch.random.get_rng_state().clone()
            actual = cpu_multinomial_sample(weights, actual_gens)
            after = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(before)
            expected = (
                torch.multinomial(weights, 2, replacement=True)[:, 0].clone()
                if count != 32
                else torch.empty(32, dtype=torch.long)
            )
            for row, gen in expected_gens.items():
                expected[row] = torch.multinomial(weights[row], 2, replacement=True, generator=gen)[0]
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(after, torch.random.get_rng_state()))
            self.assertEqual(torch.equal(before, after), count == 32)
            for row in actual_gens:
                self.assertTrue(torch.equal(actual_gens[row].get_state(), expected_gens[row].get_state()))

    def test_row_generator_reordering(self):
        weights = torch.tensor([[0.125, 0.25, 0.625], [0.125, 0.25, 0.625]])
        a, b = torch.Generator().manual_seed(81), torch.Generator().manual_seed(91)
        state_a, state_b = a.get_state(), b.get_state()
        original = cpu_multinomial_sample(weights, {0: a, 1: b})
        a.set_state(state_a)
        b.set_state(state_b)
        reordered = cpu_multinomial_sample(weights, {0: b, 1: a})
        self.assertTrue(torch.equal(original, reordered.flip(0)))

    def test_filter_and_logprob_modes(self):
        logits = torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0], [4.0, 1.0, 1.0, -float("inf"), -2.0]])
        for k in [None, torch.tensor([1, 3]), torch.tensor([5, 5])]:
            for p in [
                None,
                torch.tensor([0.0, 1.0]),
                torch.tensor([0.5, 0.95]),
                torch.tensor([-0.1, 1.1]),
                torch.tensor([float("nan"), 0.95]),
            ]:
                for mode in ["raw_logprobs", "raw_logits", "processed_logits", "processed_logprobs"]:
                    expected = apply_top_k_top_p(logits.clone(), k, p)
                    captured = {}

                    def draw(probs, generators, use_fp64_gumbel=False):
                        captured["probs"] = probs.clone()
                        return cpu_multinomial_sample(probs, generators, use_fp64_gumbel)

                    with patch("benchmark_multinomial_sampler.cpu_multinomial_sample", side_effect=draw):
                        tokens, scores = ExperimentalMultinomialTopKTopPSampler(mode).forward_native(
                            logits.clone(), {}, k, p
                        )
                    self.assertTrue(torch.equal(captured["probs"], expected.softmax(-1)))
                    self.assertTrue(torch.isfinite(expected.gather(1, tokens[:, None])).all())
                    if mode == "processed_logits":
                        self.assertTrue(torch.equal(scores, expected))
                    elif mode == "processed_logprobs":
                        self.assertTrue(torch.equal(scores, expected.log_softmax(-1)))
                    else:
                        self.assertIsNone(scores)

    def test_empty_batch(self):
        for mode in ["raw_logprobs", "processed_logits", "processed_logprobs"]:
            tokens, scores = ExperimentalMultinomialTopKTopPSampler(mode).forward_native(
                torch.empty(0, 10), {}, None, torch.empty(0)
            )
            self.assertEqual(tuple(tokens.shape), (0,))
            if scores is not None:
                self.assertEqual(tuple(scores.shape), (0, 10))

    def test_empirical_marginal_distribution(self):
        torch.manual_seed(810)
        weights = torch.tensor([[0.0, 0.125, 0.25, 0.625, 0.0]]).expand(100000, -1)
        tokens = cpu_multinomial_sample(weights, {})
        frequencies = torch.bincount(tokens, minlength=5).double() / len(tokens)
        self.assertTrue((frequencies - weights[0]).abs().max() < 0.005)
        self.assertEqual(int(((tokens == 0) | (tokens == 4)).sum()), 0)


if __name__ == "__main__":
    assert not any(n == "ttnn" or n.startswith("ttnn.") for n in sys.modules)
    unittest.main(verbosity=2)
