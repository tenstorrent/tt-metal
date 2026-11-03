# SPDX-License-Identifier: Apache-2.0
"""Direct host-only tests, avoiding tt-metal's hardware pytest conftest."""
import os

os.environ["VLLM_PLUGINS"] = ""
import sys
import unittest
from unittest.mock import patch

import torch
from benchmark_cdf_sampler import ExperimentalCDFTopKTopPSampler, cpu_cdf_sample, exact_cdf_or_none
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample


class CDFTests(unittest.TestCase):
    def test_zero_bins_and_single_support(self):
        weights = torch.tensor([[0.0, 0.0, 1.0, 0.0]]).expand(1000, -1)
        self.assertTrue((cpu_cdf_sample(weights, {}) == 2).all())

    def test_cdf_interval_weights(self):
        weights = torch.tensor([[0.125, 0.0, 0.25, 0.625], [0.0, 0.5, 0.5, 0.0]])
        cumulative = exact_cdf_or_none(weights)
        self.assertIsNotNone(cumulative)
        widths = torch.diff(cumulative, dim=-1, prepend=torch.zeros_like(cumulative[:, :1]))
        self.assertTrue(torch.equal(widths, weights.double()))

    def test_rounding_and_invalid_fallback(self):
        for weights in [torch.tensor([[1.0, 1e-20]]), torch.tensor([[float("nan"), 1.0]]), torch.zeros(1, 3)]:
            self.assertIsNone(exact_cdf_or_none(weights))
            state = torch.random.get_rng_state()
            expected = random_sample(weights.clone(), {})
            after = torch.random.get_rng_state()
            torch.random.set_rng_state(state)
            actual = cpu_cdf_sample(weights.clone(), {})
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(after, torch.random.get_rng_state()))

    def test_fp64_requested_falls_back_with_same_rng(self):
        weights = torch.tensor([[0.1, 0.2, 0.7]])
        a = {0: torch.Generator().manual_seed(42)}
        b = {0: torch.Generator().manual_seed(42)}
        self.assertTrue(torch.equal(random_sample(weights.clone(), a, True), cpu_cdf_sample(weights.clone(), b, True)))
        self.assertTrue(torch.equal(a[0].get_state(), b[0].get_state()))

    def test_generator_ownership_and_reproducibility(self):
        weights = torch.tensor([[0.125, 0.25, 0.625]]).expand(32, -1)
        for count in [0, 7, 32]:
            a = {i: torch.Generator().manual_seed(i) for i in range(count)}
            b = {i: torch.Generator().manual_seed(i) for i in range(count)}
            unused = torch.Generator().manual_seed(999)
            untouched = unused.get_state().clone()
            before = torch.random.get_rng_state().clone()
            actual = cpu_cdf_sample(weights, a)
            after = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(before)
            replay = cpu_cdf_sample(weights, b)
            self.assertTrue(torch.equal(actual, replay))
            self.assertTrue(torch.equal(untouched, unused.get_state()))
            self.assertEqual(torch.equal(before, after), count == 32)
            for i in a:
                self.assertTrue(torch.equal(a[i].get_state(), b[i].get_state()))

    def test_k_p_and_score_modes_match_upstream(self):
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
                    sampler = ExperimentalCDFTopKTopPSampler(mode)
                    expected = apply_top_k_top_p(logits.clone(), k, p)
                    captured = {}

                    def draw(probs, generators, use_fp64_gumbel=False):
                        captured["probs"] = probs.clone()
                        return cpu_cdf_sample(probs, generators, use_fp64_gumbel)

                    with patch("benchmark_cdf_sampler.cpu_cdf_sample", side_effect=draw):
                        tokens, scores = sampler.forward_native(logits.clone(), {}, k, p)
                    self.assertTrue(torch.equal(captured["probs"], expected.softmax(-1)))
                    self.assertTrue(torch.isfinite(expected.gather(1, tokens[:, None])).all())
                    if mode == "processed_logits":
                        self.assertTrue(torch.equal(scores, expected))
                    elif mode == "processed_logprobs":
                        self.assertTrue(torch.equal(scores, expected.log_softmax(-1)))
                    else:
                        self.assertIsNone(scores)

    def test_invalid_top_k_preserves_upstream_error(self):
        logits = torch.zeros(2, 5)
        for k in [torch.tensor([0, 1]), torch.tensor([6, 1])]:
            with self.assertRaises(RuntimeError):
                apply_top_k_top_p(logits.clone(), k, None)
            with self.assertRaises(RuntimeError):
                ExperimentalCDFTopKTopPSampler().forward_native(logits.clone(), {}, k, None)

    def test_empty_batch(self):
        for mode in ["raw_logprobs", "processed_logits", "processed_logprobs"]:
            sampler = ExperimentalCDFTopKTopPSampler(mode)
            tokens, scores = sampler.forward_native(torch.empty(0, 10), {}, None, torch.empty(0))
            self.assertEqual(tuple(tokens.shape), (0,))
            if scores is not None:
                self.assertEqual(tuple(scores.shape), (0, 10))

    def test_empirical_distribution(self):
        torch.manual_seed(810)
        weights = torch.tensor([[0.0, 0.125, 0.25, 0.625, 0.0]]).expand(100000, -1)
        tokens = cpu_cdf_sample(weights, {})
        observed = torch.bincount(tokens, minlength=5).double() / 100000
        self.assertTrue((observed - weights[0]).abs().max() < 0.005)
        self.assertEqual(int(((tokens == 0) | (tokens == 4)).sum()), 0)


if __name__ == "__main__":
    assert not any(n == "ttnn" or n.startswith("ttnn.") for n in sys.modules)
    unittest.main(verbosity=2)
