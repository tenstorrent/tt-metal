"""Focused CPU-only equivalence checks for the optional K2 host sampler."""

import unittest

import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample
from vllm.v1.sample.sampler import Sampler

from models.demos.k2_horizon_7b_qb2.tt.host_sampling import K2HostSampler


def make_generators(rows, *, same_seed=False):
    return {row: torch.Generator().manual_seed(1234 if same_seed else 1234 + row) for row in range(rows)}


def metadata(rows, vocab, *, same_seed=False, max_num_logprobs=None):
    return SamplingMetadata(
        temperature=torch.ones(rows),
        all_greedy=False,
        all_random=True,
        top_p=torch.full((rows,), 0.95),
        top_k=torch.full((rows,), vocab),
        generators=make_generators(rows, same_seed=same_seed),
        max_num_logprobs=max_num_logprobs,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(rows),
        presence_penalties=torch.zeros(rows),
        repetition_penalties=torch.ones(rows),
        output_token_ids=[[] for _ in range(rows)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


class HostSamplerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def compare(self, logits, left, right, mode="raw_logprobs", steps=3, **kwargs):
        upstream = Sampler(mode)
        generated = K2HostSampler(mode)
        try:
            with torch.inference_mode():
                for _ in range(steps):
                    torch.manual_seed(932)
                    expected = upstream(logits.clone(), left, **kwargs)
                    state = torch.get_rng_state()
                    torch.manual_seed(932)
                    actual = generated(logits.clone(), right, **kwargs)
                    self.assertTrue(torch.equal(expected.sampled_token_ids, actual.sampled_token_ids))
                    self.assertTrue(torch.equal(state, torch.get_rng_state()))
                    self.assertEqual(expected.logprobs_tensors is None, actual.logprobs_tensors is None)
                    if expected.logprobs_tensors is not None:
                        for a, b in zip(expected.logprobs_tensors, actual.logprobs_tensors):
                            if isinstance(a, torch.Tensor):
                                self.assertTrue(torch.equal(a, b))
                            else:
                                self.assertEqual(a, b)
                    for index in left.generators:
                        self.assertTrue(
                            torch.equal(left.generators[index].get_state(), right.generators[index].get_state())
                        )
        finally:
            generated.close()
        return generated

    def test_full_vocabulary_seeded_streams_and_inference_mode(self):
        torch.manual_seed(77)
        logits = (torch.randn(32, 250624) * 6).bfloat16()
        for same_seed in (False, True):
            with self.subTest(same_seed=same_seed):
                self.compare(
                    logits, metadata(32, 250624, same_seed=same_seed), metadata(32, 250624, same_seed=same_seed)
                )

    def test_logprob_modes(self):
        torch.manual_seed(91)
        logits = torch.randn(4, 1024).bfloat16()
        for mode in ("raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits"):
            with self.subTest(mode=mode):
                self.compare(
                    logits, metadata(4, 1024, max_num_logprobs=-1), metadata(4, 1024, max_num_logprobs=-1), mode=mode
                )

    def test_mixed_greedy_random_and_top_k(self):
        torch.manual_seed(92)
        logits = torch.randn(4, 512).bfloat16()
        left, right = metadata(4, 512), metadata(4, 512)
        for item in (left, right):
            item.all_random = False
            item.temperature = torch.tensor([0.0, 0.7, 0.0, 1.3])
            item.top_k = torch.tensor([1, 20, 1, 512])
            item.top_p = torch.tensor([1.0, 0.9, 1.0, 0.95])
        self.compare(logits, left, right)

    def test_all_greedy_does_not_create_pool(self):
        left, right = metadata(4, 512), metadata(4, 512)
        for item in (left, right):
            item.all_random = False
            item.all_greedy = True
            item.temperature = None
        sampler = self.compare(torch.randn(4, 512), left, right)
        self.assertIsNone(sampler.topk_topp_sampler._pool)

    def test_absent_partial_aliased_and_single_row_fallback(self):
        for kind in ("absent", "partial", "aliased", "single"):
            with self.subTest(kind=kind):
                rows = 1 if kind == "single" else 4
                left, right = metadata(rows, 512), metadata(rows, 512)
                if kind == "absent":
                    left.generators, right.generators = {}, {}
                elif kind == "partial":
                    left.generators, right.generators = make_generators(2), make_generators(2)
                elif kind == "aliased":
                    left.generators = dict.fromkeys(range(rows), torch.Generator().manual_seed(42))
                    right.generators = dict.fromkeys(range(rows), torch.Generator().manual_seed(42))
                sampler = self.compare(torch.randn(rows, 512), left, right)
                self.assertEqual(sampler.stats_snapshot()["parallel_rng_calls"], 0)
                self.assertEqual(sampler.stats_snapshot()["stock_rng_calls"], 3)

    def test_filter_exact_bits_guards_ties_and_source_ownership(self):
        torch.manual_seed(721)
        base = torch.randn(4, 512)
        cases = {
            "float32": base,
            "bfloat16_values": base.bfloat16().float(),
            "temperature_scaled": base / 0.7,
            "all_zero": torch.zeros_like(base),
            "negative_zero": torch.full_like(base, -0.0),
            "tied_positive": torch.ones_like(base),
            "tied_negative": -torch.ones_like(base),
            "noncontiguous": base.t().contiguous().t(),
            "bfloat16_dtype": base.bfloat16(),
            "float64_dtype": base.double(),
        }
        mixed = torch.zeros_like(base)
        mixed[:, ::2] = -0.0
        cases["mixed_zero"] = mixed
        for name, value in (("nan", float("nan")), ("inf", float("inf")), ("negative_inf", -float("inf"))):
            cases[name] = base.clone()
            cases[name][:, 7] = value
        sampler = K2HostSampler()
        try:
            for name, logits in cases.items():
                for k in (None, torch.full((4,), 512), torch.full((4,), 20)):
                    for p in (None, torch.tensor([0.95, 0.01, 1.0, 0.5])):
                        with self.subTest(case=name, k=k, p=p):
                            a, b = logits.clone(), logits.clone()
                            expected = apply_top_k_top_p(a, k, p)
                            actual = sampler.topk_topp_sampler.apply_top_k_top_p(b, k, p)
                            self.assertTrue(
                                torch.equal(
                                    expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)
                                )
                            )
                            # Match both returned values and upstream input-buffer mutations.
                            self.assertTrue(
                                torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                            )
            self.assertGreater(sampler.stats_snapshot()["tie_fallback_rows"], 0)
            self.assertGreater(sampler.stats_snapshot()["stock_sort_calls"], 0)
        finally:
            sampler.close()

    def test_invalid_generator_mapping_uses_stock_error_path(self):
        sampler = K2HostSampler()
        try:
            for generators in ({4: torch.Generator()}, {0: "invalid"}):
                with self.subTest(generators=generators):
                    torch.manual_seed(98)
                    with self.assertRaises(Exception) as expected:
                        random_sample(torch.ones(4, 32), generators)
                    expected_state = torch.get_rng_state()
                    torch.manual_seed(98)
                    with self.assertRaises(type(expected.exception)):
                        sampler.topk_topp_sampler._random_sample(torch.ones(4, 32), generators)
                    self.assertTrue(torch.equal(expected_state, torch.get_rng_state()))
                    self.assertIsNone(sampler.topk_topp_sampler._pool)
        finally:
            sampler.close()

    def test_top_p_cutoff_boundaries_match_upstream(self):
        torch.manual_seed(381)
        logits = torch.randn(4, 129)
        logits[0] *= 1000  # Exact-zero softmax tails remain in the cumulative sum.
        logits[1].zero_()  # A cutoff may split equal-valued vocabulary entries.
        cumulative = logits.sort(dim=-1).values.softmax(dim=-1).cumsum(dim=-1)
        boundary = 1 - cumulative[:, 64]
        probabilities = (
            boundary,
            torch.nextafter(boundary, torch.zeros_like(boundary)),
            torch.nextafter(boundary, torch.ones_like(boundary)),
            torch.tensor([0.0, 1.0, -0.1, 1.1]),
            torch.tensor([float("nan"), float("inf"), -float("inf"), 0.95]),
        )
        sampler = K2HostSampler()
        try:
            for p in probabilities:
                with self.subTest(p=p.tolist()):
                    expected = apply_top_k_top_p(logits.clone(), None, p)
                    actual = sampler.topk_topp_sampler.apply_top_k_top_p(logits.clone(), None, p)
                    self.assertTrue(torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)))
        finally:
            sampler.close()

    def test_pool_is_lazy_and_close_is_idempotent(self):
        sampler = K2HostSampler()
        self.assertIsNone(sampler.topk_topp_sampler._pool)
        sampler(torch.randn(4, 32), metadata(4, 32))
        pool = sampler.topk_topp_sampler._pool
        self.assertIsNotNone(pool)
        sampler.close()
        sampler.close()
        self.assertTrue(pool._shutdown)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            sampler(torch.randn(4, 32), metadata(4, 32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
