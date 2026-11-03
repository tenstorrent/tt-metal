# SPDX-License-Identifier: Apache-2.0
"""Experimental CPU categorical draw, not installed into the live plugin.

Keep upstream filters and logprob handling. Two replacement draws deliberately
select PyTorch's CPU multinomial kernel; only the first draw is returned. Its
categorical marginal is unchanged. RNG advancement and seeded sequences change.
"""
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p, random_sample


def cpu_multinomial_sample(probs, generators, use_fp64_gumbel=False):
    if probs.device.type != "cpu" or use_fp64_gumbel:
        return random_sample(probs, generators, use_fp64_gumbel)
    batch = probs.shape[0]
    if batch == 0:
        return torch.empty((0,), dtype=torch.long, device=probs.device)
    totals = probs.sum(-1)
    if not (torch.isfinite(totals).all() and (totals > 0).all() and (probs >= 0).all()):
        # Preserve upstream behavior for NaNs, infinities, zero/negative rows.
        # No random numbers have been consumed before this fallback.
        return random_sample(probs, generators, use_fp64_gumbel)
    if len(generators) != batch:
        tokens = torch.multinomial(probs, num_samples=2, replacement=True)[:, 0].contiguous()
    else:
        tokens = torch.empty((batch,), dtype=torch.long)
    for row, generator in generators.items():
        tokens[row] = torch.multinomial(probs[row], num_samples=2, replacement=True, generator=generator)[0]
    return tokens


class ExperimentalMultinomialTopKTopPSampler(TopKTopPSampler):
    def __init__(self, logprobs_mode="raw_logprobs", use_fp64_gumbel=False):
        super().__init__(logprobs_mode, use_fp64_gumbel)
        self.forward = self.forward_native

    def forward_native(self, logits, generators, k, p):
        logits = apply_top_k_top_p(logits, k, p)
        scores = None
        if self.logprobs_mode == "processed_logits":
            scores = logits
        elif self.logprobs_mode == "processed_logprobs":
            scores = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return cpu_multinomial_sample(probs, generators, self.use_fp64_gumbel), scores
