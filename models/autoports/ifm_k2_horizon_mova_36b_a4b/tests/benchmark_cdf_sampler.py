# SPDX-License-Identifier: Apache-2.0
"""Experimental CPU categorical sampler; not installed into the live plugin.

Filtering/logprob calculation stay upstream. CDF bins must exactly equal the
float32 probability weights in double precision, otherwise use upstream's
exponential race. The finite RNG algorithm and stream advancement change.
"""
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p, random_sample


def exact_cdf_or_none(probs):
    """Return cumulative weights only when no rounding changes interval widths."""
    weights = probs.double()
    cumulative = weights.cumsum(-1)
    if cumulative.shape[0] == 0:
        return cumulative
    widths = torch.diff(cumulative, dim=-1, prepend=torch.zeros_like(cumulative[:, :1]))
    if (
        not torch.isfinite(cumulative[:, -1]).all()
        or not (cumulative[:, -1] > 0).all()
        or not (weights >= 0).all()
        or not torch.equal(widths, weights)
    ):
        return None
    return cumulative


def cpu_cdf_sample(probs, generators, use_fp64_gumbel=False):
    """Categorical sampling with original generator ownership, new RNG draws.

    In real arithmetic each output has probability w_i / sum(w), as in the
    exponential race. Discrete float64 uniforms quantize interval probabilities;
    seed streams and tokens need not match the original algorithm. No state is
    stored on this helper. Fallback occurs before any random draws.
    """
    if probs.device.type != "cpu" or use_fp64_gumbel:
        return random_sample(probs, generators, use_fp64_gumbel)
    batch, vocab = probs.shape
    if batch == 0:
        return torch.empty((0,), dtype=torch.long, device=probs.device)
    cumulative = exact_cdf_or_none(probs)
    if cumulative is None:
        return random_sample(probs, generators, use_fp64_gumbel)
    if len(generators) != batch:
        uniform = torch.rand((batch, 1), dtype=torch.float64)
    else:
        uniform = torch.empty((batch, 1), dtype=torch.float64)
    for row, generator in generators.items():
        uniform[row] = torch.rand((1,), dtype=torch.float64, generator=generator)
    total = cumulative[:, -1:]
    # Prevent a rounded product equal to total from indexing past vocabulary.
    target = torch.minimum(uniform * total, torch.nextafter(total, torch.zeros_like(total)))
    return torch.searchsorted(cumulative, target, right=True).flatten()


class ExperimentalCDFTopKTopPSampler(TopKTopPSampler):
    """Same filtering/score contract; CPU inverse-CDF categorical draw only."""

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
        return cpu_cdf_sample(probs, generators, self.use_fp64_gumbel), scores
