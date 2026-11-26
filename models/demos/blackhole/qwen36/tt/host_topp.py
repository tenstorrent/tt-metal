# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host top-p sampling of unseeded rows from their largest logits instead of the whole vocabulary.

vllm-tt-plugin samples the first token of every prefill on the host, with its TTTopKTopPSampler. For a request with
top_k disabled and 0 < top_p < 1, vLLM's apply_top_k_top_p sorts all 248320 logits of the row. topp_candidates finds
vLLM's kept set among the row's TOPP_CANDIDATES largest logits when it can prove the set is the same, and
install_plugin_topp_candidates lets the plugin's sampler use it: the sampled distribution is vLLM's (the noise draw
differs, as in the plugin's top-k candidate path).
"""

import sys

import torch
from loguru import logger

# Candidates read per row; vLLM's top-p kept set must end inside them.
TOPP_CANDIDATES = 1024
# vLLM decides the cut with a float32 cumulative sum over the sorted vocabulary, topp_candidates with float64 sums over
# the candidates. A cut whose probability mass is this close to top_p goes to the full path, so that the two cut at
# the same token.
TOPP_MARGIN = 1e-4


def topp_candidates(logits, k, p):
    """Candidates that sample like vLLM's apply_top_k_top_p(logits, k, p) followed by softmax, for a batch whose rows
    all have top_k disabled and 0 < top_p < 1.

    Parameters
    ----------
    logits : torch.Tensor
        Host logits after temperature, float ``[B, V]``. Not modified.
    k : torch.Tensor or None
        Per-row top_k ``[B]`` (disabled: ``V``), or None when no row uses top_k.
    p : torch.Tensor or None
        Per-row top_p ``[B]``, or None when no row uses top_p.

    Returns
    -------
    tuple of torch.Tensor or None
        ``(values, ids)``, both ``[B, C]`` with ``C = min(V, TOPP_CANDIDATES)``: each row's largest logits in
        descending order with the tokens that vLLM removes set to ``-inf``, and their token ids. None when the batch
        must take vLLM's full path: a row with top_k, no top_p, a row with top_p outside (0, 1), a cut outside the
        candidates or within TOPP_MARGIN of top_p, or tied logits on both sides of the cut (vLLM's sort order then
        decides which one stays).
    """
    if not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or logits.dim() != 2:
        return None
    batch, vocab = logits.shape
    if batch == 0 or p is None:
        return None
    if k is not None and bool((torch.as_tensor(k) < vocab).any()):
        return None
    top_p = torch.as_tensor(p).to(torch.float64).reshape(-1)
    if top_p.numel() != batch or not bool(((top_p > 0.0) & (top_p < 1.0)).all()):
        return None
    count = min(vocab, TOPP_CANDIDATES)
    if count < 2:
        return None
    values, ids = logits.topk(count, dim=-1)
    lse = torch.logsumexp(logits.to(torch.float64), dim=-1, keepdim=True)
    if not bool(torch.isfinite(lse).all()):
        return None
    probs = (values.to(torch.float64) - lse).exp()  # probabilities over the whole vocabulary
    above = torch.cumsum(probs, dim=-1) - probs  # mass of the candidates ranked above each one
    # vLLM removes a token when the ascending cumulative sum up to it is at most 1 - top_p, which is when the mass
    # ranked above it is at least top_p.
    keep = above < top_p.unsqueeze(1)
    kept = keep.sum(dim=-1)
    if bool((kept >= count).any()):
        return None
    last = (kept - 1).unsqueeze(1)
    if bool((top_p - above.gather(1, last).squeeze(1) <= TOPP_MARGIN).any()):
        return None
    if bool((above.gather(1, last + 1).squeeze(1) - top_p <= TOPP_MARGIN).any()):
        return None
    if bool((values.gather(1, last) == values.gather(1, last + 1)).any()):
        return None
    return values.masked_fill(~keep, float("-inf")), ids


_WRAPPED_ATTR = "_qwen36_topp_wrapped"


def install_plugin_topp_candidates(sampler_cls=None):
    """Let vllm-tt-plugin's host sampler take topp_candidates when its own top-k candidates decline a batch.

    The plugin's TTTopKTopPSampler.forward_native samples an unseeded batch from the candidates that its
    ``_top_k_candidates`` method returns. This wraps that method on the class: the plugin's top-k candidates first,
    then topp_candidates, except under a ``processed_*`` logprobs_mode, whose scores cover the whole vocabulary.
    Nothing is installed when the class has no ``_top_k_candidates`` (the sampler then sorts as before), or when the
    plugin already samples top-p rows from candidates itself (``_sample_top_p_candidates``). Idempotent.

    Parameters
    ----------
    sampler_cls : type or None
        The sampler class to wrap; None imports vllm_tt_plugin.host_sampler.TTTopKTopPSampler.

    Returns
    -------
    bool
        True when the class samples through topp_candidates after the call.
    """
    if sampler_cls is None:
        try:
            from vllm_tt_plugin.host_sampler import TTTopKTopPSampler as sampler_cls
        except ImportError as exc:
            logger.info(f"Host top-p candidates not installed: no vllm-tt-plugin host sampler ({exc})")
            return False
    name = f"{sampler_cls.__module__}.{sampler_cls.__qualname__}"
    orig = getattr(sampler_cls, "_top_k_candidates", None)
    if orig is None:
        # forward_native binds to each sampler instance when it is built, before the model loads, so the candidate
        # method is the only place a class-level change reaches an existing sampler.
        logger.info(f"Host top-p candidates not installed: {name} has no _top_k_candidates")
        return False
    if getattr(orig, _WRAPPED_ATTR, False):
        return True
    module = sys.modules.get(sampler_cls.__module__)
    if hasattr(sampler_cls, "_sample_top_p_candidates") or hasattr(module, "_sample_top_p_candidates"):
        logger.info(f"Host top-p candidates not installed: {name} samples top-p rows from candidates itself")
        return False
    from vllm.config.model import PROCESSED_LOGPROBS_MODES

    def _top_k_candidates(self, logits, k, p):
        got = orig(self, logits, k, p)
        if got is not None or self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
            return got
        return topp_candidates(logits, k, p)

    setattr(_top_k_candidates, _WRAPPED_ATTR, True)
    _top_k_candidates.__wrapped__ = orig
    sampler_cls._top_k_candidates = _top_k_candidates
    logger.info(f"Host top-p candidates installed on {name}")
    return True
