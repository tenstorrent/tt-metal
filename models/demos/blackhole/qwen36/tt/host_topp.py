# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_JOIN_TOPP_CANDIDATES: host top-p sampling without the full-vocabulary sort.

The vLLM TT plugin samples a prefill's first token (every join) on the host. Its sampler
(vllm_tt_plugin.host_sampler.TTTopKTopPSampler) has a top-k candidate path, but a request that disables top-k
(top_k = vocab size) falls through to vLLM's apply_top_k_top_p, which sorts all 248K logits whenever top_p is set:
~21 ms per join on the pod CPU. This module gives the sampler two more candidate paths, for unseeded requests whose
rows all have top-k disabled:

* every row's top_p >= 1: no filter at all. vLLM's sort would mask only tokens whose probability already underflows to
  0, so the candidates are the whole vocabulary (identity ids): the same probabilities, the same noise draw, the same
  token, bit for bit.
* every row's 0 < top_p < 1: the top-M logits (topk, M = QWEN36_JOIN_TOPP_M, default 1024) with probabilities taken
  against the FULL vocabulary (fp64 logsumexp). vLLM keeps a token iff the probability mass ranked above it is < top_p;
  when the cut lies inside the M candidates, more than QWEN36_JOIN_TOPP_MARGIN (default 1e-4; vLLM's own fp32 cumsum is
  good to ~1e-5) from top_p on both sides, and not on a tied logit, the kept set is vLLM's and the candidates carry it
  (dropped ones at -inf); the plugin then samples them with its own softmax + exponential noise. Same distribution,
  different noise draw (as the plugin's top-k path). Any other case returns None: the full sort runs as today.

The plugin is not part of this tree, so the paths are installed into its sampler class at model init
(install_plugin_topp_candidates, idempotent, only with the flag on): the plugin's own top-k path runs first, these only
when it declines.
"""

import os

import torch

TOPP_M_ENV = "QWEN36_JOIN_TOPP_M"
TOPP_MARGIN_ENV = "QWEN36_JOIN_TOPP_MARGIN"


def _m():
    return max(1, int(os.environ.get(TOPP_M_ENV, "1024")))


def _margin():
    return float(os.environ.get(TOPP_MARGIN_ENV, "1e-4"))


def topp_candidates(logits, k, p, m=None, margin=None):
    """(candidate logits [B, C], candidate ids [B, C]) that sample like vLLM's apply_top_k_top_p(logits, k, p) followed
    by softmax, for a batch whose rows all have top-k disabled; None when not applicable (the caller runs the full path).
    ``logits`` [B, V] float CPU (not modified); ``k`` None or [B] (disabled = V); ``p`` None or [B]."""
    if not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or logits.dim() != 2 or logits.shape[0] == 0:
        return None
    B, V = logits.shape
    if k is not None and bool((torch.as_tensor(k) < V).any()):
        return None  # a row with top-k: the plugin's top-k path or the full path
    if p is None:
        return None  # vLLM returns early (no sort) when neither filter is set
    pf = torch.as_tensor(p).to(torch.float64).reshape(-1)
    if pf.numel() != B:
        return None
    if bool((pf >= 1.0).all()):
        # No filter: the whole vocabulary in its own order (a view, no copy); the plugin's softmax / noise / gather then
        # do exactly what the full path does after its no-op mask.
        return logits, torch.arange(V).expand(B, V)
    if bool((pf >= 1.0).any()) or bool((pf <= 0.0).any()) or bool(torch.isnan(pf).any()):
        return None
    M = min(V, int(m) if m is not None else _m())
    if M < 2:
        return None
    mg = float(margin) if margin is not None else _margin()
    vals, ids = logits.topk(M, dim=-1)  # descending
    lse = torch.logsumexp(logits.to(torch.float64), dim=-1, keepdim=True)
    if not bool(torch.isfinite(lse).all()):
        return None
    pr = (vals.to(torch.float64) - lse).exp()  # full-vocabulary probabilities of the candidates
    above = torch.cumsum(pr, dim=-1) - pr  # mass ranked strictly above each candidate
    keep = above < pf.unsqueeze(1)  # vLLM: masked iff cumsum(ascending) <= 1 - p, i.e. mass above >= p
    n = keep.sum(dim=-1)  # kept count; >= 1 (above[0] = 0 < p)
    if bool((n >= M).any()):
        return None  # the cut is not inside the candidates
    last = (n - 1).unsqueeze(1)
    a_last = above.gather(1, last).squeeze(1)  # < p
    a_next = above.gather(1, last + 1).squeeze(1)  # >= p
    if bool(((pf - a_last) <= mg).any()) or bool(((a_next - pf) <= mg).any()):
        return None  # too close to p to be sure vLLM's fp32 arithmetic cuts at the same token
    if bool((vals.gather(1, last) == vals.gather(1, last + 1)).any()):
        return None  # a tie straddles the cut: vLLM's sort order decides which one stays
    return vals.masked_fill(~keep, float("-inf")), ids


_INSTALLED = False


def install_plugin_topp_candidates():
    """Wrap vllm_tt_plugin's TTTopKTopPSampler._top_k_candidates: its own top-k candidates first, then topp_candidates
    (skipped for processed-logprobs modes, which need the full filtered logits). Returns True when installed."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from vllm.config.model import PROCESSED_LOGPROBS_MODES
        from vllm_tt_plugin import host_sampler as hs
    except Exception:  # noqa: BLE001 - no plugin (tests, the demo runners): nothing to install
        return False
    cls = getattr(hs, "TTTopKTopPSampler", None)
    orig = getattr(cls, "_top_k_candidates", None)
    if cls is None or orig is None:
        return False

    def _top_k_candidates(self, logits, k, p):
        got = orig(self, logits, k, p)
        if got is not None or self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
            return got
        return topp_candidates(logits, k, p)

    _top_k_candidates._qwen36_orig = orig
    cls._top_k_candidates = _top_k_candidates
    _INSTALLED = True
    return True
