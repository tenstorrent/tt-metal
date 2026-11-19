# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic B1 mesh routing using the original optimized decoder's policy.

This deliberately rounds the 256 FP32 router logits to BF16 BEFORE selection.
The inherited InlineSelector then uses its original BF16 comparison, partition
tie ordering, selected-value FP32 softmax, and BF16 score output. This is not an
exact-FP32 selector and is not a promoted precision policy or gate waiver.
The experiment tests whether matching the baseline's router boundary explains
whole-output disagreement. It does not replace the mesh router projection.

Installation changes one decoder's function globals only. Multirow calls keep
native FP32 topk/softmax. The selector's setup-owned tensors stay referenced by
the decoder through all live traces; this module allocates no device tensors
at setup. Runtime operations are device-only, with no host read or conversion.
"""

import types

import ttnn


class _BaselineRoutingProxy:
    def __init__(self, selector):
        self.selector = selector
        self.pending_scores = None

    def __getattr__(self, name):
        return getattr(ttnn, name)

    def topk(self, logits, *args, **kwargs):
        if logits.shape[-2] != 1:
            return ttnn.topk(logits, *args, **kwargs)
        if args or kwargs != {"k": 8, "dim": -1, "sorted": True}:
            raise ValueError("baseline router probe requires the exact sorted 256-to-8 call")
        if tuple(logits.shape) != (1, 1, 1, 256) or logits.dtype != ttnn.float32:
            raise ValueError("baseline router probe requires singleton FP32 logits [1,1,1,256]")
        if self.pending_scores is not None:
            raise RuntimeError("baseline router selector scores were not consumed")
        rounded = ttnn.reshape(ttnn.typecast(logits, ttnn.bfloat16), (1, 256))
        routing, ids = self.selector(rounded)
        scores = ttnn.reshape(routing, (1, 1, 1, 8))
        self.pending_scores = scores
        return scores, ids

    def softmax(self, values, *args, **kwargs):
        if self.pending_scores is not None and values is self.pending_scores:
            if args or kwargs.get("dim") != -1 or kwargs.get("numeric_stable") is not True:
                raise ValueError("baseline router probe requires the original stable score softmax call")
            if set(kwargs) != {"dim", "numeric_stable", "compute_kernel_config"}:
                raise ValueError("baseline router probe encountered an unexpected softmax option")
            self.pending_scores = None
            # InlineSelector already computed these normalized BF16 scores.
            # The ordinary MoE cast/layout tail is retained by the cloned body.
            return values
        return ttnn.softmax(values, *args, **kwargs)


def install(dec):
    """Install during setup; return the previous bound _moe for explicit A/B."""
    if dec.kind != "linear_attention" or not hasattr(dec, "inline_selector"):
        raise ValueError("baseline router diagnostic requires the inherited linear-attention InlineSelector")
    if getattr(dec, "_baseline_router_probe_installed", False):
        raise ValueError("baseline router diagnostic is already installed")
    original = dec._moe
    source = original.__func__
    if not {"topk", "softmax"}.issubset(source.__code__.co_names):
        raise ValueError("baseline router diagnostic requires the direct MoE body with topk/softmax consumers")
    proxy = _BaselineRoutingProxy(dec.inline_selector)
    function = types.FunctionType(
        source.__code__, dict(source.__globals__, ttnn=proxy), source.__name__, source.__defaults__, source.__closure__
    )
    function.__kwdefaults__ = source.__kwdefaults__

    def routed(self, x, *, ordered_router=False, batched=False):
        if proxy.pending_scores is not None:
            raise RuntimeError("baseline router diagnostic does not support concurrent forwards")
        try:
            result = function(self, x, ordered_router=ordered_router, batched=batched)
            if proxy.pending_scores is not None:
                raise RuntimeError("baseline router selector scores bypassed the expected softmax consumer")
            return result
        finally:
            # Never retain a per-forward tensor after success or failure; this
            # is host reference cleanup, not a device allocation or transfer.
            proxy.pending_scores = None

    dec._moe = types.MethodType(routed, dec)
    dec._baseline_router_probe_installed = True
    dec.precision_policy["experimental_baseline_router"] = {
        "scope": "singleton linear-attention MoE only",
        "projection": "unchanged mesh router; FP32 output rounded to BF16 at selection boundary",
        "selection": "inherited InlineSelector BF16 comparison and original partition tie ordering",
        "scores": "inherited selected-value FP32 softmax rounded to BF16; no second softmax",
        "expert_count": 8,
        "status": "diagnostic only; full-decoder PCC gates unchanged",
    }
    return original
