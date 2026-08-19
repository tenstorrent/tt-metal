# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Nested tracy signposts that bracket a sub-region of a decoder layer.

WHY
---
The single-layer profilers signpost the whole ``layer.forward``, but a decoder layer is only ~1/3
MLP -- MEASURED at seq 2048, T3K TP=8, 27B: 8,157us total, of which the MLP block is 2,667us (32.7%)
and GDN + attention_norm is the other 67.3%. So a whole-layer number is dominated by the half you
are not working on, and the MLP's own op percentages are diluted to the point of being misleading.

Extract just the MLP with::

    tt-perf-report --start-signpost mlp_start --end-signpost mlp_stop <ops_perf_results_*.csv>

(``tt-perf-report`` anchors on the LAST signpost by default and reads past ``stop``, which is empty,
so it prints "No device operations found" unless you pass an explicit range. ``--print-signposts``
lists what a capture actually has.)
"""

from __future__ import annotations


def install_attention_signposts(layer):
    """Bracket the attention half of ``layer`` with ``attn_start`` / ``attn_stop``.

    On a ``linear_attention`` layer that region IS the Gated DeltaNet block: attention_norm (its
    pre/post distributed-norm ops and the full-width all-gather), the fused in-projection, the
    conv1d + FIR, the chunked delta-rule prep/scan, the out-projection and its reduce-scatter. On a
    ``full_attention`` layer it is QKV + RoPE + paged SDPA + wo instead. On a vision block
    (``tt/vision/vision_block.py``) it is the LayerNorm + all-gather, QKV + RoPE, non-causal SDPA
    over the whole padded sequence, and wo + reduce-scatter. Everything except the trailing residual
    add, which layer.forward does after the call returns.

    Extract it with::

        tt-perf-report --start-signpost attn_start --end-signpost attn_stop <csv>

    WHY THE ENTRY POINTS ARE A LIST: unlike the MLP, the attention modules have no single
    ``forward``. ``TPGatedDeltaNet`` defines neither ``forward`` nor ``__call__``; layer.forward calls
    ``forward_prefill`` / ``forward_prefill_collect`` / ``forward_decode`` (and the batched variant)
    directly, and the single-device classes do use ``forward``. So every entry point that exists gets
    wrapped. Outer ones can delegate to inner ones, so a depth counter makes only the OUTERMOST call
    emit ``attn_stop`` -- otherwise a nested call closes the region early and the report silently
    covers a fraction of the block.

    Returns a callable that restores the original methods.
    """
    from tracy import signpost

    orig_norm = layer.attention_norm.forward
    depth = {"n": 0}

    def norm_wrapped(*a, **kw):
        signpost("attn_start")
        return orig_norm(*a, **kw)

    layer.attention_norm.forward = norm_wrapped

    patched = []
    for name in (
        "forward",
        "forward_prefill",
        "forward_prefill_paged",
        "forward_prefill_collect",
        "forward_prefill_batched",
        "forward_decode",
    ):
        fn = getattr(layer.attention, name, None)
        if fn is None or not callable(fn):
            continue

        def make(fn=fn):
            def wrapped(*a, **kw):
                depth["n"] += 1
                try:
                    return fn(*a, **kw)
                finally:
                    depth["n"] -= 1
                    if depth["n"] == 0:
                        signpost("attn_stop")

            return wrapped

        setattr(layer.attention, name, make())
        patched.append((name, fn))

    def restore():
        layer.attention_norm.forward = orig_norm
        for name, fn in patched:
            setattr(layer.attention, name, fn)

    return restore


def install_mlp_signposts(layer):
    """Bracket the MLP sub-region of ``layer`` with ``mlp_start`` / ``mlp_stop``.

    Wraps the two instance methods that bound the MLP -- the pre-MLP norm's ``forward`` and
    ``feed_forward.forward`` -- rather than duplicating layer.forward's body in a test, so this
    cannot drift when that body changes. ``LightweightModule.__call__`` delegates to ``self.forward``
    and layer.forward calls ``feed_forward.forward`` directly, so patching the instance attribute
    catches both call styles. The wrappers only emit signposts; they do not touch the tensors.

    The pre-MLP norm is ``ffn_norm`` on a text decoder layer but ``ff_norm`` on a vision block, so it
    is resolved by name rather than assumed and one helper serves both profilers.

    WHAT IS INSIDE THE BRACKET: ff_norm (pre-AG stats, stats all-gather, post-AG, and in 27B prefill
    the bf8 typecast + the full-width all-gather) then gate/up, the SwiGLU multiply, down, and the
    MLP's reduce-scatter. That is the "MLP block" the tt/mlp.py measurements refer to.

    WHAT IS NOT: the trailing residual add (``h + ff_output``), which happens in layer.forward after
    feed_forward returns -- ~42us at seq 2048, TP=8. Add it back when comparing against a number
    that included it.

    Returns a callable that restores the original methods.
    """
    from tracy import signpost

    norm = getattr(layer, "ffn_norm", None)
    if norm is None:
        norm = getattr(layer, "ff_norm", None)
    if norm is None:
        raise AttributeError(f"{type(layer).__name__} has neither `ffn_norm` nor `ff_norm`")

    orig_norm, orig_mlp = norm.forward, layer.feed_forward.forward

    def norm_wrapped(*a, **kw):
        signpost("mlp_start")
        return orig_norm(*a, **kw)

    def mlp_wrapped(*a, **kw):
        out = orig_mlp(*a, **kw)
        signpost("mlp_stop")
        return out

    norm.forward, layer.feed_forward.forward = norm_wrapped, mlp_wrapped

    def restore():
        norm.forward, layer.feed_forward.forward = orig_norm, orig_mlp

    return restore
