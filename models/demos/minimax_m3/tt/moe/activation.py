# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 expert activation: the clamped gpt-oss "swigluoai" SwiGLU.

Used by the dense MLP (dense_mlp.py, layers 0-2) and the shared expert. The MoE routed experts apply
the same clamped swigluoai inside the fused unified_routed_expert_ffn kernel
(RoutedExpertActivation.SwiGluOai), so this Python implementation is the dense/shared path's.
Anchor: transformers gpt_oss modeling_gpt_oss.py lines 119-122.

Two implementations of the same math:

  * ``apply_swiglu``       — the readable 7-op chain.
  * ``apply_swiglu_fused`` — ONE ``ttnn.multiply`` with the whole activation folded into its two
                             per-operand activation spans, collapsing 7 device ops into 1.
"""

import os

from loguru import logger

import ttnn

# Stand-in for "no lower bound" in CLAMP_TSS, which requires both bounds. Far outside any activation
# magnitude, and far inside bf16/fp32 range so it can never itself overflow.
_NO_LOWER_BOUND = -1.0e30

# The fused single-op form is the default: 7 device ops -> 1 on every dense MLP and every shared
# expert (60 calls per chunk, so ~360 fewer op launches). M3_FUSED_SWIGLU=0 restores the chain so a
# numerics question stays bisectable; the two agree to ~2e-6 of PCC (tests/unit/test_swiglu_vs_ref.py).
DEFAULT_USE_FUSED_SWIGLU = True

_swiglu_path_logged = False


def use_fused_swiglu() -> bool:
    """True -> one multiply with activation spans; False -> the readable 7-op chain."""
    v = os.environ.get("M3_FUSED_SWIGLU")
    if v is None:
        return DEFAULT_USE_FUSED_SWIGLU
    return v.strip().lower() in ("1", "true", "yes", "on")


def swiglu(gate, up, config):
    """M3's clamped swigluoai by whichever implementation is selected; logs the choice once.

    CONSUMES both ``gate`` and ``up`` — the caller must treat them as dead and free only the result.
    (The chain writes through its inputs and returns one of them; the fused op returns a fresh tensor,
    so this frees the two inputs itself. Uniform contract either way.)
    """
    global _swiglu_path_logged
    fused = use_fused_swiglu()
    if not _swiglu_path_logged:
        _swiglu_path_logged = True
        logger.info(f"[swiglu] clamped swigluoai via the {'FUSED single multiply' if fused else 'LEGACY 7-op chain'}")
    if not fused:
        return apply_swiglu(gate, up, config)
    out = apply_swiglu_fused(gate, up, config)
    gate.deallocate(True)
    up.deallocate(True)
    return out


def swiglu_activation_spans(config):
    """The (a_activations, b_activations) spans that turn a plain ``multiply(up, gate)`` into M3's
    clamped swigluoai.

    M3:  out = (clamp(up, -L, L) + 1) * (clamp(gate, max=L) * sigmoid(alpha * clamp(gate, max=L)))

    The `a` operand (up) is just clamp-then-+1. The `b` operand (gate) needs ``g * sigmoid(alpha*g)``,
    for which there is no single SFPU op: plain SILU has no alpha, and there is no swish-with-beta
    (HARDSWISH is a different function). Use the identity

        silu(x) = x * sigmoid(x)   =>   silu(alpha*g) = alpha*g * sigmoid(alpha*g)
                                   =>   g * sigmoid(alpha*g) = silu(alpha*g) / alpha

    i.e. scale up by alpha, take SILU, scale back down by 1/alpha.
    """
    limit, alpha = config.swiglu_limit, config.alpha
    up_spans = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.CLAMP_TSS, -limit, limit),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, 1.0),
    ]
    gate_spans = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.CLAMP_TSS, _NO_LOWER_BOUND, limit),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, alpha),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 1.0 / alpha),
    ]
    return up_spans, gate_spans


def apply_swiglu_fused(gate, up, config, memory_config=None):
    """Clamped swigluoai as ONE device op.

    Numerically equivalent to ``apply_swiglu``: measured against an fp32 reference at the real
    per-device shape (640 x 768), fused 0.9999927 vs chain 0.9999949, and fused-vs-chain 0.9999917
    (tests/unit/test_swiglu_vs_ref.py). The fused form is a hair LESS accurate, not more — the SFPU's
    SILU approximation differs slightly from an explicit sigmoid+multiply — but 2.5e-6 of PCC is far
    below anything that matters here.

    Neither input is consumed: the op writes a fresh output. Callers own freeing gate/up.
    """
    up_spans, gate_spans = swiglu_activation_spans(config)
    kwargs = {"input_tensor_a_activations": up_spans, "input_tensor_b_activations": gate_spans}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    return ttnn.multiply(up, gate, **kwargs)


def apply_swiglu(gate, up, config):
    """Clamped swigluoai: ``(up + 1) * (gate * sigmoid(alpha * gate))``, with gate clamped to
    ``max=swiglu_limit`` and up clamped to ``[-swiglu_limit, swiglu_limit]``.

    M3 deltas vs M2's plain SiLU SwiGLU: the gate/up clamp, the ``alpha`` inside the sigmoid, and the
    ``(up + 1)`` linear term. ``config`` is any object exposing ``.swiglu_limit`` and ``.alpha``.
    """
    gate = ttnn.clamp(gate, min=None, max=config.swiglu_limit, output_tensor=gate)
    up = ttnn.clamp(up, min=-config.swiglu_limit, max=config.swiglu_limit, output_tensor=up)

    # glu = gate * sigmoid(alpha * gate)
    gate_alpha = ttnn.mul(gate, config.alpha)
    gate_sigmoid = ttnn.sigmoid(gate_alpha)
    gate_alpha.deallocate(True)
    glu = ttnn.mul(gate, gate_sigmoid, output_tensor=gate)
    gate_sigmoid.deallocate(True)

    # out = (up + 1) * glu
    up = ttnn.add(up, 1, output_tensor=up)
    result = ttnn.mul(up, glu, output_tensor=up)
    ttnn.deallocate(glu)
    return result
