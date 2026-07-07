# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 expert activation: the clamped gpt-oss "swigluoai" SwiGLU.

Used by the dense MLP (dense_mlp.py, layers 0-2). The MoE routed experts now apply the same clamped
swigluoai inside the fused unified_routed_expert_ffn kernel (RoutedExpertActivation.SwiGluOai), so
this Python implementation is the dense path's activation. Anchor: transformers gpt_oss
modeling_gpt_oss.py lines 119-122.
"""

import ttnn


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
