# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared expert operations for Gemma4.

GeGLU activation: gelu(gate) * up (different from GPT-OSS SwiGLU).
"""

import ttnn


def apply_geglu(gate, up):
    """GeGLU activation: gelu(gate) * up.

    Fused into one BinaryNg mul with lhs GELU. Param 0.0 is the Accurate
    variant, matching ``gelu_variant()`` — ttnn maps GeluVariant::ACCURATE to
    ``UnaryWithParam(GELU, 0.0f)`` and FAST_LUT to 1.0f (unary.cpp). Keep 0.0:
    Gemma4 device PCC is gated on Accurate, and the fusion is what saves the op.
    """
    return ttnn.mul(
        gate,
        up,
        input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)],
    )
