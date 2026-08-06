# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared expert operations for Gemma4.

GeGLU activation: gelu(gate) * up (different from GPT-OSS SwiGLU).
"""

import ttnn


def apply_geglu(gate, up):
    """GeGLU activation: gelu(gate) * up. Gemma4 uses gelu_pytorch_tanh.

    Fused into one BinaryNg mul with lhs GELU (param 1.0 == fast tanh approx).
    """
    return ttnn.mul(
        gate,
        up,
        input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)],
    )
