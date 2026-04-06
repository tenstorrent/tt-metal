# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared expert operations for Gemma4.

GeGLU activation: gelu(gate) * up (different from GPT-OSS SwiGLU).
"""

import ttnn


def apply_geglu(gate, up):
    """GeGLU activation: gelu(gate) * up. Gemma4 uses gelu_pytorch_tanh."""
    activated = ttnn.gelu(gate, fast_and_approximate_mode=True)
    result = ttnn.mul(activated, up)
    return result
