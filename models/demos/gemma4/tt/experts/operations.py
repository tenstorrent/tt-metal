# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared expert operations for Gemma4.

GeGLU activation: gelu(gate) * up (different from GPT-OSS SwiGLU).
"""

import ttnn
from models.demos.gemma4.tt.compute_config import gelu_variant


def apply_geglu(gate, up):
    """GeGLU activation: gelu(gate) * up (Accurate variant; see compute_config)."""
    activated = ttnn.gelu(gate, variant=gelu_variant())
    result = ttnn.mul(activated, up)
    return result
