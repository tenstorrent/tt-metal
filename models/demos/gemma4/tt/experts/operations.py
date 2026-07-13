# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared expert operations for Gemma4.

GeGLU activation: gelu(gate) * up (different from GPT-OSS SwiGLU).
"""

import ttnn


def apply_geglu(gate, up):
    """GeGLU activation: gelu(gate) * up.

    ``gate`` must already carry GELU (fused into the gate sparse_matmul / linear
    epilogue). This helper only performs the elementwise multiply with ``up``.
    """
    return ttnn.mul(gate, up)
