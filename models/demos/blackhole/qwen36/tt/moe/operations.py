# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert activation for Qwen3.5-MoE: SwiGLU = silu(gate) * up (hidden_act=silu)."""

import ttnn


def apply_swiglu(gate, up):
    """SwiGLU activation: silu(gate) * up."""
    activated = ttnn.silu(gate)
    return ttnn.mul(activated, up)
