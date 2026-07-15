# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert activation for Qwen3.5-MoE: SwiGLU = silu(gate) * up (hidden_act=silu)."""

import ttnn


def apply_swiglu(up_gate):
    """SwiGLU activation over concatenated [up | gate]: up * silu(gate)."""
    return ttnn.swiglu(up_gate, dim=-1)
