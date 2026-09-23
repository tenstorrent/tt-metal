# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Expert activation (SwiGLU)."""

import ttnn


def apply_swiglu(up_gate):
    """SwiGLU over concatenated [up | gate]: up * silu(gate)."""
    return ttnn.swiglu(up_gate, dim=-1)
