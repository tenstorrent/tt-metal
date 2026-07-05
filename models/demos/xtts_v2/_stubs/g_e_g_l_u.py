# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `g_e_g_l_u` of coqui/XTTS-v2.

Reference submodule: `gpt.conditioning_perceiver.layers.0.1.1`, an instance of
`TTS.tts.layers.tortoise.transformer.GEGLU`:

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

No learned parameters — the last dim is split in half and the first half is
gated by the (exact/erf) GELU of the second half. Captured shapes: in
`[1, 32, 5460]`, out `[1, 32, 2730]`.
"""

from __future__ import annotations

import ttnn


def _geglu(x):
    half = x.shape[-1] // 2
    rank = len(x.shape)

    # Split the last dim in half. Do the slice in row-major so a non-tile-
    # aligned half (here 2730) is handled cleanly, then return to tile for gelu.
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    starts_a = [0] * rank
    ends_a = list(x.shape)
    ends_a[-1] = half
    starts_b = [0] * rank
    starts_b[-1] = half
    ends_b = list(x.shape)
    steps = [1] * rank

    a = ttnn.slice(x_rm, starts_a, ends_a, steps)
    gates = ttnn.slice(x_rm, starts_b, ends_b, steps)

    a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
    gates = ttnn.to_layout(gates, ttnn.TILE_LAYOUT)

    # F.gelu defaults to the exact (erf) formulation; match it (not the tanh
    # approximation) so PCC stays tight.
    return ttnn.multiply(a, ttnn.gelu(gates, fast_and_approximate_mode=False))


def build(device, torch_module):
    """GEGLU has no parameters; `torch_module` is unused."""

    def forward(x, *args, **kwargs):
        return _geglu(x)

    return forward


def g_e_g_l_u(x, *args, **kwargs):
    """Bare module-level callable (no parameters)."""
    return _geglu(x)
