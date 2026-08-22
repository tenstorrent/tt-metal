# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `feed_forward_network` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.prediction_head.layers.0.ffn`, a
`vibevoice.modular.modular_vibevoice_diffusion_head.FeedForwardNetwork` — a
SwiGLU FFN (no biases):

    gate = SiLU(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs import _precision as prec

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained SwiGLU-FFN weights and return a native ttnn forward closure."""
    m = torch_module
    gate_w = m.gate_proj.weight.detach().float()  # [ffn_dim, dim]
    up_w = m.up_proj.weight.detach().float()  # [ffn_dim, dim]
    down_w = m.down_proj.weight.detach().float()  # [dim, ffn_dim]

    gate_proj_w = prec.mm_weight(gate_w.t().contiguous(), device)
    up_proj_w = prec.mm_weight(up_w.t().contiguous(), device)
    down_proj_w = prec.mm_weight(down_w.t().contiguous(), device)

    compute_config = prec.compute_config(device)

    def forward(x, *args, **kwargs):
        gate = prec.matmul(x, gate_proj_w, compute_config)
        gate = ttnn.silu(gate, memory_config=_DRAM)
        up = prec.matmul(x, up_proj_w, compute_config)
        h = ttnn.mul(gate, up, memory_config=_DRAM)
        return prec.matmul(h, down_proj_w, compute_config)

    return forward


def feed_forward_network(*args, **kwargs):
    raise RuntimeError(
        "feed_forward_network requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
