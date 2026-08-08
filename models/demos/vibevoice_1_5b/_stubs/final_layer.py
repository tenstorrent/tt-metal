# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `final_layer` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.prediction_head.final_layer`, a
`vibevoice.modular.modular_vibevoice_diffusion_head.FinalLayer` — the
AdaLN-modulated output projection of the diffusion head:

    shift, scale = adaLN_modulation(c).chunk(2, dim=-1)   # SiLU -> Linear(cond, 2*hidden), no bias
    x = modulate(RMSNorm_no_affine(x), shift, scale)       # x * (1 + scale) + shift
    x = linear(x)                                          # Linear(hidden, out), no bias

`norm_final` is a plain (non-affine, no learned weight) RMSNorm: eps=1e-5.
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs import _precision as prec

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained FinalLayer weights and return a native ttnn forward closure."""
    m = torch_module
    eps = float(m.norm_final.eps)

    linear_w = m.linear.weight.detach().float()  # [out, hidden]
    adaln_w = m.adaLN_modulation[1].weight.detach().float()  # [2*hidden, cond]
    hidden_size = linear_w.shape[1]

    linear_w_t = prec.mm_weight(linear_w.t().contiguous(), device)
    adaln_w_t = prec.mm_weight(adaln_w.t().contiguous(), device)

    compute_config = prec.compute_config(device)

    def _to_ttnn_f32(t):
        if not isinstance(t, ttnn.Tensor):
            return ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.typecast(t, ttnn.float32) if t.get_dtype() != ttnn.float32 else t

    def forward(x, c, *args, **kwargs):
        x = _to_ttnn_f32(x)
        c = _to_ttnn_f32(c)

        c_act = ttnn.silu(c, memory_config=_DRAM)
        ada = prec.matmul(c_act, adaln_w_t, compute_config)
        if ada.get_dtype() != ttnn.float32:
            ada = ttnn.typecast(ada, ttnn.float32)
        shift = ttnn.slice(ada, [0, 0, 0], [ada.shape[0], ada.shape[1], hidden_size])
        scale = ttnn.slice(ada, [0, 0, hidden_size], [ada.shape[0], ada.shape[1], 2 * hidden_size])

        sq = ttnn.mul(x, x, memory_config=_DRAM)
        mean_sq = ttnn.mean(sq, dim=-1, keepdim=True)
        denom = ttnn.rsqrt(ttnn.add(mean_sq, eps), memory_config=_DRAM)
        x_norm = ttnn.mul(x, denom, memory_config=_DRAM)

        one_plus_scale = ttnn.add(scale, 1.0, memory_config=_DRAM)
        modulated = ttnn.add(ttnn.mul(x_norm, one_plus_scale, memory_config=_DRAM), shift, memory_config=_DRAM)

        out = prec.matmul(modulated, linear_w_t, compute_config)
        return ttnn.typecast(out, ttnn.float32) if out.get_dtype() != ttnn.float32 else out

    return forward


def final_layer(*args, **kwargs):
    raise RuntimeError(
        "final_layer requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
