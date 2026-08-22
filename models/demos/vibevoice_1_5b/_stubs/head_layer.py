# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `head_layer` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.prediction_head.layers.0`, a
`vibevoice.modular.modular_vibevoice_diffusion_head.HeadLayer` — an AdaLN-gated
SwiGLU FFN block (no self-attention):

    shift, scale, gate = adaLN_modulation(c).chunk(3, dim=-1)  # SiLU -> Linear(cond, 3*embed), no bias
    x = x + gate * ffn(modulate(RMSNorm(x), shift, scale))     # modulate(y,s,sc) = y*(1+sc)+s

`norm` is an affine RMSNorm (has a learned `weight`), eps=1e-5. The SwiGLU
`ffn` is delegated to the already-graduated child stub
`_stubs/feed_forward_network.build`; HeadLayer itself owns only the adaLN
modulation, the RMSNorm, the gate, and the residual add.
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs import _precision as prec
from models.demos.vibevoice_1_5b._stubs.feed_forward_network import build as _build_feed_forward_network

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained HeadLayer weights and return a native ttnn forward closure."""
    m = torch_module
    eps = float(m.norm.eps)
    embed_dim = m.norm.weight.shape[0]

    norm_w = m.norm.weight.detach().float()
    adaln_w = m.adaLN_modulation[1].weight.detach().float()  # [3*embed, cond]

    def _row(t_1d):
        return ttnn.from_torch(
            t_1d.reshape(1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )

    norm_weight = _row(norm_w)
    adaln_w_t = prec.mm_weight(adaln_w.t().contiguous(), device)

    # Compose the graduated SwiGLU FFN child stub.
    ffn_forward = _build_feed_forward_network(device, m.ffn)

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
        shift = ttnn.slice(ada, [0, 0, 0], [ada.shape[0], ada.shape[1], embed_dim])
        scale = ttnn.slice(ada, [0, 0, embed_dim], [ada.shape[0], ada.shape[1], 2 * embed_dim])
        gate = ttnn.slice(ada, [0, 0, 2 * embed_dim], [ada.shape[0], ada.shape[1], 3 * embed_dim])

        sq = ttnn.mul(x, x, memory_config=_DRAM)
        mean_sq = ttnn.mean(sq, dim=-1, keepdim=True)
        denom = ttnn.rsqrt(ttnn.add(mean_sq, eps), memory_config=_DRAM)
        x_norm = ttnn.mul(ttnn.mul(x, denom, memory_config=_DRAM), norm_weight, memory_config=_DRAM)

        one_plus_scale = ttnn.add(scale, 1.0, memory_config=_DRAM)
        modulated = ttnn.add(ttnn.mul(x_norm, one_plus_scale, memory_config=_DRAM), shift, memory_config=_DRAM)

        ffn_out = ffn_forward(modulated)  # feed_forward_network child: [1, N, C] -> [1, N, C]
        return ttnn.add(x, ttnn.mul(gate, ffn_out, memory_config=_DRAM), memory_config=_DRAM)

    return forward


def head_layer(*args, **kwargs):
    raise RuntimeError(
        "head_layer requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )
