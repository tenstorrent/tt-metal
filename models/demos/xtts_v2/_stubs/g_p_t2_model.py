# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `g_p_t2_model` of coqui/XTTS-v2.

Reference submodule: `gpt.gpt`, a HuggingFace `GPT2Model` adapted by XTTS.

Structure (probed): 30 x `GPT2Block`, final `ln_f` (LayerNorm eps 1e-5), and a
`wte` token embedding that is UNUSED here (the module is called with
`inputs_embeds`). Crucially, XTTS replaces the learned position embedding
`wpe` with a null partial that returns zeros, so NO position embedding is added
inside this module — `hidden = inputs_embeds`.

Forward (captured: `inputs_embeds` `[1, 33, 1024]`, all-ones `attention_mask`
`[1, 33]`, `return_dict=True`; output `last_hidden_state` `[1, 33, 1024]`):

    hidden = inputs_embeds            # wpe == 0, drop is identity in eval
    causal_mask = <lower-triangular>  # GPT2Model builds it and passes to blocks
    for blk in h: hidden = blk(hidden, attn_bias=causal_mask)
    hidden = ln_f(hidden)

Each block is the native `g_p_t2_block` port. Unlike the standalone per-block
PCC test (attention_mask=None -> full attention), the model builds a CAUSAL
mask, so we pass an additive causal bias to every block.

Harness note: the primary arg `inputs_embeds` arrives as a ttnn tensor; the
remaining kwargs (attention_mask / return_dict / None-valued) arrive as host
values and are not needed (the captured mask is all-ones -> causal-only).
"""

from __future__ import annotations

import ttnn

from models.demos.xtts_v2._stubs.g_p_t2_block import build_gpt2_block

_LN_EPS = 1e-5
# Large negative additive bias for masked (future) positions; after softmax
# these collapse to ~0. Representable in bf16.
_MASK_NEG = -1e9


def build(device, torch_module):
    """Compose the native ttnn GPT2Model: 30 causal blocks + final layer norm."""
    import torch

    m = torch_module.float()

    block_forwards = [build_gpt2_block(device, blk) for blk in m.h]

    lnf_w = ttnn.as_tensor(
        m.ln_f.weight.detach().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    lnf_b = ttnn.as_tensor(
        m.ln_f.bias.detach().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    _mask_cache = {}

    def _causal_bias(t):
        if t not in _mask_cache:
            # [1, 1, T, T] additive bias: 0 on/below diagonal, _MASK_NEG above.
            m_t = torch.triu(torch.full((t, t), _MASK_NEG, dtype=torch.float32), diagonal=1)
            _mask_cache[t] = ttnn.as_tensor(
                m_t.reshape(1, 1, t, t).to(torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return _mask_cache[t]

    def forward(inputs_embeds, *args, **kwargs):
        hidden = inputs_embeds                       # wpe == 0, drop == identity
        bias = _causal_bias(hidden.shape[1])
        for blk in block_forwards:
            hidden = blk(hidden, attn_bias=bias)
        hidden = ttnn.layer_norm(hidden, epsilon=_LN_EPS, weight=lnf_w, bias=lnf_b)
        return hidden

    return forward


def g_p_t2_model(inputs_embeds, *args, **kwargs):
    raise RuntimeError(
        "g_p_t2_model requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )
