# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of the Qwen2.5-VL text self-attention
(`Qwen2_5_VLAttention`, `model.language_model.layers[i].self_attn`), prefill, no KV cache.

Scheme (see layer.py): q/k/v column-parallel by KV group (TP=4 -> 1 kv head + its 7 q heads per chip),
o_proj row-parallel + all_reduce; mRoPE sections combined on host, rotation applied on device.
"""

from __future__ import annotations

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs.attention import pad_to_tile
from models.demos.qwen_image_edit_text_encoder._stubs.layer import TtTextAttention


class TtVLAttention:
    def __init__(self, device, torch_module):
        self.attn = TtTextAttention(device, torch_module)

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        b, s, c = hidden_states.shape
        s_pad = pad_to_tile(s)
        tt_cos, tt_sin = self.attn.rope_tables(position_embeddings, s_pad)
        tt_mask = self.attn.mask(attention_mask, b, s, s_pad)
        x = ttnn.reshape(hidden_states, (b, 1, s, c))
        if s_pad != s:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, s_pad - s), (0, 0)], 0.0)
        out = self.attn.forward_padded(x, tt_cos, tt_sin, tt_mask)
        if s_pad != s:
            out = ttnn.slice(out, [0, 0, 0, 0], [b, 1, s, c])
        return ttnn.reshape(out, (b, s, c))


def build(device, torch_module=None):
    return TtVLAttention(device, torch_module)


def v_l_attention(device, torch_module=None):
    return build(device, torch_module)
