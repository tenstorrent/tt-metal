# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of `Qwen2_5_VLVisionBlock` (`model.visual.blocks[i]`):
x + attn(norm1(x)); x + mlp(norm2(x)). Attention heads and MLP intermediate split across TP with an
all_reduce after each row-parallel projection; norms replicated; fp32 residual stream.
"""

from __future__ import annotations

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs.attention import block_mask, pad_to_tile, upload
from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import TtVisionBlock, _fp32


class TtVLVisionBlock:
    def __init__(self, device, torch_module):
        self.device = device
        self.block = TtVisionBlock(device, torch_module)

    def forward_padded(self, x, tt_cos, tt_sin, tt_mask):
        """Device path used by the vision tower: x [N, 1, s_pad, C] fp32 residual stream."""
        return self.block.forward_padded(x, tt_cos, tt_sin, tt_mask)

    def __call__(self, hidden_states, cu_seqlens=None, position_embeddings=None, **kwargs):
        s, c = hidden_states.shape
        s_pad = pad_to_tile(s)
        cos, sin = position_embeddings
        tt_cos, tt_sin = self.block.attn.rope_tables(cos.float().numpy(), sin.float().numpy(), s_pad)
        bounds = cu_seqlens.tolist() if cu_seqlens is not None else [0, s]
        tt_mask = upload(self.device, block_mask(bounds, s, s_pad))
        x = ttnn.reshape(hidden_states, (1, 1, s, c))
        if s_pad != s:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, s_pad - s), (0, 0)], 0.0)
        x = self.block.forward_padded(_fp32(x), tt_cos, tt_sin, tt_mask)
        if s_pad != s:
            x = ttnn.slice(x, [0, 0, 0, 0], [1, 1, s, c])
        return ttnn.reshape(x, (s, c))


def build(device, torch_module=None):
    return TtVLVisionBlock(device, torch_module)


def v_l_vision_block(device, torch_module=None):
    return build(device, torch_module)
