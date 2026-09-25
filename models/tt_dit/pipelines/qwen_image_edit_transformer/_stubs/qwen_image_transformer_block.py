# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of one QwenImageTransformerBlock (`transformer_blocks.N`).

The block math and its TP scheme live in the block-stack port (`encoder_stack._TtBlock`):
img_mod/txt_mod COLUMN-parallel + all_gather, TP joint attention (heads split, row-parallel out +
all_reduce), FeedForward COLUMN/ROW-parallel + all_reduce, float32 LayerNorm/modulation/residuals.
This wraps that block for a single-block call and returns (encoder_hidden_states, hidden_states),
the same order diffusers returns.
"""

from __future__ import annotations

from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs.encoder_stack import TtQwenBlockStack


class TtQwenTransformerBlock:
    def __init__(self, device, torch_module):
        self.stack = TtQwenBlockStack(device, [torch_module])
        self.device = device

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        temb=None,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        modulate_index=None,
    ):
        img, txt = self.stack.run_streams(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            modulate_index=modulate_index,
        )
        return txt, img


def build(device, torch_module=None):
    return TtQwenTransformerBlock(device, torch_module)


def qwen_image_transformer_block(device, torch_module=None):
    return TtQwenTransformerBlock(device, torch_module)
