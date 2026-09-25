# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN, tensor-parallel port of a QwenImage block's self-attention (`transformer_blocks.N.attn`).
QwenImage's only self-attention is the joint (dual-stream) attention over cat[txt, img], so this
uses the joint-attention port: heads split across chips (q/k/v COLUMN-parallel), to_out/to_add_out
ROW-parallel + all_reduce, norms and rotary tables replicated.
"""

from __future__ import annotations

from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs.attention import TtQwenJointAttention


def build(device, torch_module=None):
    return TtQwenJointAttention(device, torch_module)


def self_attention(device, torch_module=None):
    return TtQwenJointAttention(device, torch_module)
