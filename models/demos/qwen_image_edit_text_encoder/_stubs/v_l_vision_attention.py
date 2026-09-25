# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of `Qwen2_5_VLVisionAttention` (`model.visual.blocks[i].attn`).

Same module as `attention`: heads split across TP (column-parallel qkv, row-parallel proj +
all_reduce), cu_seqlens expressed as an additive block mask. See attention.py.
"""

from __future__ import annotations

from models.demos.qwen_image_edit_text_encoder._stubs.attention import TtVisionAttention


def build(device, torch_module=None):
    return TtVisionAttention(device, torch_module)


def v_l_vision_attention(device, torch_module=None):
    return build(device, torch_module)
