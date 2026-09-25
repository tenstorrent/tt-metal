# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `Qwen2_5_VisionRotaryEmbedding` (`model.visual.rotary_pos_emb`).

    out = (position_ids[..., None] * inv_freq).flatten(1)      position_ids: [S, 2] (row, col)

i.e. out[:, :F] = row * inv_freq and out[:, F:] = col * inv_freq, computed on device in fp32.
A table op -- replicated on every chip of a mesh.
"""

from __future__ import annotations

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs.attention import upload


class TtVisionRotaryEmbedding:
    def __init__(self, device, torch_module):
        self.device = device
        inv = torch_module.inv_freq.detach().float()
        self.inv_freq = upload(device, inv.reshape(1, -1), dtype=ttnn.float32)

    def __call__(self, position_ids, **kwargs):
        pos = position_ids
        if pos.layout != ttnn.TILE_LAYOUT:
            pos = ttnn.to_layout(pos, ttnn.TILE_LAYOUT)
        if pos.dtype != ttnn.float32:
            pos = ttnn.typecast(pos, ttnn.float32)
        s, n = pos.shape[0], pos.shape[1]
        cols = [ttnn.multiply(ttnn.slice(pos, [0, i], [s, i + 1]), self.inv_freq) for i in range(n)]
        return ttnn.concat(cols, dim=-1)


def build(device, torch_module=None):
    return TtVisionRotaryEmbedding(device, torch_module)


def vision_rotary_embedding(device, torch_module=None):
    return build(device, torch_module)
