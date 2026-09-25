# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `Qwen2_5_VLRotaryEmbedding` (`model.language_model.rotary_emb`), mRoPE tables.

    freqs[a, b, s, :] = position_ids[a, b, s] * inv_freq          a in (t, h, w), fp32
    emb = cat(freqs, freqs);  returns (cos(emb) * scale, sin(emb) * scale) in x's dtype

The position ids are uploaded as fp32 (exact for integer positions); the outer product, concat and
cos/sin all run on device. A table op -- replicated on every chip of a mesh.
"""

from __future__ import annotations

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs.attention import upload


class TtTextRotaryEmbedding:
    def __init__(self, device, torch_module):
        self.device = device
        self.scale = float(getattr(torch_module, "attention_scaling", 1.0))
        inv = torch_module.inv_freq.detach().float()
        self.inv_freq = upload(device, inv.reshape(1, 1, 1, -1), dtype=ttnn.float32)

    def __call__(self, x, position_ids=None, **kwargs):
        a, b, s = position_ids.shape[0], position_ids.shape[1], position_ids.shape[2]
        if isinstance(position_ids, ttnn.Tensor):  # already on device (fp32 [3, B, S, 1])
            pos = position_ids
        else:
            pos = upload(self.device, position_ids.float().reshape(a, b, s, 1), dtype=ttnn.float32)
        freqs = ttnn.multiply(pos, self.inv_freq)  # [3, B, S, D/2]
        emb = ttnn.concat([freqs, freqs], dim=-1)
        cos, sin = ttnn.cos(emb), ttnn.sin(emb)
        if self.scale != 1.0:
            cos, sin = ttnn.multiply(cos, self.scale), ttnn.multiply(sin, self.scale)
        out_dtype = kwargs.get("dtype") or (x.dtype if isinstance(x, ttnn.Tensor) else ttnn.bfloat16)
        return ttnn.typecast(cos, out_dtype), ttnn.typecast(sin, out_dtype)


def build(device, torch_module=None):
    return TtTextRotaryEmbedding(device, torch_module)


def v_l_rotary_embedding(device, torch_module=None):
    return build(device, torch_module)
