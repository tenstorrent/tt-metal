# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""A single dots.ocr decoder block in TP4 (replicated-hidden Megatron design).

    residual = x
    h = input_layernorm(x)
    h = self_attn(h)            # all-reduced -> replicated
    x = residual + h
    residual = x
    h = post_attention_layernorm(x)
    h = mlp(h)                  # all-reduced -> replicated
    x = residual + h

Everything stays replicated full-width on every chip.
"""

import ttnn

from models.experimental.dots_ocr_tp4.tt.attention import DotsOCRAttentionTP4
from models.experimental.dots_ocr_tp4.tt.mlp import DotsOCRMLPTP4
from models.experimental.dots_ocr_tp4.tt.rmsnorm import DotsOCRRMSNormTP4


class DotsOCRDecoderBlockTP4:
    def __init__(self, mesh_device, config, layer_idx=0, weight_dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.config = config
        self.layer_idx = layer_idx
        self.weight_dtype = weight_dtype
        self.input_layernorm = None
        self.self_attn = None
        self.post_attention_layernorm = None
        self.mlp = None

    @classmethod
    def from_torch(cls, mesh_device, config, torch_layer, layer_idx=0, weight_dtype=ttnn.bfloat16):
        b = cls(mesh_device, config, layer_idx=layer_idx, weight_dtype=weight_dtype)
        b.input_layernorm = DotsOCRRMSNormTP4.from_torch(
            mesh_device, torch_layer.input_layernorm, eps=config.rms_norm_eps
        )
        b.self_attn = DotsOCRAttentionTP4.from_torch(
            mesh_device, config, torch_layer.self_attn, layer_idx=layer_idx, weight_dtype=weight_dtype
        )
        b.post_attention_layernorm = DotsOCRRMSNormTP4.from_torch(
            mesh_device, torch_layer.post_attention_layernorm, eps=config.rms_norm_eps
        )
        b.mlp = DotsOCRMLPTP4.from_torch(mesh_device, config, torch_layer.mlp, weight_dtype=weight_dtype)
        return b

    def forward(self, x: ttnn.Tensor, past_key_value=None, cache_position=None) -> ttnn.Tensor:
        residual = x
        h = self.input_layernorm.forward(x)
        h = self.self_attn.forward(h, past_key_value=past_key_value, cache_position=cache_position)
        x = ttnn.add(residual, h)
        ttnn.deallocate(h)

        residual = x
        h = self.post_attention_layernorm.forward(x)
        h = self.mlp.forward(h)
        x = ttnn.add(residual, h)
        ttnn.deallocate(h)
        return x
