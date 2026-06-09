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
from models.experimental.tt_symbiote.core.module import TTNNModule


def _gate_up_dtype_for_layer(layer_idx) -> "ttnn.DataType":
    # Mirror the production dots.ocr recipe: gate/up is BFP4 on the early layers
    # (0-6) and promoted to BFP8 on the later, accuracy-sensitive layers (>=7).
    # Pure BFP4 everywhere over-degrades the 28-layer real-weight PCC.
    if layer_idx is not None and int(layer_idx) >= 7:
        return ttnn.bfloat8_b
    from models.experimental.dots_ocr_tp4.tt.common import tp4_lossy_matmul_dtype

    return tp4_lossy_matmul_dtype()


class DotsOCRDecoderBlockTP4(TTNNModule):
    def __init__(self, mesh_device, config, layer_idx=0, weight_dtype=ttnn.bfloat16):
        super().__init__()
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
        b.mlp = DotsOCRMLPTP4.from_torch(
            mesh_device,
            config,
            torch_layer.mlp,
            weight_dtype=weight_dtype,
            gate_up_weight_dtype=_gate_up_dtype_for_layer(layer_idx),
        )
        b.to_device(mesh_device)
        b._preprocessed_weight = True
        b._weights_on_device = True
        return b

    def to_device(self, device):
        super().to_device(device)
        for child in (self.input_layernorm, self.self_attn, self.post_attention_layernorm, self.mlp):
            if child is not None:
                child.to_device(device)
        return self

    def forward(self, x: ttnn.Tensor, past_key_value=None, cache_position=None) -> ttnn.Tensor:
        # Decode (seq==1) keeps the residual stream L1-resident; prefill stays DRAM.
        is_decode = int(x.shape[-2]) == 1
        add_mc = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG

        residual = x
        h = self.input_layernorm.forward(x)
        h = self.self_attn.forward(h, past_key_value=past_key_value, cache_position=cache_position)
        x = ttnn.add(residual, h, memory_config=add_mc)
        ttnn.deallocate(h)

        residual = x
        h = self.post_attention_layernorm.forward(x)
        h = self.mlp.forward(h)
        x = ttnn.add(residual, h, memory_config=add_mc)
        ttnn.deallocate(h)
        return x
