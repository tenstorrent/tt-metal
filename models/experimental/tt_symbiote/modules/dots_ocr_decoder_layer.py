# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.modules.dots_ocr_attention import TTNNDotsOCRAttention
from models.experimental.tt_symbiote.modules.dots_ocr_mlp import TTNNDotsOCRMLP
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm


@trace_enabled
class TTNNDotsOCRDecoderLayer(TTNNModule):
    def __init__(self):
        super().__init__()
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.self_attn = None
        self.mlp = None

    @classmethod
    def from_torch(cls, torch_layer):
        new_layer = cls()
        new_layer._fallback_torch_layer = torch_layer
        new_layer.attention_type = getattr(torch_layer, "attention_type", "full_attention")
        new_layer.input_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.input_layernorm)
        new_layer.post_attention_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.post_attention_layernorm)
        new_layer.self_attn = TTNNDotsOCRAttention.from_torch(torch_layer.self_attn)
        new_layer.mlp = TTNNDotsOCRMLP.from_torch(torch_layer.mlp)
        return new_layer

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        hidden_states = func_args[0]
        seq_len = hidden_states.shape[-2]
        layer_idx = self.self_attn.layer_idx
        past_key_value.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        **kwargs,
    ):
        hs = hidden_states

        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hs.dtype != ttnn.bfloat16:
            hs = ttnn.typecast(hs, ttnn.bfloat16)

        # Attention block
        residual = hs
        hs = self.input_layernorm(hs)

        seq_len = hs.shape[-2]
        is_decode = seq_len == 1
        attn_out, _ = self.self_attn(
            hidden_states=hs,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=past_key_value,
            cache_position=kwargs.get("cache_position"),
        )

        hs = ttnn.add(residual, attn_out)
        ttnn.deallocate(attn_out)

        # MLP block
        residual = hs
        hs = self.post_attention_layernorm(hs)
        mlp_out = self.mlp(hs)

        hs = ttnn.add(residual, mlp_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)

        # CRITICAL: Return tuple — Qwen2Model does layer_outputs[0]
        return (hs,)
