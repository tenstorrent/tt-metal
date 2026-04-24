# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNLayerStack, TTNNModule
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

    def call(self, *args, **kwds):
        # Keep only kwargs used by forward — unused kwargs with incompatible
        # dtypes (e.g. UINT8 from bool masks) cause ttnn.copy failures in trace replay.
        filtered = {k: kwds[k] for k in ("past_key_value", "cache_position") if k in kwds}
        return super().call(*args, **filtered)

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


class TTNNDotsOCRLayerStack(TTNNLayerStack):
    def call(self, *args, **kwds):
        filtered = {k: kwds[k] for k in ("past_key_value", "cache_position") if k in kwds}
        return super().call(*args, **filtered)

    def forward(self, hidden_states, **kwargs):
        for layer in self.layers:
            layer_output = layer.forward(hidden_states, **kwargs)
            hidden_states = layer_output[0]
        return hidden_states

    def pre_trace_execute(self, func_args, func_kwargs):
        cache_position = func_kwargs.get("cache_position")
        if cache_position is None:
            return

        cp = cache_position
        if hasattr(cp, "ttnn_tensor") and cp.ttnn_tensor is not None:
            cp = cp.ttnn_tensor

        if len(cp.shape) > 1:
            total = 1
            for d in cp.shape:
                total *= d
            cp = ttnn.reshape(cp, (total,))

        for layer in self.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "_decode_cur_pos") and attn._decode_cur_pos is not None:
                cur = cp
                if cur.shape[0] > 1:
                    cur = ttnn.slice(cur, [0], [1])
                ttnn.copy(cur, attn._decode_cur_pos)

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        hidden_states = func_args[0]
        seq_len = hidden_states.shape[-2]
        for layer in self.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer_idx = layer.self_attn.layer_idx
                past_key_value.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)
