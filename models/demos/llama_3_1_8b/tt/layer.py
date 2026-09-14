# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One decoder layer: the composition D2 fixes the signatures of and D3 fills in.

    x -> input_layernorm -> attention -> + x
      -> post_attention_layernorm -> mlp -> + x

Every layer of Llama-3.1-8B is identical — no hybrid dense/MoE schedule, no sliding/full alternation,
no per-layer type dispatch. The per-layer dispatch that M3 and gpt-oss need therefore has nothing to
dispatch on here, and adding an always-true branch would only invite a future model to take the
wrong one silently.
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule

from .attention import Attention, AttentionConfig
from .mlp import MLP
from .rms_norm import RMSNorm
from ..utils.general import cache_name, substate


class DecoderLayer(LightweightModule):
    def __init__(
        self,
        mesh_device,
        cfg,
        mesh_config,
        ccl_manager,
        rope_setup,
        layer_idx: int,
        state_dict=None,
        max_seq_len: int = 10240,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        sequence_parallel: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(
            mesh_device,
            cfg.hidden_size,
            cfg.rms_norm_eps,
            state_dict=substate(state_dict, "input_layernorm") if state_dict else None,
            cache_file_name=cache_name(tensor_cache_path, "input_layernorm"),
        )
        self.self_attn = Attention(
            mesh_device,
            AttentionConfig.from_model_config(cfg, max_seq_len=max_seq_len, sequence_parallel=sequence_parallel),
            mesh_config,
            ccl_manager,
            rope_setup,
            state_dict=substate(state_dict, "self_attn") if state_dict else None,
            layer_idx=layer_idx,
            weight_dtype=weight_dtype,
            tensor_cache_path=cache_name(tensor_cache_path, "self_attn"),
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            cfg.hidden_size,
            cfg.rms_norm_eps,
            state_dict=substate(state_dict, "post_attention_layernorm") if state_dict else None,
            cache_file_name=cache_name(tensor_cache_path, "post_attention_layernorm"),
        )
        self.mlp = MLP(
            mesh_device,
            cfg,
            mesh_config,
            ccl_manager,
            state_dict=substate(state_dict, "mlp") if state_dict else None,
            weight_dtype=weight_dtype,
            tensor_cache_path=cache_name(tensor_cache_path, "mlp"),
        )

    def forward(self, hidden_states, rope_mats, *, kv_cache=None, user_id=0, cached_len=0, indexed_rope=False):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            normed,
            rope_mats,
            kv_cache=kv_cache,
            user_id=user_id,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )
        normed.deallocate(True)
        hidden_states = ttnn.add(residual, attn_out, output_tensor=attn_out)
        residual.deallocate(True)

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed)
        normed.deallocate(True)
        hidden_states = ttnn.add(residual, mlp_out, output_tensor=mlp_out)
        residual.deallocate(True)
        return hidden_states
