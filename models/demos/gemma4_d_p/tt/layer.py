# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4-31B dense decoder layer: attention, MLP, residuals and layer scalar."""


import ttnn
from models.demos.gemma4_d_p.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4_d_p.tt.mlp import MLP
from models.demos.gemma4_d_p.tt.rms_norm import RMSNorm
from models.demos.gemma4_d_p.utils.substate import substate


class Gemma4DecoderLayer:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        ccl_manager,
        dtype,
        tensor_cache_path,
        mesh_config,
        max_seq_len,
        max_local_batch_size,
        mlp_dtype=None,
        attention_dtype=None,
        ring_kv_cache=None,
        ring_layer_idx=0,
        ring_num_layers=1,
    ):
        # Per-module dtype overrides default to the model-wide ``dtype`` so
        # callers that don't care about precision config see no change.
        if mlp_dtype is None:
            mlp_dtype = dtype
        if attention_dtype is None:
            attention_dtype = dtype
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.hidden_size = hf_config.hidden_size
        self.layer_type = hf_config.layer_types[layer_idx]

        # Try both key formats (HF uses "model.language_model.layers", tests use "model.layers")
        layer_state = {}
        if state_dict:
            for prefix in [f"model.language_model.layers.{layer_idx}", f"model.layers.{layer_idx}"]:
                layer_state = substate(state_dict, prefix)
                if layer_state:
                    break

        def _norm(name, with_scale=True):
            return RMSNorm(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=substate(layer_state, name) if layer_state else {},
                tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/{name}" if tensor_cache_path else None,
                mesh_config=mesh_config,
                with_scale=with_scale,
            )

        # 4 norms present on every layer
        self.input_layernorm = _norm("input_layernorm")
        self.post_attention_layernorm = _norm("post_attention_layernorm")
        self.pre_feedforward_layernorm = _norm("pre_feedforward_layernorm")
        self.post_feedforward_layernorm = _norm("post_feedforward_layernorm")

        # Layer scalar
        if layer_state and "layer_scalar" in layer_state:
            self.layer_scalar = layer_state["layer_scalar"].item()
        else:
            self.layer_scalar = 1.0

        # Attention
        attn_config = Gemma4AttentionConfig(hf_config, layer_idx)
        self.self_attn = Gemma4Attention(
            mesh_device=mesh_device,
            config=attn_config,
            state_dict=substate(layer_state, "self_attn") if layer_state else {},
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            layer_idx=layer_idx,
            tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/self_attn" if tensor_cache_path else None,
            weight_dtype=attention_dtype,
            ring_kv_cache=ring_kv_cache,
            ring_layer_idx=ring_layer_idx,
            ring_num_layers=ring_num_layers,
            max_seq_len=max_seq_len,
            max_batch_size=max_local_batch_size,
        )

        # Dense MLP (HF key: "mlp")
        self.mlp = MLP(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=substate(layer_state, "mlp") if layer_state else {},
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            dtype=mlp_dtype,
            tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/mlp" if tensor_cache_path else None,
        )

    def __call__(
        self,
        hidden_states,
        rope_mats,
        chunk_start_idx=0,
        packed_global_rope=None,
        packed_sliding_rope=None,
    ):
        """Prefill one CP-sharded chunk."""
        # 1. Attention block: norm -> attn -> post_attn_norm -> residual add
        residual = hidden_states
        normed = self.input_layernorm.forward(hidden_states)
        attn_output = self.self_attn(
            normed,
            rope_mats=rope_mats,
            chunk_start_idx=chunk_start_idx,
            packed_global_rope=packed_global_rope,
            packed_sliding_rope=packed_sliding_rope,
        )

        attn_output = self.post_attention_layernorm.forward(attn_output)
        hidden_states = ttnn.add(residual, attn_output)
        residual.deallocate(True)
        attn_output.deallocate(True)

        # 2. Dense MLP block
        residual = hidden_states
        normed = self.pre_feedforward_layernorm.forward(hidden_states)
        mlp_output = self.mlp(normed)
        normed.deallocate(True)

        hidden_states = mlp_output

        # post_feedforward_layernorm -> residual add
        hidden_states = self.post_feedforward_layernorm.forward(hidden_states)
        combined = ttnn.add(residual, hidden_states)
        residual.deallocate(True)
        hidden_states.deallocate(True)

        hidden_states = combined

        if self.layer_scalar != 1.0:
            hidden_states = ttnn.mul(hidden_states, self.layer_scalar)

        return hidden_states
