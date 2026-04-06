# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Decoder Layer.

Each layer has 7 RMSNorms + layer_scalar:
  - input_layernorm: before attention
  - post_attention_layernorm: after attention, before residual add
  - pre_feedforward_layernorm: before shared MLP
  - post_feedforward_layernorm: after combined MLP+MoE, before final residual add
  - post_feedforward_layernorm_1: after shared MLP output (MoE path only)
  - pre_feedforward_layernorm_2: before expert input (MoE path only)
  - post_feedforward_layernorm_2: after expert output (MoE path only)
  - layer_scalar: learned per-layer scalar

Forward flow (matching HF exactly):
  residual = x
  x = input_layernorm(x)
  x = self_attn(x)
  x = post_attention_layernorm(x)
  x = residual + x

  residual = x
  x = pre_feedforward_layernorm(x)
  x = mlp(x)

  if enable_moe_block:
    x_1 = post_feedforward_layernorm_1(x)
    x_flat = residual.reshape(-1, H)     # router input = pre-norm residual
    _, top_k_w, top_k_idx = router(x_flat)
    x_2 = pre_feedforward_layernorm_2(x_flat)
    x_2 = experts(x_2, top_k_idx, top_k_w)
    x_2 = post_feedforward_layernorm_2(x_2)
    x = x_1 + x_2

  x = post_feedforward_layernorm(x)
  x = residual + x
  x *= layer_scalar
"""

import torch

import ttnn
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.gemma4_attention_config import get_attention_program_config
from models.demos.gemma4.tt.moe import MoEBlock
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.tt.shared_mlp import SharedMLP
from models.demos.gemma4.utils.substate import substate


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
        transformation_mats=None,  # Legacy — ignored (HF-style RoPE needs no transformation mats)
    ):
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.hidden_size = hf_config.hidden_size
        self.layer_type = hf_config.layer_types[layer_idx]
        self.enable_moe_block = hf_config.enable_moe_block

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

        # 3 additional norms for MoE layers
        if self.enable_moe_block:
            self.post_feedforward_layernorm_1 = _norm("post_feedforward_layernorm_1")
            self.pre_feedforward_layernorm_2 = _norm("pre_feedforward_layernorm_2")
            self.post_feedforward_layernorm_2 = _norm("post_feedforward_layernorm_2")

        # Layer scalar
        if layer_state and "layer_scalar" in layer_state:
            self.layer_scalar = layer_state["layer_scalar"].item()
        else:
            self.layer_scalar = 1.0

        # Attention
        attn_config = Gemma4AttentionConfig(hf_config, layer_idx)
        attn_program_config = get_attention_program_config(attn_config, mesh_config, is_decode=True)
        self.self_attn = Gemma4Attention(
            mesh_device=mesh_device,
            config=attn_config,
            state_dict=substate(layer_state, "self_attn") if layer_state else {},
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=attn_program_config,
            layer_idx=layer_idx,
            tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/self_attn" if tensor_cache_path else None,
        )

        # Shared/dense MLP (HF key: "mlp")
        self.shared_mlp = SharedMLP(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=substate(layer_state, "mlp") if layer_state else {},
            mesh_config=mesh_config,
            dtype=dtype,
            tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/mlp" if tensor_cache_path else None,
        )

        # MoE block (router + routed experts)
        if self.enable_moe_block:
            self.moe = MoEBlock(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=layer_state,  # MoE expects "router.*" and "experts.*" keys
                ccl_manager=ccl_manager,
                mesh_config=mesh_config,
                dtype=dtype,
                tensor_cache_path=f"{tensor_cache_path}/layer_{layer_idx}/moe" if tensor_cache_path else None,
            )

    def __call__(self, hidden_states, rope_mats, position_idx, page_table, kv_cache, is_decode, token_index=None):
        """
        Decoder layer forward pass.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device
            rope_mats: precomputed RoPE matrices
            position_idx: current position index
            page_table: paged attention page table
            kv_cache: KV cache for this layer
            is_decode: True for decode mode

        Returns:
            hidden_states: [1, 1, seq_len, hidden_size] on device
        """
        # 1. Attention block: norm -> attn -> post_attn_norm -> residual add
        residual = hidden_states
        normed = self.input_layernorm.forward(hidden_states)
        attn_output = self.self_attn(
            normed,
            rope_mats=rope_mats,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            token_index=token_index,
        )

        # attn_output is placeholder (torch tensor) until attention is implemented
        if isinstance(attn_output, torch.Tensor):
            hidden_states = residual
        else:
            attn_output = self.post_attention_layernorm.forward(attn_output)
            hidden_states = ttnn.add(residual, attn_output)
            residual.deallocate(True)
            attn_output.deallocate(True)

        # 2. MLP + MoE block
        residual = hidden_states
        normed = self.pre_feedforward_layernorm.forward(hidden_states)
        mlp_output = self.shared_mlp(normed)
        normed.deallocate(True)

        if self.enable_moe_block:
            # post_feedforward_layernorm_1 on MLP output
            mlp_normed = self.post_feedforward_layernorm_1.forward(mlp_output)
            mlp_output.deallocate(True)

            # Router input = pre-MLP residual, expert input = normed residual
            # All on device — no CPU round-trip
            residual_for_router = residual
            expert_input = self.pre_feedforward_layernorm_2.forward(residual_for_router)

            # MoE: router(residual) → dense_routing → experts(normed_input, routing)
            expert_output = self.moe(residual_for_router, expert_input)
            expert_input.deallocate(True)

            # post_feedforward_layernorm_2 on expert output
            expert_normed = self.post_feedforward_layernorm_2.forward(expert_output)
            expert_output.deallocate(True)

            # Combine: mlp_normed + expert_normed
            hidden_states = ttnn.add(mlp_normed, expert_normed)
            mlp_normed.deallocate(True)
            expert_normed.deallocate(True)
        else:
            hidden_states = mlp_output

        # post_feedforward_layernorm -> residual add
        hidden_states = self.post_feedforward_layernorm.forward(hidden_states)
        combined = ttnn.add(residual, hidden_states)
        residual.deallocate(True)
        hidden_states.deallocate(True)

        # Layer scalar
        if self.layer_scalar != 1.0:
            hidden_states = ttnn.mul(combined, self.layer_scalar)
            combined.deallocate(True)
        else:
            hidden_states = combined

        return hidden_states
