# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for GLM4.5 Air model with TTNN backend."""

import os

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.attention import LlamaAttention, TTNNSDPAAttention
from models.experimental.tt_symbiote.modules.normalization import TTNNRMSNorm
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearLLama, TTNNLinearSilu
from models.experimental.tt_symbiote.modules.rope import TTNNRotaryPositionEmbedding
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

import transformers

assert transformers.__version__.startswith("5."), "This test requires transformers version 5.0.0.dev0"

from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeRMSNorm


def get_full_attention_mappings():
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeAttention as OriginalGlm4MoeAttention

    class GLMLlamaAttention(LlamaAttention):
        """Multi-headed attention from 'Attention Is All You Need' paper"""

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,  # will become mandatory in v4.46
            **kwargs,
        ):
            if attention_mask is not None:
                print(
                    "Warning: attention_mask is not None, but TTNN LlamaAttention does not support it yet."
                )  # --- IGNORE ---
            past_key_values = (
                kwargs.get("past_key_value", past_key_values) if past_key_values is None else past_key_values
            )
            if self.qkv_same_shape:
                query_states, key_states, value_states = self.qkv_proj(hidden_states)
            else:
                input_shape = list(hidden_states.shape)[:-1]
                hidden_shape = (*input_shape, -1, self.torch_layer.head_dim)
                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings

            query_states, key_states = self.rope(query_states, key_states, cos, sin)

            if past_key_values is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.torch_layer.layer_idx, cache_kwargs
                )

            original_q_len = query_states.shape[2]
            kv_len = key_states.shape[2]

            if self.torch_layer.is_causal and original_q_len < kv_len:
                # Pad query: [B, H, q_len, D] -> [B, H, kv_len, D]
                pad_len = kv_len - original_q_len
                # Create zero padding on device
                pad_shape = (query_states.shape[0], query_states.shape[1], pad_len, query_states.shape[3])
                zero_pad = ttnn.zeros(
                    pad_shape,
                    device=hidden_states.device(),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=hidden_states.dtype,
                )
                query_states = ttnn.concat([zero_pad, query_states.to_ttnn], dim=2)

            attn_out = self.sdpa(
                self,
                query_states,
                key_states,
                value_states,
                None,
                dropout=0.0,
                scaling=self.torch_layer.scaling,
                is_causal=self.torch_layer.is_causal,
                transpose_output=False,
            )
            attn_out = ttnn.transpose(attn_out.to_ttnn, 1, 2)
            attn_out = ttnn.reshape(attn_out, [-1, attn_out.shape[1], attn_out.shape[2] * attn_out.shape[3]])
            # Slice output if query was padded
            if self.torch_layer.is_causal and original_q_len < kv_len:
                # Slice: [B, kv_len, D] -> [B, q_len, D]
                attn_out = attn_out[:, -original_q_len:, :]

            return self.o_proj(attn_out), None

    return {OriginalGlm4MoeAttention: GLMLlamaAttention}


def get_attention_mappings():
    from transformers.integrations import use_kernelized_func
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeAttention as OriginalGlm4MoeAttention
    from transformers.models.glm4_moe.modeling_glm4_moe import apply_rotary_pos_emb

    @use_kernelized_func(apply_rotary_pos_emb)
    class Glm4MoeAttention(nn.Module):
        """Multi-headed attention from 'Attention Is All You Need' paper"""

        def __init__(self, orig_layer: OriginalGlm4MoeAttention):
            super().__init__()
            self.config = orig_layer.config
            self.layer_idx = orig_layer.layer_idx
            self.head_dim = orig_layer.head_dim
            self.num_key_value_groups = orig_layer.num_key_value_groups
            self.scaling = orig_layer.scaling
            self.rope_parameters = orig_layer.rope_parameters
            self.attention_dropout = orig_layer.attention_dropout
            self.is_causal = True

            self.q_proj = orig_layer.q_proj
            self.k_proj = orig_layer.k_proj
            self.v_proj = orig_layer.v_proj
            self.o_proj = orig_layer.o_proj
            self.use_qk_norm = orig_layer.use_qk_norm
            if self.use_qk_norm:
                self.q_norm = orig_layer.q_norm
                self.k_norm = orig_layer.k_norm
            self.ttnn_attention_module = TTNNSDPAAttention()
            self.ttnn_rope = TTNNRotaryPositionEmbedding()
            # self.ttnn_permute = TTNNPermute()
            # self.ttnn_reshape = TTNNReshape()

        @classmethod
        def from_torch(cls, attention_module: OriginalGlm4MoeAttention) -> "TTNNGlm4MoeAttention":
            """Create TTNNBottleneck from PyTorch Bottleneck layer."""
            new_attention = cls(attention_module)
            return new_attention

        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            assert not self.use_qk_norm, "QK norm not supported in rewritten attention yet."
            query_states = self.q_proj(hidden_states).reshape(hidden_shape)
            key_states = self.k_proj(hidden_states).reshape(hidden_shape)
            value_states = self.v_proj(hidden_states).reshape(hidden_shape)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = self.ttnn_rope(query_states, key_states, cos, sin)

            if past_key_values is not None:
                # sin and cos are specific to RoPE models; position_ids needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            attn_output = self.ttnn_attention_module(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1)
            attn_output = self.o_proj(attn_output)
            return attn_output, None

    return {OriginalGlm4MoeAttention: Glm4MoeAttention}


def get_rewritten_glm_moe_mlp():
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMLP as OriginalGlm4MoeMLP

    class Glm4MoeMLP(nn.Module):
        def __init__(self, old_layer):
            super().__init__()
            self.config = old_layer.config
            self.hidden_size = old_layer.hidden_size
            self.intermediate_size = old_layer.intermediate_size
            self.gate_proj = old_layer.gate_proj
            self.up_proj = old_layer.up_proj
            self.down_proj = old_layer.down_proj
            assert old_layer.config.hidden_act == "silu", "Only SiLU activation is supported in rewritten MLP."
            self.act_fn = nn.SiLU()

        @classmethod
        def from_torch(cls, mlp_module: OriginalGlm4MoeMLP) -> "TTNNGlm4MoeMLP":
            """Create TTNNGlm4MoeMLP from PyTorch Glm4MoeMLP layer."""
            new_mlp = cls(mlp_module)
            return new_mlp

        def forward(self, x):
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return down_proj

    return {OriginalGlm4MoeMLP: Glm4MoeMLP}


def get_router_mapping():
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeTopkRouter
    from ttnn.model_preprocessing import preprocess_linear_weight

    from models.experimental.tt_symbiote.core.module import TTNNModule, deallocate_weights_after

    class TTNNGlm4MoeTopkRouter(TTNNModule):
        def preprocess_weights_impl(self):
            """Preprocess linear weights for TTNN."""
            self.tt_weight_host = preprocess_linear_weight(
                self.torch_layer.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            self.e_score_correction_bias = self.torch_layer.e_score_correction_bias

        def move_weights_to_device_impl(self):
            """Move weights to TTNN device."""
            self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)

        def deallocate_weights_impl(self):
            """Deallocate weights from device."""
            ttnn.deallocate(self.tt_weight)
            super().deallocate_weights_impl()

        @deallocate_weights_after
        def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
            """Forward pass through linear layer."""
            if input_tensor.layout != ttnn.TILE_LAYOUT:
                input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            input_tensor_shape = list(input_tensor.shape)
            input_shape = list(input_tensor_shape)
            while len(input_shape) < 4:
                input_shape.insert(0, 1)  # Add batch dimensions if needed
            input_tensor = ttnn.reshape(input_tensor, input_shape)
            tt_output = ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tt_output = ttnn.reshape(tt_output, [-1] + [tt_output.shape[-1]])
            return tt_output

    return {Glm4MoeTopkRouter: TTNNGlm4MoeTopkRouter}


def get_naive_moe_mapping():
    return {}
    from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeNaiveMoe as OriginalGlm4MoeNaiveMoe

    class Glm4MoeNaiveMoe(nn.Module):
        """Collection of expert weights stored as 3D tensors."""

        def __init__(self, old_layer):
            super().__init__()
            self.num_experts = old_layer.num_experts
            self.hidden_dim = old_layer.hidden_dim
            self.intermediate_dim = old_layer.intermediate_dim
            self.gate_layers = {
                i: TTNNLinearSilu.from_parameters(
                    old_layer.gate_up_proj[i, : self.intermediate_dim, :], linear_class=TTNNLinearLLama
                )
                for i in range(self.num_experts)
            }
            self.up_layers = {
                i: TTNNLinearLLama.from_parameters(old_layer.gate_up_proj[i, self.intermediate_dim :, :])
                for i in range(self.num_experts)
            }
            del old_layer.gate_up_proj
            self.down_layers = {
                i: TTNNLinearLLama.from_parameters(old_layer.down_proj[i, :, :]) for i in range(self.num_experts)
            }
            del old_layer.down_proj
            assert old_layer.config.hidden_act == "silu", "Only SiLU activation is supported in naive MoE."

        @classmethod
        def from_torch(cls, moe_module: OriginalGlm4MoeNaiveMoe) -> "TTNNGlm4MoeNaiveMoe":
            """Create TTNNGlm4MoeNaiveMoe from PyTorch Glm4MoeNaiveMoe layer."""
            new_moe = cls(moe_module)
            return new_moe

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                int_expert = expert_idx.item()
                gate = self.gate_layers[int_expert](current_state)
                up = self.up_layers[int_expert](current_state)
                current_hidden_states = gate * up
                current_hidden_states = self.down_layers[int_expert](current_hidden_states)
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

            return final_hidden_states

    return {OriginalGlm4MoeNaiveMoe: Glm4MoeNaiveMoe}


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_glm(mesh_device):
    """Test GLM model with TTNN acceleration."""
    nn_to_ttnn = {
        nn.Linear: TTNNLinearLLama,
        nn.SiLU: TTNNSilu,
        Glm4MoeRMSNorm: TTNNRMSNorm,
    }
    nn_to_ttnn_2 = {
        nn.Linear: TTNNLinear,
    }
    nn_to_ttnn_attention = get_attention_mappings()
    nn_to_ttnn_router = get_router_mapping()
    nn_to_ttnn_naive_moe = get_naive_moe_mapping()
    nn_to_ttnn_glm_moe_mlp = get_rewritten_glm_moe_mlp()

    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5-Air")
    model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.5-Air")
    messages = [
        {
            "role": "user",
            "content": "What is your favorite condiment? There are so many condiments to choose from, each bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe you enjoy the tangy zest of salsa or the smooth and savory taste of aioli. Share what your favorite condiment is and why you love it. Does it remind you of a specific dish or meal?",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    modules0 = register_module_replacement_dict(model, nn_to_ttnn_glm_moe_mlp, model_config=None)
    persistent_weights = set(
        [
            "lm_head",
            "model.layers.23.self_attn.q_proj",
            "model.layers.18.self_attn.q_proj",
            "model.layers.27.self_attn.q_proj",
            "model.layers.44.self_attn.q_proj",
            "model.layers.40.self_attn.q_proj",
            "model.layers.19.self_attn.q_proj",
            "model.layers.28.self_attn.q_proj",
            "model.layers.34.self_attn.q_proj",
            "model.layers.42.self_attn.q_proj",
            "model.layers.32.self_attn.q_proj",
            "model.layers.29.self_attn.q_proj",
            "model.layers.38.self_attn.q_proj",
            "model.layers.11.self_attn.q_proj",
            "model.layers.24.self_attn.q_proj",
            "model.layers.35.self_attn.q_proj",
            "model.layers.37.self_attn.q_proj",
            "model.layers.14.self_attn.q_proj",
            "model.layers.26.self_attn.q_proj",
            "model.layers.36.self_attn.q_proj",
            "model.layers.22.self_attn.q_proj",
            "model.layers.30.self_attn.q_proj",
            "model.layers.20.self_attn.q_proj",
            "model.layers.39.self_attn.q_proj",
            "model.layers.33.self_attn.q_proj",
            "model.layers.21.self_attn.q_proj",
            "model.layers.9.self_attn.q_proj",
            "model.layers.13.self_attn.q_proj",
            "model.layers.12.self_attn.q_proj",
            "model.layers.45.self_attn.q_proj",
            "model.layers.31.self_attn.q_proj",
            "model.layers.17.self_attn.q_proj",
            "model.layers.25.self_attn.q_proj",
            "model.layers.16.self_attn.q_proj",
            "model.layers.41.self_attn.q_proj",
            "model.layers.43.self_attn.q_proj",
            "model.layers.15.self_attn.q_proj",
            "model.layers.10.self_attn.q_proj",
            "model.layers.33.self_attn.o_proj",
            "model.layers.38.self_attn.o_proj",
            "model.layers.3.self_attn.q_proj",
            "model.layers.41.self_attn.o_proj",
            "model.layers.43.self_attn.o_proj",
            "model.layers.8.self_attn.o_proj",
            "model.layers.30.self_attn.o_proj",
            "model.layers.13.self_attn.o_proj",
            "model.layers.16.self_attn.o_proj",
            "model.layers.7.self_attn.q_proj",
            "model.layers.18.self_attn.o_proj",
            "model.layers.6.self_attn.q_proj",
            "model.layers.10.self_attn.o_proj",
            "model.layers.27.self_attn.o_proj",
            "model.layers.8.self_attn.q_proj",
            "model.layers.44.self_attn.o_proj",
            "model.layers.25.self_attn.o_proj",
            "model.layers.39.self_attn.o_proj",
            "model.layers.20.self_attn.o_proj",
            "model.layers.5.self_attn.q_proj",
            "model.layers.19.self_attn.o_proj",
            "model.layers.17.self_attn.o_proj",
            "model.layers.14.self_attn.o_proj",
            "model.layers.12.self_attn.o_proj",
            "model.layers.32.self_attn.o_proj",
            "model.layers.4.self_attn.q_proj",
            "model.layers.26.self_attn.o_proj",
            "model.layers.37.self_attn.o_proj",
            "model.layers.34.self_attn.o_proj",
            "model.layers.31.self_attn.o_proj",
            "model.layers.24.self_attn.o_proj",
            "model.layers.11.self_attn.o_proj",
            "model.layers.0.self_attn.q_proj",
            "model.layers.29.self_attn.o_proj",
            "model.layers.36.self_attn.o_proj",
            "model.layers.23.self_attn.o_proj",
            "model.layers.22.self_attn.o_proj",
            "model.layers.7.self_attn.o_proj",
            "model.layers.35.self_attn.o_proj",
            "model.layers.42.self_attn.o_proj",
            "model.layers.9.self_attn.o_proj",
            "model.layers.28.self_attn.o_proj",
            "model.layers.45.self_attn.o_proj",
            "model.layers.21.self_attn.o_proj",
            "model.layers.1.self_attn.o_proj",
            "model.layers.40.self_attn.o_proj",
            "model.layers.15.self_attn.o_proj",
            "model.layers.3.self_attn.o_proj",
            "model.layers.6.self_attn.o_proj",
            "model.layers.2.self_attn.o_proj",
            "model.layers.2.self_attn.q_proj",
            "model.layers.5.self_attn.o_proj",
            "model.layers.1.self_attn.q_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.4.self_attn.o_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.19.self_attn.k_proj",
            "model.layers.43.self_attn.k_proj",
            "model.layers.45.self_attn.k_proj",
            "model.layers.9.self_attn.k_proj",
            "model.layers.24.self_attn.v_proj",
            "model.layers.20.self_attn.v_proj",
            "model.layers.4.self_attn.k_proj",
            "model.layers.7.self_attn.k_proj",
            "model.layers.17.self_attn.k_proj",
            "model.layers.11.self_attn.k_proj",
            "model.layers.16.self_attn.k_proj",
            "model.layers.38.self_attn.k_proj",
            "model.layers.23.self_attn.k_proj",
            "model.layers.24.self_attn.k_proj",
            "model.layers.15.self_attn.k_proj",
            "model.layers.12.self_attn.k_proj",
            "model.layers.10.self_attn.k_proj",
            "model.layers.14.self_attn.k_proj",
            "model.layers.26.self_attn.k_proj",
            "model.layers.20.self_attn.k_proj",
            "model.layers.12.self_attn.v_proj",
            "model.layers.13.self_attn.k_proj",
            "model.layers.27.self_attn.k_proj",
            "model.layers.18.self_attn.k_proj",
            "model.layers.30.self_attn.k_proj",
            "model.layers.28.self_attn.k_proj",
            "model.layers.25.self_attn.k_proj",
            "model.layers.22.self_attn.k_proj",
            "model.layers.34.self_attn.k_proj",
            "model.layers.19.self_attn.v_proj",
            "model.layers.32.self_attn.k_proj",
            "model.layers.11.self_attn.v_proj",
            "model.layers.33.self_attn.k_proj",
            "model.layers.29.self_attn.k_proj",
            "model.layers.35.self_attn.k_proj",
            "model.layers.21.self_attn.k_proj",
            "model.layers.34.self_attn.v_proj",
            "model.layers.17.self_attn.v_proj",
            "model.layers.35.self_attn.v_proj",
            "model.layers.37.self_attn.k_proj",
            "model.layers.8.self_attn.k_proj",
            "model.layers.28.self_attn.v_proj",
            "model.layers.25.self_attn.v_proj",
            "model.layers.36.self_attn.k_proj",
            "model.layers.21.self_attn.v_proj",
            "model.layers.42.self_attn.k_proj",
            "model.layers.15.self_attn.v_proj",
            "model.layers.39.self_attn.k_proj",
            "model.layers.23.self_attn.v_proj",
            "model.layers.16.self_attn.v_proj",
            "model.layers.18.self_attn.v_proj",
            "model.layers.26.self_attn.v_proj",
            "model.layers.31.self_attn.k_proj",
            "model.layers.13.self_attn.v_proj",
            "model.layers.40.self_attn.k_proj",
            "model.layers.27.self_attn.v_proj",
            "model.layers.41.self_attn.k_proj",
            "model.layers.31.self_attn.v_proj",
            "model.layers.38.self_attn.v_proj",
            "model.layers.10.self_attn.v_proj",
            "model.layers.22.self_attn.v_proj",
            "model.layers.36.self_attn.v_proj",
            "model.layers.14.self_attn.v_proj",
            "model.layers.37.self_attn.v_proj",
            "model.layers.33.self_attn.v_proj",
            "model.layers.29.self_attn.v_proj",
            "model.layers.30.self_attn.v_proj",
            "model.layers.32.self_attn.v_proj",
            "model.layers.44.self_attn.k_proj",
            "model.layers.5.self_attn.k_proj",
            "model.layers.8.self_attn.v_proj",
            "model.layers.40.self_attn.v_proj",
            "model.layers.7.self_attn.v_proj",
            "model.layers.2.self_attn.v_proj",
            "model.layers.6.self_attn.k_proj",
            "model.layers.4.self_attn.v_proj",
            "model.layers.2.self_attn.k_proj",
            "model.layers.9.self_attn.v_proj",
            "model.layers.3.self_attn.k_proj",
            "model.layers.45.self_attn.v_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.43.self_attn.v_proj",
            "model.layers.42.self_attn.v_proj",
            "model.layers.1.self_attn.k_proj",
            "model.layers.39.self_attn.v_proj",
            "model.layers.41.self_attn.v_proj",
            "model.layers.5.self_attn.v_proj",
            "model.layers.44.self_attn.v_proj",
            "model.layers.1.self_attn.v_proj",
            "model.layers.6.self_attn.v_proj",
            "model.layers.3.self_attn.v_proj",
        ]
    )
    modules1 = register_module_replacement_dict(
        model, nn_to_ttnn, model_config=None, exclude_replacement=persistent_weights
    )
    modules2 = register_module_replacement_dict(model, nn_to_ttnn_2, model_config=None)
    modules3 = {}
    if nn_to_ttnn_attention:
        modules3 = register_module_replacement_dict(model, nn_to_ttnn_attention, model_config=None)

    modules4 = {}
    if nn_to_ttnn_router:
        modules4 = register_module_replacement_dict(model, nn_to_ttnn_router, model_config=None)
    modules5 = {}
    if nn_to_ttnn_naive_moe:
        modules5 = register_module_replacement_dict(model, nn_to_ttnn_naive_moe, model_config=None)
    set_device(model, mesh_device)
    all_modules = {**modules0, **modules1, **modules3, **modules4, **modules5}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        if not isinstance(v, TTNNLinearLLama):
            v.move_weights_to_device()
    print("Running inference...")
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    outputs = model.generate(**inputs, max_new_tokens=1, use_cache=True)
    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    print(f"GLM OUTPUT: {tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])}")
    DispatchManager.save_stats_to_file("glm_timing_stats.csv")
