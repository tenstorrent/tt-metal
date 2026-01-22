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
from models.experimental.tt_symbiote.modules.attention import TTNNSDPAAttention
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearLLama
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

import transformers

assert transformers.__version__.startswith("5."), "This test requires transformers version 5.0.0.dev0"


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

            query_states = self.q_proj(hidden_states).reshape(hidden_shape)
            key_states = self.k_proj(hidden_states).reshape(hidden_shape)
            value_states = self.v_proj(hidden_states).reshape(hidden_shape)

            if self.use_qk_norm:  # main diff from Llama
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

    return {OriginalGlm4MoeAttention: Glm4MoeAttention}


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
    }
    nn_to_ttnn_2 = {
        nn.Linear: TTNNLinear,
    }
    nn_to_ttnn_attention = get_attention_mappings()
    nn_to_ttnn_router = get_router_mapping()
    nn_to_ttnn_naive_moe = get_naive_moe_mapping()

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
    exclude_list = set(
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
    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None, exclude_replacement=exclude_list)
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
    all_modules = {**modules1, **modules2, **modules3, **modules4, **modules5}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        if k in exclude_list:
            v.move_weights_to_device()
    print("Running inference...")
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    print(f"GLM OUTPUT: {tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])}")
    DispatchManager.save_stats_to_file("glm_timing_stats.csv")
