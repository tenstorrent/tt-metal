# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_cross_attention import cross_attention
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_feedforward import feedforward
from models.utility_functions import comp_pcc
import torch


def compare(tensor, name, permute=False):
    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
    tensor = ttnn.from_device(tensor)
    tensor = ttnn.to_torch(tensor)

    golden = torch.load(name)
    if permute:
        golden = golden.permute(0, 2, 3, 1)
        golden = golden.reshape(tensor.shape)

    while len(tensor.shape) > len(golden.shape):
        golden = golden.unsqueeze(0)
    while len(golden.shape) > len(tensor.shape):
        tensor = tensor.unsqueeze(0)

    passed, message = comp_pcc(tensor, golden, 0.95)
    print(f"Maches on {name}: {passed} with message {message}, tensor shape: {tensor.shape}")


class basic_transformer_block:
    def __init__(self, device, parameters):
        self.device = device
        self.parameters = parameters
        self.cross_attention_1 = cross_attention(device, self.parameters.attn1)
        self.cross_attention_2 = cross_attention(device, self.parameters.attn2)
        self.ff = feedforward(device, parameters=self.parameters.ff)

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        class_labels=None,
        config=None,
        num_embeds_ada_norm=False,
        norm_type: str = "layer_norm",
        cross_attention_dim: int = None,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        attention_bias: bool = False,
        attention_head_dim=None,
    ):
        use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if use_ada_layer_norm:
            assert False, "AdaLayerNorm not supported and not used in stable diffusion"
        elif use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_memory_config(
            hidden_states, ttnn.L1_MEMORY_CONFIG
        )  # layernorm doesn't support block_sharding
        norm_hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=self.parameters.norm1.weight,
            bias=self.parameters.norm1.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        cross_attention_dim = config.cross_attention_dim if cross_attention_dim is None else cross_attention_dim

        attn_output = self.cross_attention_1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if only_cross_attention else None,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            cross_attention_dim=cross_attention_dim,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )

        if use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"

        hidden_states = ttnn.add(attn_output, hidden_states)
        if cross_attention_dim is not None:
            norm_hidden_states = ttnn.layer_norm(
                hidden_states, epsilon=1e-05, weight=self.parameters.norm2.weight, bias=self.parameters.norm2.bias
            )

            # 2. Cross-Attention
            attn_output = self.cross_attention_2(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                cross_attention_dim=cross_attention_dim,
                dim_head=attention_head_dim,
                upcast_attention=upcast_attention,
            )

            hidden_states = ttnn.add(attn_output, hidden_states)

        # 3. Feed-forward
        norm_hidden_states = ttnn.layer_norm(
            hidden_states, epsilon=1e-05, weight=self.parameters.norm3.weight, bias=self.parameters.norm3.bias
        )
        if use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"

        norm_hidden_states = ttnn.clone(
            norm_hidden_states, memory_config=ttnn.get_memory_config(hidden_states), dtype=ttnn.bfloat16
        )
        ff_output = self.ff(config=config, hidden_states=norm_hidden_states)

        hidden_states = ttnn.add(ff_output, hidden_states)

        return hidden_states
