# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_configs import (
    GROUPNORM_EPSILON,
    GROUPNORM_GROUPS,
    get_default_compute_config,
)
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_utils import prepare_group_norm


class Attention:
    def __init__(self, torch_attention, device, in_channels):
        self.device = device
        self.heads = torch_attention.heads
        self.compute_config = get_default_compute_config(device)

        # groupnorm
        self.norm_num_blocks = 1
        self.norm_grid_core = ttnn.CoreGrid(y=4, x=8) if in_channels == 128 else ttnn.CoreGrid(y=8, x=8)
        (
            self.norm_input_mask,
            self.norm_weights,
            self.norm_bias,
        ) = prepare_group_norm(
            self.device,
            in_channels,
            self.norm_grid_core,
            torch_attention.group_norm.weight,
            torch_attention.group_norm.bias,
        )

        # query
        self.query_weights = ttnn.from_torch(
            torch_attention.to_q.weight.permute(1, 0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
        )
        self.query_bias = ttnn.from_torch(
            torch_attention.to_q.bias,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
        )

        # key
        self.key_weights = ttnn.from_torch(
            torch_attention.to_k.weight.permute(1, 0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
        )
        self.key_bias = ttnn.from_torch(
            torch_attention.to_k.bias,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
        )

        # value
        self.value_weights = ttnn.from_torch(
            torch_attention.to_v.weight.permute(1, 0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
        )
        self.value_bias = ttnn.from_torch(
            torch_attention.to_v.bias,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
        )

        # SDPA
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        # linear out
        self.out_weights = ttnn.from_torch(
            torch_attention.to_out[0].weight.permute(1, 0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
        )
        self.out_bias = ttnn.from_torch(
            torch_attention.to_out[0].bias,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
        )

    def __call__(self, input_tensor):
        if input_tensor.shape[1] != 1:
            input_tensor = ttnn.reshape(
                input_tensor, [1, 1, input_tensor.shape[1] * input_tensor.shape[1], input_tensor.shape[3]]
            )
        hidden_states = input_tensor

        # prepare groupnorm
        if hidden_states.dtype != ttnn.bfloat16:
            hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=GROUPNORM_GROUPS,
            input_mask=self.norm_input_mask,
            weight=self.norm_weights,
            bias=self.norm_bias,
            epsilon=GROUPNORM_EPSILON,
            core_grid=self.norm_grid_core,
            dtype=ttnn.bfloat16,
            inplace=False,
            num_out_blocks=self.norm_num_blocks,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        query = ttnn.linear(
            hidden_states,
            self.query_weights,
            bias=self.query_bias,
        )
        key = ttnn.linear(
            hidden_states,
            self.key_weights,
            bias=self.key_bias,
        )
        value = ttnn.linear(
            hidden_states,
            self.value_weights,
            bias=self.value_bias,
        )

        # prepare SDPA
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = ttnn.reshape(query, [1, -1, self.heads, head_dim])
        query = ttnn.transpose(query, 1, 2)

        key = ttnn.reshape(key, [1, -1, self.heads, head_dim])
        key = ttnn.transpose(key, 1, 2)

        value = ttnn.reshape(value, [1, -1, self.heads, head_dim])
        value = ttnn.transpose(value, 1, 2)

        hidden_states = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
            attn_mask=None,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.compute_config,
        )
        hidden_states = ttnn.transpose(hidden_states, 1, 2)
        hidden_states = ttnn.reshape(hidden_states, [1, -1, self.heads * head_dim])

        hidden_states = ttnn.linear(
            hidden_states,
            self.out_weights,
            bias=self.out_bias,
        )

        return ttnn.add(hidden_states, input_tensor)
