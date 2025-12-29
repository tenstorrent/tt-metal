# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import re

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.tt_transformerblock import TtBasicTransformerBlock
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_linear_params,
)


class TtTransformer2DModel(LightweightModule):
    def __init__(self, device, state_dict, module_path, model_config, query_dim, num_attn_heads, out_dim):
        super().__init__()

        self.device = device

        self.norm_groups = 32
        self.norm_eps = 1e-6

        pattern = re.compile(rf"^{re.escape(module_path)}\.transformer_blocks\.(\d+)")
        transformer_blocks = set(int(match.group(1)) for key in state_dict.keys() if (match := pattern.match(key)))
        self.num_layers = len(transformer_blocks)

        self.transformer_blocks = []
        for i in range(self.num_layers):
            self.transformer_blocks.append(
                TtBasicTransformerBlock(
                    device,
                    state_dict,
                    f"{module_path}.transformer_blocks.{i}",
                    model_config,
                    query_dim,
                    num_attn_heads,
                    out_dim,
                )
            )

        norm_weights = state_dict[f"{module_path}.norm.weight"]
        norm_bias = state_dict[f"{module_path}.norm.bias"]
        (
            self.groupnorm_config,
            self.groupnorm_memory_config,
            self.input_mask,
            self.input_negative_mask,
            self.gamma_t,
            self.beta_t,
        ) = model_config.get_groupnorm_params(f"{module_path}.norm", norm_weights, norm_bias, self.norm_groups, device)
        assert (
            self.groupnorm_memory_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
            or self.groupnorm_memory_config == ttnn.DRAM_MEMORY_CONFIG
        ), "Only L1_BLOCK_SHARDED_MEMORY_CONFIG and DRAM_MEMORY_CONFIG is supported for GN"

        proj_weights_dtype = (
            model_config.attention_weights_dtype
        )  # keep this same as attention weights dtype at the moment
        weights = state_dict[f"{module_path}.proj_in.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.proj_in.bias"]
        self.tt_weights_in, self.tt_bias_in = prepare_linear_params(device, weights, bias, proj_weights_dtype)

        weights = state_dict[f"{module_path}.proj_out.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.proj_out.bias"]
        self.tt_weights_out, self.tt_bias_out = prepare_linear_params(device, weights, bias, proj_weights_dtype)

        self.program_config_in = model_config.get_matmul_config(matmul_path=f"{module_path}.proj_in")
        self.compute_config_in = model_config.get_mm_compute_config(f"{module_path}.proj_in")
        self.memory_config_in = model_config.get_mm_output_memory_config(f"{module_path}.proj_in")
        self.program_config_out = model_config.get_matmul_config(matmul_path=f"{module_path}.proj_out")
        self.compute_config_out = model_config.get_mm_compute_config(f"{module_path}.proj_out")
        self.memory_config_out = model_config.get_mm_output_memory_config(f"{module_path}.proj_out")

    def forward(self, input_tensor, input_shape, attention_mask=None, encoder_hidden_states=None):
        B, C, H, W = input_shape

        hidden_states = input_tensor

        mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        if self.groupnorm_memory_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
            mem_cfg = ttnn.create_sharded_memory_config(
                shape=hidden_states.shape,
                core_grid=self.groupnorm_config["core_grid"],
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )

        hidden_states = ttnn.to_memory_config(hidden_states, mem_cfg)
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask,
            negative_mask=self.input_negative_mask,
            weight=self.gamma_t,
            bias=self.beta_t,
            epsilon=self.norm_eps,
            memory_config=hidden_states.memory_config(),
            **self.groupnorm_config,
        )

        if C == 1280:
            # For 1280 channels shard layout will be over 64 cores, but MM runs on 40
            # To avoid assertion error we move data to L1 interleaved
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_weights_in,
            bias=self.tt_bias_in,
            program_config=self.program_config_in,
            compute_kernel_config=self.compute_config_in,
            memory_config=self.memory_config_in,
        )

        for i, transformer_block in enumerate(self.transformer_blocks):
            hidden_states = transformer_block(hidden_states, attention_mask, encoder_hidden_states)

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_weights_out,
            bias=self.tt_bias_out,
            program_config=self.program_config_out,
            compute_kernel_config=self.compute_config_out,
            memory_config=self.memory_config_out,
        )

        hidden_states = ttnn.add(hidden_states, input_tensor, use_legacy=False)

        return hidden_states
