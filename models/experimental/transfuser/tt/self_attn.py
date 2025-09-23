# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.common.lightweightmodule import LightweightModule


class TTSelfAttention(LightweightModule):
    def __init__(self, device, parameters, n_embed, n_head, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG):
        self.parameters = parameters
        self.device = device
        self.n_embed = n_embed
        self.out_features = n_embed
        self.dtype = dtype
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG

    def forward(self, x):
        key_weight = self.parameters["key"]["weight"]
        key_bias = self.parameters["key"]["bias"]
        query_weight = self.parameters["query"]["weight"]
        query_bias = self.parameters["query"]["bias"]
        value_weight = self.parameters["value"]["weight"]
        value_bias = self.parameters["value"]["bias"]
        proj_weight = self.parameters["proj"]["weight"]
        proj_bias = self.parameters["proj"]["bias"]

        key = ttnn.linear(
            x,
            key_weight,
            bias=key_bias,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(x=8, y=8),
            dtype=self.dtype,
        )
        key = ttnn.reshape(key, (1, 174, 4, 18))
        key = ttnn.permute(key, (0, 2, 1, 3))
        query = ttnn.linear(
            x,
            query_weight,
            bias=query_bias,
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(x=8, y=8),
            dtype=self.dtype,
        )
        query = ttnn.reshape(query, (1, 174, 4, 18))
        query = ttnn.permute(query, (0, 2, 1, 3))
        value = ttnn.linear(
            x,
            value_weight,
            bias=value_bias,
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(x=8, y=8),
            dtype=self.dtype,
        )
        value = ttnn.reshape(value, (1, 174, 4, 18))
        value = ttnn.permute(value, (0, 2, 1, 3))
        # TODO: Add program_config and compute_kernel_config
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=[8, 7],
            q_chunk_size=512,
            k_chunk_size=512,
            exp_approx_mode=False,
        )
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
            scale=1.0 / math.sqrt(key.shape[-1]),  # Your scaling factor
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # program_config=program_config,
            # compute_kernel_config=compute_kernel_config
        )
        return attn_output
