# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from models.common.lightweightmodule import LightweightModule


class TTSelfAttention(LightweightModule):
    def __init__(
        self,
        device,
        parameters,
        n_embed,
        n_head,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=None,
        use_optimized=True,
    ):
        self.parameters = parameters
        self.device = device
        self.n_embed = n_embed
        self.n_head = n_head
        self.dtype = dtype
        self.memory_config = memory_config
        self.compute_kernel_config = compute_kernel_config
        self.use_optimized = use_optimized

    def forward(self, x):
        B, T, C = x.shape

        if self.use_optimized:
            # Optimized version: Use ttnn.transformer functions
            # Ensure input is in tile layout
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

            # Fused QKV projection
            query_key_value = ttnn.linear(
                x,
                self.parameters["query_key_value"]["weight"],
                bias=self.parameters["query_key_value"]["bias"],
                memory_config=self.memory_config,
                dtype=self.dtype,
                core_grid=ttnn.CoreGrid(y=B, x=8),
            )

            # Split QKV and split heads
            query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
                query_key_value, memory_config=ttnn.DRAM_MEMORY_CONFIG, num_heads=self.n_head, transpose_key=False
            )
            ttnn.deallocate(query_key_value)

            # Scaled dot product attention - single fused operation
            attn_output = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=False,
                scale=None,  # Will use default 1/sqrt(head_dim)
                program_config=None,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=self.memory_config,
            )

            ttnn.deallocate(query)
            ttnn.deallocate(key)
            ttnn.deallocate(value)

            # Concatenate heads
            y = ttnn.transformer.concatenate_heads(
                attn_output,
                memory_config=self.memory_config,
            )
            ttnn.deallocate(attn_output)

            if y.shape[-1] != self.n_embed:
                # slice and remove padding
                y = ttnn.to_torch(y)[..., : self.n_embed]
                y = ttnn.from_torch(
                    y,
                    device=self.device,
                    dtype=self.dtype,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                )
        else:
            # Old version from commit 8ceddd60: Separate Q, K, V and manual attention
            query = ttnn.linear(
                x,
                self.parameters["query"]["weight"],
                bias=self.parameters["query"]["bias"],
                memory_config=self.memory_config,
                core_grid=ttnn.CoreGrid(x=8, y=8),
                dtype=self.dtype,
            )
            key = ttnn.linear(
                x,
                self.parameters["key"]["weight"],
                bias=self.parameters["key"]["bias"],
                compute_kernel_config=self.compute_kernel_config,
                memory_config=self.memory_config,
                core_grid=ttnn.CoreGrid(x=8, y=8),
                dtype=self.dtype,
            )
            value = ttnn.linear(
                x,
                self.parameters["value"]["weight"],
                bias=self.parameters["value"]["bias"],
                memory_config=self.memory_config,
                core_grid=ttnn.CoreGrid(x=8, y=8),
                dtype=self.dtype,
            )
            head_dim = C // self.n_head
            query = ttnn.reshape(query, (B, T, self.n_head, head_dim))
            query = ttnn.transpose(query, 1, 2)  # (B, self.n_head, T, head_dim)

            key = ttnn.reshape(key, (B, T, self.n_head, head_dim))
            key = ttnn.transpose(key, 1, 2)  # (B, self.n_head, T, head_dim)
            key = ttnn.transpose(key, -2, -1)  # (B, self.n_head, head_dim, T)

            att = ttnn.matmul(
                query,
                key,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=self.memory_config,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                dtype=self.dtype,
            )
            value = ttnn.reshape(value, (B, T, self.n_head, head_dim))
            value = ttnn.transpose(value, 1, 2)
            dim_size = float(key.shape[-1])
            sqrt_dim = math.sqrt(dim_size)
            scale = 1.0 / sqrt_dim
            att = ttnn.multiply(att, scale, memory_config=self.memory_config, dtype=self.dtype)
            att = ttnn.softmax(att, dim=-1, memory_config=self.memory_config)
            y = ttnn.matmul(
                att,
                value,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=self.memory_config,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                dtype=self.dtype,
            )

            y = ttnn.transpose(y, 1, 2)
            y = ttnn.reshape(y, (B, T, C))

            # Convert to tile layout for linear operation
            y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)

        # Output projection (same for both versions)
        y = ttnn.linear(
            y,
            self.parameters["proj"]["weight"],
            bias=self.parameters["proj"]["bias"],
            compute_kernel_config=self.compute_kernel_config,
            memory_config=self.memory_config,
            core_grid=ttnn.CoreGrid(y=B, x=8) if self.use_optimized else ttnn.CoreGrid(y=8, x=8),
            dtype=self.dtype,
        )

        return y
