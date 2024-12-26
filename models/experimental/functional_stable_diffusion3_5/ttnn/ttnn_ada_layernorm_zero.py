# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional, Tuple


class ttnn_AdaLayerNormZero:
    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        self.emb = None

        self.silu = ttnn.silu
        self.linear = ttnn.linear
        if norm_type == "layer_norm":
            self.norm = ttnn.layer_norm
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[ttnn.Tensor] = None,
        class_labels: Optional[ttnn.Tensor] = None,
        hidden_dtype=None,
        emb: Optional[ttnn.Tensor] = None,
        parameters=None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        hifi2_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)

        mm_a_y = 8
        mm_a_x = 8
        mm_a_x_strategy = ttnn.ShardStrategy.WIDTH
        mm_a_x_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

        # emb = ttnn.reshape(emb, (emb.shape[0], 1, emb.shape[1]))

        emb = self.linear(
            self.silu(emb),
            parameters["linear"]["weight"],
            bias=parameters["linear"]["bias"],
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
            compute_kernel_config=hifi2_kernel_config,
        )
        emb = ttnn.to_memory_config(emb, ttnn.L1_MEMORY_CONFIG)
        one_chunk = emb.shape[-1] // 6

        i_beg = 0
        i_end = one_chunk
        shift_msa = ttnn.slice(emb, [0, 0, 0, i_beg], [2, 1, 1, i_end])
        i_beg += one_chunk
        i_end += one_chunk
        scale_msa = ttnn.slice(emb, [0, 0, 0, i_beg], [2, 1, 1, i_end])
        i_beg += one_chunk
        i_end += one_chunk
        gate_msa = ttnn.slice(emb, [0, 0, 0, i_beg], [2, 1, 1, i_end])
        i_beg += one_chunk
        i_end += one_chunk
        shift_mlp = ttnn.slice(emb, [0, 0, 0, i_beg], [2, 1, 1, i_end])
        i_beg += one_chunk
        i_end += one_chunk
        scale_mlp = ttnn.slice(emb, [0, 0, 0, i_beg], [2, 1, 1, i_end])
        i_beg += one_chunk
        i_end += one_chunk
        gate_mlp = ttnn.slice(emb, [0, 0, 0, i_beg], [2, 1, 1, i_end])

        ttnn.deallocate(emb)

        # x = self.norm(x, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi2_kernel_config)
        # x = x * (1 + scale_msa) + shift_msa

        norm_hidden_states = self.norm(hidden_states, compute_kernel_config=hifi2_kernel_config)
        scale_msa = scale_msa + 1
        # TODO: can we shard the hidden state but keep scale tensor as interleaved?
        norm_hidden_states = norm_hidden_states * scale_msa
        norm_hidden_states = norm_hidden_states + shift_msa

        return norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp
