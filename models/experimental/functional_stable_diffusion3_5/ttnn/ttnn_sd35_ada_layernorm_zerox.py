# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from typing import Optional, Tuple
import ttnn


class ttnn_SD35AdaLayerNormZeroX:
    r"""
    Norm layer adaptive layer norm zero (AdaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True) -> None:
        self.silu = ttnn.silu
        self.linear = ttnn.linear
        if norm_type == "layer_norm":
            self.norm = ttnn.layer_norm
        else:
            raise ValueError(f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'.")

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        emb: Optional[ttnn.Tensor] = None,
        parameters=None,
    ) -> Tuple[ttnn.Tensor, ...]:
        hifi2_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )

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
        one_chunk = emb.shape[-1] // 9

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
        i_beg += one_chunk
        i_end += one_chunk
        shift_msa2 = ttnn.slice(emb, [0, 0, 0, i_beg], [2, 1, 1, i_end])
        i_beg += one_chunk
        i_end += one_chunk
        scale_msa2 = ttnn.slice(emb, [0, 0, 0, i_beg], [2, 1, 1, i_end])
        i_beg += one_chunk
        i_end += one_chunk
        gate_msa2 = ttnn.slice(emb, [0, 0, 0, i_beg], [2, 1, 1, i_end])

        ttnn.deallocate(emb)

        norm_hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(hidden_states)
        norm_hidden_states = self.norm(norm_hidden_states, compute_kernel_config=hifi2_kernel_config)
        scale_msa = scale_msa + 1
        scale_msa2 = scale_msa2 + 1
        # the following step is inserted here to save wasted memory gaps
        gate_msa2 = ttnn.reallocate(gate_msa2)
        hidden_states = norm_hidden_states * scale_msa
        hidden_states = hidden_states + shift_msa
        hidden_states2 = norm_hidden_states * scale_msa2
        hidden_states2 = hidden_states2 + shift_msa2
        ttnn.deallocate(norm_hidden_states)
        # hidden_states2 = ttnn.reallocate(hidden_states2)
        # hidden_states = ttnn.reallocate(hidden_states)

        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, hidden_states2, gate_msa2
