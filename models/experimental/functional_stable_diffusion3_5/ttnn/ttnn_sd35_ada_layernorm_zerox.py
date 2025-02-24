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
        emb = self.linear(self.silu(emb), parameters["linear"]["weight"], bias=parameters["linear"]["bias"])
        """
        emb = ttnn.to_torch(emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.chunk(
            9, dim=1
        )
        shift_msa = ttnn.from_torch(shift_msa, layout=ttnn.TILE_LAYOUT, device=hidden_states.device())
        scale_msa = ttnn.from_torch(scale_msa, layout=ttnn.TILE_LAYOUT, device=hidden_states.device())
        gate_msa = ttnn.from_torch(gate_msa, layout=ttnn.TILE_LAYOUT, device=hidden_states.device())
        shift_mlp = ttnn.from_torch(shift_mlp, layout=ttnn.TILE_LAYOUT, device=hidden_states.device())
        scale_mlp = ttnn.from_torch(scale_mlp, layout=ttnn.TILE_LAYOUT, device=hidden_states.device())
        gate_mlp = ttnn.from_torch(gate_mlp, layout=ttnn.TILE_LAYOUT, device=hidden_states.device())
        shift_msa2 = ttnn.from_torch(shift_msa2, layout=ttnn.TILE_LAYOUT, device=hidden_states.device())
        scale_msa2 = ttnn.from_torch(scale_msa2, layout=ttnn.TILE_LAYOUT, device=hidden_states.device())
        gate_msa2 = ttnn.from_torch(gate_msa2, layout=ttnn.TILE_LAYOUT, device=hidden_states.device())

        norm_hidden_states = self.norm(hidden_states)
        hidden_states = norm_hidden_states * (
            1 + ttnn.reshape(scale_msa, (scale_msa.shape[0], 1, scale_msa.shape[1]))
        ) + ttnn.reshape(shift_msa, (shift_msa.shape[0], 1, shift_msa.shape[1]))
        norm_hidden_states2 = norm_hidden_states * (
            1 + ttnn.reshape(scale_msa2, (scale_msa2.shape[0], 1, scale_msa2.shape[1]))
        ) + ttnn.reshape(shift_msa2, (shift_msa2.shape[0], 1, shift_msa2.shape[1]))
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2
        """

        emb = ttnn.to_memory_config(emb, ttnn.L1_MEMORY_CONFIG)
        one_chunk = emb.shape[-1] // 9
        emb = ttnn.permute(emb, (2, 0, 1, 3))
        # emb = ttnn.permute(emb, (1,0,2))

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

        # the following step is inserted here to save wasted memory gaps
        ttnn.deallocate(emb)
        gate_msa2 = ttnn.reallocate(gate_msa2)

        # print("zerox", hidden_states.shape)
        norm_hidden_states = self.norm(hidden_states)  # , compute_kernel_config=hifi2_kernel_config)
        scale_msa = scale_msa + 1
        scale_msa2 = scale_msa2 + 1
        out_hidden_states = norm_hidden_states * scale_msa
        out_hidden_states = out_hidden_states + shift_msa
        out_hidden_states2 = norm_hidden_states * scale_msa2
        out_hidden_states2 = out_hidden_states2 + shift_msa2
        ttnn.deallocate(norm_hidden_states)
        ttnn.deallocate(scale_msa)
        ttnn.deallocate(scale_msa2)
        out_hidden_states = ttnn.reallocate(out_hidden_states)
        out_hidden_states2 = ttnn.reallocate(out_hidden_states2)
        # print("zerox", out_hidden_states.shape)

        return out_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, out_hidden_states2, gate_msa2
