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
        x: torch.Tensor,
        timestep: Optional[ttnn.Tensor] = None,
        class_labels: Optional[ttnn.Tensor] = None,
        hidden_dtype=None,
        emb: Optional[ttnn.Tensor] = None,
        parameters=None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb), parameters["linear"]["weight"], bias=parameters["linear"]["bias"])

        """
        emb = ttnn.to_torch(emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, 1)

        shift_msa = ttnn.from_torch(shift_msa, layout=ttnn.TILE_LAYOUT, device=x.device())
        scale_msa = ttnn.from_torch(scale_msa, layout=ttnn.TILE_LAYOUT, device=x.device())
        gate_msa = ttnn.from_torch(gate_msa, layout=ttnn.TILE_LAYOUT, device=x.device())
        shift_mlp = ttnn.from_torch(shift_mlp, layout=ttnn.TILE_LAYOUT, device=x.device())
        scale_mlp = ttnn.from_torch(scale_mlp, layout=ttnn.TILE_LAYOUT, device=x.device())
        gate_mlp = ttnn.from_torch(gate_mlp, layout=ttnn.TILE_LAYOUT, device=x.device())

        x = self.norm(x) * (1 + ttnn.reshape(scale_msa, (scale_msa.shape[0], 1, scale_msa.shape[1]))) + ttnn.reshape(
            shift_msa, (shift_msa.shape[0], 1, shift_msa.shape[1])
        )  # shift_msa[:, None] replaced with ttnn.reshape(shift_msa,(shift_msa.shape[0],1,shift_msa.shape[1])) same for scale_msa[:,None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
        """

        emb = ttnn.to_memory_config(emb, ttnn.L1_MEMORY_CONFIG)
        one_chunk = emb.shape[-1] // 6

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

        ttnn.deallocate(emb)

        # print("zero", x.shape)
        norm_hidden_states = self.norm(x)  # , compute_kernel_config=hifi2_kernel_config)
        scale_msa = scale_msa + 1
        # TODO: can we shard the hidden state but keep scale tensor as interleaved?
        norm_hidden_states = norm_hidden_states * scale_msa
        norm_hidden_states = norm_hidden_states + shift_msa
        # print("zero", norm_hidden_states.shape)

        return norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp
