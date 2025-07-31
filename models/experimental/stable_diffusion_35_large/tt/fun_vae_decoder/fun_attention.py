# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import ttnn

from .fun_linear import vae_linear, TtLinearParameters
from .fun_group_norm import vae_group_norm, TtGroupNormParameters
from ..parallel_config import VAEParallelConfig

import torch

## Parameters


# Assumptions: If input is sharded, output will be sharded. The sharding is only applied to the final linear layer.
@dataclass
class TtAttentionParameters:
    group_norm: TtGroupNormParameters
    to_q: TtLinearParameters
    to_k: TtLinearParameters
    to_v: TtLinearParameters
    to_out: list[TtLinearParameters]
    heads: int

    @classmethod
    def from_torch(
        cls,
        torch_attention: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        parallel_config: VAEParallelConfig,
        mesh_sharded_input: bool = True,
    ) -> TtAttentionParameters:
        return cls(
            group_norm=TtGroupNormParameters.from_torch(
                torch_attention.group_norm, parallel_config=parallel_config, mesh_sharded_input=mesh_sharded_input
            ),
            to_q=TtLinearParameters.from_torch(
                torch_attention.to_q, dtype=dtype, parallel_config=parallel_config, mesh_sharded_output=False
            ),
            to_k=TtLinearParameters.from_torch(
                torch_attention.to_k, dtype=dtype, parallel_config=parallel_config, mesh_sharded_output=False
            ),
            to_v=TtLinearParameters.from_torch(
                torch_attention.to_v, dtype=dtype, parallel_config=parallel_config, mesh_sharded_output=False
            ),
            to_out=[
                TtLinearParameters.from_torch(
                    linear_out, dtype=dtype, parallel_config=parallel_config, mesh_sharded_output=mesh_sharded_input
                )
                for linear_out in torch_attention.to_out
                if not isinstance(linear_out, torch.nn.Dropout)
            ],
            heads=torch_attention.heads,
        )


def reorder_for_attention(x, batch_size, n_heads, head_dim):
    return ttnn.permute(ttnn.reshape(x, (batch_size, -1, n_heads, head_dim)), (0, 2, 1, 3))


# TODO: See older vae code for optimization
def vae_attention(x: ttnn.Tensor, parameters: TtAttentionParameters) -> ttnn.Tensor:
    assert len(x.shape) == 4

    # elementwise required to be tilized
    in_layout = x.layout
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    residual = x

    [b, h, w, c] = list(x.shape)

    # No need to transpose like reference. x is alredy channel last
    x = vae_group_norm(x, parameters.group_norm)

    # output will be bxhxwx(num_heads*head_dims)
    q = vae_linear(x, parameters.to_q)
    k = vae_linear(x, parameters.to_k)
    v = vae_linear(x, parameters.to_v)
    inner_dim = k.shape[-1]
    head_dim = inner_dim // parameters.heads

    q = reorder_for_attention(q, b, parameters.heads, head_dim)
    k = reorder_for_attention(k, b, parameters.heads, head_dim)
    v = reorder_for_attention(v, b, parameters.heads, head_dim)

    x = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
    x = ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), (b, h, w, inner_dim))
    x = vae_linear(x, parameters.to_out[0])

    x = x + residual

    x = ttnn.to_layout(x, in_layout)
    return x
