# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from . import utils
from .attention import Attention, AttentionParameters
from .linear import Linear, LinearParameters
from .normalization import LayerNorm, LayerNormParameters
from .substate import substate
from .transformer_block import chunk_time


@dataclass
class FluxSingleTransformerBlockParameters:
    attn: AttentionParameters
    norm: LayerNormParameters
    time_embed: LinearParameters
    proj_mlp: LinearParameters
    proj_out: LinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        linear_on_host: bool = False,
    ) -> FluxSingleTransformerBlockParameters:
        _, mesh_width = device.shape
        embedding_dim = state["norm.linear.weight"].shape[1]

        return cls(
            attn=AttentionParameters.from_torch(substate(state, "attn"), dtype=dtype, device=device),
            norm=LayerNormParameters.from_torch(
                substate(state, "norm"),
                dtype=dtype,
                device=device,
                weight_shape=[embedding_dim],
            ),
            time_embed=LinearParameters.from_torch(
                substate(state, "norm.linear"),
                dtype=dtype,
                device=device,
                unsqueeze_bias=True,
                mesh_sharding_dim=1,
                chunks=3,
            ),
            proj_mlp=LinearParameters.from_torch(
                substate(state, "proj_mlp"),
                dtype=dtype,
                device=device,
                on_host=linear_on_host,
                mesh_sharding_dim=0,
            ),
            proj_out=LinearParameters.from_torch(
                _re_fuse_proj_out_parameters(
                    substate(state, "proj_out"),
                    embedding_dim=embedding_dim,
                    device_count=mesh_width,
                ),
                dtype=dtype,
                device=device,
                on_host=linear_on_host,
                mesh_sharding_dim=0,
            ),
        )


class FluxSingleTransformerBlock:
    def __init__(
        self,
        parameters: FluxSingleTransformerBlockParameters,
        *,
        num_heads: int,
    ) -> None:
        self._attn = Attention(parameters.attn, num_heads=num_heads)
        self._norm = LayerNorm(parameters.norm, eps=1e-6)
        self._time_embed = Linear(parameters.time_embed)

        self._proj_mlp = Linear(parameters.proj_mlp)
        self._proj_out = Linear(parameters.proj_out)

    def forward(
        self,
        *,
        combined: ttnn.Tensor,
        time_embed: ttnn.Tensor,
        image_rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        skip_time_embed_activation: bool = False,
    ) -> ttnn.Tensor:
        utils.signpost("single transformer block")

        if not skip_time_embed_activation:
            time_embed = ttnn.silu(time_embed)
        time = self._time_embed.forward(time_embed)

        combined_normed = self._norm.forward(combined)
        combined_normed = ttnn.clone(combined_normed, dtype=ttnn.bfloat8_b)

        shift_msa, scale_msa, gate_msa = chunk_time(time, 3)
        norm_combined = combined_normed * (1 + scale_msa) + shift_msa

        mlp_combined = self._proj_mlp.forward(norm_combined)
        # Fusing the activation function currently gives worse PCC
        ttnn.gelu(mlp_combined, output_tensor=mlp_combined, fast_and_approximate_mode=False)

        # PCC of attn seems a bit low
        attn, _ = self._attn.forward(spatial=norm_combined, image_rotary_emb=image_rotary_emb)
        del norm_combined

        utils.signpost("postprocess combined attention")

        additional = ttnn.concat([attn, mlp_combined], dim=-1)
        additional = self._proj_out.forward(additional)
        additional = gate_msa * additional

        combined += additional

        return combined
        # return ttnn.clamp(combined, -65504, 65504)  # clamp gives worse PCC


def _re_fuse_proj_out_parameters(
    state: dict[str, torch.Tensor],
    *,
    embedding_dim: int,
    device_count: int,
) -> dict[str, torch.Tensor]:
    """
    The out-projection layer gets as inputs fused activations coming from the attention network and
    the MLP. In order to get the correct behavior on a mesh device, its weights must be re-fused to
    take into account mesh sharding.
    """
    if device_count == 1:
        return state

    w = state["weight"]
    _, in_dim = w.shape

    in_dim1 = embedding_dim
    in_dim2 = in_dim - in_dim1

    # unfuse
    w1, w2 = w.split([in_dim1, in_dim2], dim=-1)

    # re-fuse
    w1 = w1.reshape([-1, device_count, in_dim1 // device_count])
    w2 = w2.reshape([-1, device_count, in_dim2 // device_count])
    w = torch.cat([w1, w2], dim=-1).reshape([-1, in_dim])

    return {"weight": w, "bias": state["bias"]}
