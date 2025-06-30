# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from . import utils
from .attention import Attention, AttentionParameters
from .feed_forward import FeedForward, FeedForwardParameters
from .linear import Linear, LinearParameters
from .normalization import LayerNorm, LayerNormParameters
from .substate import substate

if TYPE_CHECKING:
    import torch


@dataclass
class TransformerBlockParameters:
    dual_attn: AttentionParameters
    prompt_time_embed: LinearParameters
    spatial_time_embed: LinearParameters
    prompt_norm_1: LayerNormParameters
    prompt_norm_2: LayerNormParameters
    spatial_norm_1: LayerNormParameters
    spatial_norm_2: LayerNormParameters
    prompt_ff: FeedForwardParameters
    spatial_ff: FeedForwardParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        linear_on_host: bool = False,
    ) -> TransformerBlockParameters:
        embedding_dim = state["norm1.linear.weight"].shape[1]

        def norm(state: dict[str, torch.Tensor]) -> LayerNormParameters:
            return LayerNormParameters.from_torch(
                state,
                dtype=dtype,
                device=device,
                weight_shape=[embedding_dim],
            )

        def linear(state: dict[str, torch.Tensor], chunks: int) -> LinearParameters:
            return LinearParameters.from_torch(
                state,
                dtype=dtype,
                device=device,
                unsqueeze_bias=True,
                on_host=linear_on_host,
                mesh_sharding_dim=1,
                chunks=chunks,
            )

        def ff(state: dict[str, torch.Tensor]) -> FeedForwardParameters:
            return FeedForwardParameters.from_torch(
                state,
                dtype=dtype,
                device=device,
                linear_on_host=linear_on_host,
                mesh_sharded_input=True,
            )

        return cls(
            dual_attn=AttentionParameters.from_torch(substate(state, "attn"), dtype=dtype, device=device),
            spatial_norm_1=norm(substate(state, "norm1.norm")),
            spatial_norm_2=norm(substate(state, "norm2")),
            prompt_norm_1=norm(substate(state, "norm1_context.norm")),
            prompt_norm_2=norm({}),
            spatial_time_embed=linear(substate(state, "norm1.linear"), chunks=6),
            prompt_time_embed=linear(substate(state, "norm1_context.linear"), chunks=6),
            spatial_ff=ff(substate(state, "ff")),
            prompt_ff=ff(substate(state, "ff_context")),
        )


class TransformerBlock:
    def __init__(
        self,
        parameters: TransformerBlockParameters,
        *,
        num_heads: int,
    ) -> None:
        eps = 1e-6

        self._dual_attn = Attention(parameters.dual_attn, num_heads=num_heads)

        self._spatial_norm_1 = LayerNorm(parameters.spatial_norm_1, eps=eps)
        self._spatial_norm_2 = LayerNorm(parameters.spatial_norm_2, eps=eps)
        self._prompt_norm_1 = LayerNorm(parameters.prompt_norm_1, eps=eps)
        self._prompt_norm_2 = LayerNorm(parameters.prompt_norm_2, eps=eps)

        self._spatial_ff = FeedForward(parameters.spatial_ff)
        self._prompt_ff = FeedForward(parameters.prompt_ff)

        self._spatial_time_embed = Linear(parameters.spatial_time_embed)
        self._prompt_time_embed = Linear(parameters.prompt_time_embed)

    def _dual_attn_block(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        spatial_gate: ttnn.Tensor,
        prompt_gate: ttnn.Tensor | None,
        prompt_scale: ttnn.Tensor,
        prompt_shift: ttnn.Tensor,
        spatial_scale: ttnn.Tensor,
        spatial_shift: ttnn.Tensor,
        image_rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor] | None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        spatial_scaled = spatial * (1 + spatial_scale) + spatial_shift

        prompt_scaled = prompt * (1 + prompt_scale) + prompt_shift

        spatial_attn, prompt_attn = self._dual_attn.forward(
            spatial=spatial_scaled, prompt=prompt_scaled, image_rotary_emb=image_rotary_emb
        )
        del spatial_scaled, prompt_scaled, image_rotary_emb

        utils.signpost("postprocess dual attention")

        spatial_attn_scaled = spatial_gate * spatial_attn
        prompt_attn_scaled = prompt_gate * prompt_attn if prompt_gate is not None else None

        return spatial_attn_scaled, prompt_attn_scaled

    def _spatial_ff_block(
        self, inp: ttnn.Tensor, *, gate: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor
    ) -> ttnn.Tensor:
        scaled = inp * (1 + scale) + shift
        return gate * self._spatial_ff.forward(scaled)

    def _prompt_ff_block(
        self, inp: ttnn.Tensor, *, gate: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor
    ) -> ttnn.Tensor:
        scaled = inp * (1 + scale) + shift
        return gate * self._prompt_ff.forward(scaled)

    def forward(  # noqa: PLR0915
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        time_embed: ttnn.Tensor,
        image_rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        skip_time_embed_activation: bool = False,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        utils.signpost("transformer block")

        if not skip_time_embed_activation:
            time_embed = ttnn.silu(time_embed)

        spatial_time = self._spatial_time_embed.forward(
            time_embed,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        prompt_time = self._prompt_time_embed.forward(
            time_embed,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        del time_embed

        [
            spatial_shift_dual_attn,
            spatial_scale_dual_attn,
            spatial_gate_dual_attn,
            spatial_shift_ff,
            spatial_scale_ff,
            spatial_gate_ff,
        ] = chunk_time(spatial_time, 6)

        [
            prompt_shift_attn,
            prompt_scale_attn,
            prompt_gate_attn,
            prompt_shift_ff,
            prompt_scale_ff,
            prompt_gate_ff,
        ] = chunk_time(prompt_time, 6)

        spatial_normed = self._spatial_norm_1.forward(spatial)
        prompt_normed = self._prompt_norm_1.forward(prompt)

        spatial_normed = ttnn.clone(spatial_normed, dtype=ttnn.bfloat8_b)
        prompt_normed = ttnn.clone(prompt_normed, dtype=ttnn.bfloat8_b)

        spatial_attn, prompt_attn = self._dual_attn_block(
            spatial=spatial_normed,
            prompt=prompt_normed,
            spatial_gate=spatial_gate_dual_attn,
            prompt_gate=prompt_gate_attn,
            prompt_scale=prompt_scale_attn,
            prompt_shift=prompt_shift_attn,
            spatial_scale=spatial_scale_dual_attn,
            spatial_shift=spatial_shift_dual_attn,
            image_rotary_emb=image_rotary_emb,
        )
        del (
            prompt_normed,
            spatial_gate_dual_attn,
            prompt_gate_attn,
            prompt_scale_attn,
            prompt_shift_attn,
            spatial_scale_dual_attn,
            spatial_shift_dual_attn,
        )

        spatial += spatial_attn
        del spatial_attn

        spatial_normed = self._spatial_norm_2.forward(spatial)

        spatial_normed = ttnn.clone(spatial_normed, dtype=ttnn.bfloat8_b)

        spatial += self._spatial_ff_block(
            spatial_normed, gate=spatial_gate_ff, scale=spatial_scale_ff, shift=spatial_shift_ff
        )

        del (
            spatial_normed,
            spatial_gate_ff,
            spatial_scale_ff,
            spatial_shift_ff,
        )

        assert prompt_scale_ff is not None
        assert prompt_shift_ff is not None
        assert prompt_gate_ff is not None

        prompt += prompt_attn
        del prompt_attn

        prompt_normed = self._prompt_norm_2.forward(prompt)
        prompt += self._prompt_ff_block(
            prompt_normed, gate=prompt_gate_ff, scale=prompt_scale_ff, shift=prompt_shift_ff
        )

        return spatial, prompt


def chunk_time(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    return [t[:, :, i * size : (i + 1) * size] for i in range(count)]
