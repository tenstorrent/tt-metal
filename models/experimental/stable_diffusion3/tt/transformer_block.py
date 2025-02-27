# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .attention import TtAttention, TtAttentionParameters
from .feed_forward import TtFeedForward, TtFeedForwardParameters
from .linear import TtLinear, TtLinearParameters
from .normalization import TtLayerNorm, TtLayerNormParameters
from .substate import has_substate, substate

if TYPE_CHECKING:
    import torch


@dataclass
class TtTransformerBlockParameters:
    dual_attn: TtAttentionParameters
    spatial_attn: TtAttentionParameters | None
    prompt_time_embed: TtLinearParameters
    spatial_time_embed: TtLinearParameters
    prompt_norm_1: TtLayerNormParameters
    spatial_norm_1: TtLayerNormParameters
    spatial_norm_2: TtLayerNormParameters
    prompt_ff: TtFeedForwardParameters | None
    spatial_ff: TtFeedForwardParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtTransformerBlockParameters:
        return cls(
            dual_attn=TtAttentionParameters.from_torch(substate(state, "attn"), dtype=dtype, device=device),
            spatial_attn=TtAttentionParameters.from_torch(substate(state, "attn2"), dtype=dtype, device=device)
            if has_substate(state, "attn2")
            else None,
            spatial_norm_1=TtLayerNormParameters.from_torch(substate(state, "norm1.norm"), dtype=dtype, device=device),
            spatial_norm_2=TtLayerNormParameters.from_torch(substate(state, "norm2"), dtype=dtype, device=device),
            prompt_norm_1=TtLayerNormParameters.from_torch(
                substate(state, "norm1_context.norm"), dtype=dtype, device=device
            ),
            spatial_time_embed=TtLinearParameters.from_torch(
                substate(state, "norm1.linear"), dtype=dtype, device=device, unsqueeze_bias=True
            ),
            prompt_time_embed=TtLinearParameters.from_torch(
                substate(state, "norm1_context.linear"), dtype=dtype, device=device, unsqueeze_bias=True
            ),
            spatial_ff=TtFeedForwardParameters.from_torch(substate(state, "ff"), dtype=dtype, device=device),
            prompt_ff=TtFeedForwardParameters.from_torch(substate(state, "ff_context"), dtype=dtype, device=device)
            if has_substate(state, "ff_context")
            else None,
        )


class TtTransformerBlock:
    def __init__(
        self,
        parameters: TtTransformerBlockParameters,
        *,
        num_heads: int,
    ) -> None:
        eps = 1e-6

        self._dual_attn = TtAttention(parameters.dual_attn, num_heads=num_heads)
        self._spatial_attn = (
            TtAttention(parameters.spatial_attn, num_heads=num_heads) if parameters.spatial_attn is not None else None
        )

        self._spatial_norm_1 = TtLayerNorm(parameters.spatial_norm_1, eps=eps)
        self._spatial_norm_2 = TtLayerNorm(parameters.spatial_norm_2, eps=eps)
        self._prompt_norm_1 = TtLayerNorm(parameters.prompt_norm_1, eps=eps)
        self._prompt_norm_2 = TtLayerNorm(TtLayerNormParameters(), eps=eps)

        self._spatial_ff = TtFeedForward(parameters.spatial_ff)
        self._prompt_ff = TtFeedForward(parameters.prompt_ff) if parameters.prompt_ff is not None else None

        self._spatial_time_embed = TtLinear(parameters.spatial_time_embed)
        self._prompt_time_embed = TtLinear(parameters.prompt_time_embed)

        self._context_pre_only = self._prompt_ff is None

    def _spatial_attn_block(
        self, inp: ttnn.Tensor, *, gate: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor
    ) -> ttnn.Tensor:
        assert self._spatial_attn is not None

        scaled = inp * (1 + scale) + shift
        attn, _ = self._spatial_attn(spatial=scaled, deallocate=True)

        result = gate * attn

        ttnn.deallocate(scaled)
        ttnn.deallocate(attn)
        return result

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
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        # spatial_memory_config = ttnn.create_sharded_memory_config(
        #     spatial.shape,
        #     core_grid=spatial.device().core_grid,
        #     strategy=ttnn.ShardStrategy.BLOCK,
        #     orientation=ttnn.ShardOrientation.ROW_MAJOR,
        # )
        # spatial = ttnn.to_memory_config(spatial, spatial_memory_config)

        spatial_scaled = spatial * (1 + spatial_scale) + spatial_shift
        # ttnn.deallocate(spatial)
        # spatial_scaled = ttnn.to_memory_config(spatial_scaled, ttnn.DRAM_MEMORY_CONFIG)
        # ttnn.deallocate(spatial_scaled)

        prompt_scaled = prompt * (1 + prompt_scale) + prompt_shift

        spatial_attn, prompt_attn = self._dual_attn(spatial=spatial_scaled, prompt=prompt_scaled, deallocate=True)

        spatial_attn_scaled = spatial_gate * spatial_attn
        prompt_attn_scaled = prompt_gate * prompt_attn if prompt_gate is not None else None

        ttnn.deallocate(spatial_attn)
        ttnn.deallocate(prompt_attn)
        return spatial_attn_scaled, prompt_attn_scaled

    def _spatial_ff_block(
        self, inp: ttnn.Tensor, *, gate: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor
    ) -> ttnn.Tensor:
        scaled = inp * (1 + scale) + shift
        result = gate * self._spatial_ff(scaled)
        ttnn.deallocate(scaled)
        return result

    def _prompt_ff_block(
        self, inp: ttnn.Tensor, *, gate: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor
    ) -> ttnn.Tensor:
        assert self._prompt_ff is not None

        scaled = inp * (1 + scale) + shift
        result = gate * self._prompt_ff(scaled)
        ttnn.deallocate(scaled)
        return result

    def __call__(  # noqa: PLR0915
        self, *, spatial: ttnn.Tensor, prompt: ttnn.Tensor, time_embed: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        t = ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        spatial_time = self._spatial_time_embed(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        prompt_time = self._prompt_time_embed(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(t)

        if self._spatial_attn is not None:
            [
                spatial_shift_dual_attn,
                spatial_scale_dual_attn,
                spatial_gate_dual_attn,
                spatial_shift_ff,
                spatial_scale_ff,
                spatial_gate_ff,
                spatial_shift_attn,
                spatial_scale_attn,
                spatial_gate_attn,
            ] = chunk_time(spatial_time, 9)
        else:
            [
                spatial_shift_dual_attn,
                spatial_scale_dual_attn,
                spatial_gate_dual_attn,
                spatial_shift_ff,
                spatial_scale_ff,
                spatial_gate_ff,
            ] = chunk_time(spatial_time, 6)

            spatial_gate_attn = None
            spatial_shift_attn = None
            spatial_scale_attn = None

        if self._context_pre_only:
            [
                prompt_scale_attn,
                prompt_shift_attn,
            ] = chunk_time(prompt_time, 2)

            prompt_gate_attn = None
            prompt_shift_ff = None
            prompt_scale_ff = None
            prompt_gate_ff = None
        else:
            [
                prompt_shift_attn,
                prompt_scale_attn,
                prompt_gate_attn,
                prompt_shift_ff,
                prompt_scale_ff,
                prompt_gate_ff,
            ] = chunk_time(prompt_time, 6)

        spatial_normed = self._spatial_norm_1(spatial)
        prompt_normed = self._prompt_norm_1(prompt)

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
        )
        ttnn.deallocate(prompt_normed)
        ttnn.deallocate(spatial_gate_dual_attn)
        if prompt_gate_attn is not None:
            ttnn.deallocate(prompt_gate_attn)
        ttnn.deallocate(prompt_scale_attn)
        ttnn.deallocate(prompt_shift_attn)
        ttnn.deallocate(spatial_scale_dual_attn)
        ttnn.deallocate(spatial_shift_dual_attn)

        spatial += spatial_attn
        ttnn.deallocate(spatial_attn)

        if self._spatial_attn is not None:
            assert spatial_gate_attn is not None
            assert spatial_scale_attn is not None
            assert spatial_shift_attn is not None

            spatial += self._spatial_attn_block(
                spatial_normed, gate=spatial_gate_attn, scale=spatial_scale_attn, shift=spatial_shift_attn
            )
            ttnn.deallocate(spatial_normed)
            ttnn.deallocate(spatial_gate_attn)
            ttnn.deallocate(spatial_scale_attn)
            ttnn.deallocate(spatial_shift_attn)

        spatial_normed = self._spatial_norm_2(spatial)

        spatial_normed = ttnn.clone(spatial_normed, dtype=ttnn.bfloat8_b)

        spatial += self._spatial_ff_block(
            spatial_normed, gate=spatial_gate_ff, scale=spatial_scale_ff, shift=spatial_shift_ff
        )
        ttnn.deallocate(spatial_normed)
        ttnn.deallocate(spatial_gate_ff)
        ttnn.deallocate(spatial_scale_ff)
        ttnn.deallocate(spatial_shift_ff)

        if self._context_pre_only:
            return spatial, None

        assert prompt_scale_ff is not None
        assert prompt_shift_ff is not None
        assert prompt_gate_ff is not None

        prompt += prompt_attn
        ttnn.deallocate(prompt_attn)

        prompt_normed = self._prompt_norm_2(prompt)
        prompt += self._prompt_ff_block(
            prompt_normed,
            gate=prompt_gate_ff,
            scale=prompt_scale_ff,
            shift=prompt_shift_ff,
        )
        ttnn.deallocate(prompt_normed)
        ttnn.deallocate(prompt_gate_ff)
        ttnn.deallocate(prompt_scale_ff)
        ttnn.deallocate(prompt_shift_ff)

        return spatial, prompt


def chunk_time(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[2] // count
    return [t[:, :, i * size : (i + 1) * size] for i in range(count)]
