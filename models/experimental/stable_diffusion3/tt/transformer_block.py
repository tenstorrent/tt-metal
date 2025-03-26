# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn
import tracy

from .attention import TtAttention, TtAttentionParameters
from .feed_forward import TtFeedForward, TtFeedForwardParameters
from .linear import TtLinear, TtLinearParameters
from .normalization import TtDistributedLayerNorm, TtLayerNorm, TtLayerNormParameters
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
        num_heads: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtTransformerBlockParameters:
        has_spatial_attn = has_substate(state, "attn2")
        has_ff_context = has_substate(state, "ff_context")
        spatial_time_embed_chunks = 9 if has_spatial_attn else 6
        prompt_time_embed_chunks = 2 if not has_ff_context else 6
        return cls(
            dual_attn=TtAttentionParameters.from_torch(
                substate(state, "attn"), num_heads=num_heads, dtype=dtype, device=device
            ),
            spatial_attn=TtAttentionParameters.from_torch(
                substate(state, "attn2"), num_heads=num_heads, dtype=dtype, device=device
            )
            if has_spatial_attn
            else None,
            spatial_norm_1=TtLayerNormParameters.from_torch(
                substate(state, "norm1.norm"), dtype=dtype, device=device, shard_dim=-1
            ),
            spatial_norm_2=TtLayerNormParameters.from_torch(
                substate(state, "norm2"), dtype=dtype, device=device, shard_dim=-1
            ),
            prompt_norm_1=TtLayerNormParameters.from_torch(
                substate(state, "norm1_context.norm"), dtype=dtype, device=device, shard_dim=-1
            ),
            spatial_time_embed=TtLinearParameters.from_torch_time_embed(
                substate(state, "norm1.linear"),
                dtype=dtype,
                device=device,
                num_chunks=spatial_time_embed_chunks,
            ),
            prompt_time_embed=TtLinearParameters.from_torch_time_embed(
                substate(state, "norm1_context.linear"),
                dtype=dtype,
                device=device,
                num_chunks=prompt_time_embed_chunks,
            ),
            # spatial_time_embed=TtLinearParameters.from_torch(
            #     substate(state, "norm1.linear"),
            #     dtype=dtype,
            #     device=device,
            #     shard_dim=-1,
            # ),
            # prompt_time_embed=TtLinearParameters.from_torch(
            #     substate(state, "norm1_context.linear"),
            #     dtype=dtype,
            #     device=device,
            #     shard_dim=-1,
            # ),
            spatial_ff=TtFeedForwardParameters.from_torch(substate(state, "ff"), dtype=dtype, device=device),
            prompt_ff=TtFeedForwardParameters.from_torch(substate(state, "ff_context"), dtype=dtype, device=device)
            if has_ff_context
            else None,
        )


class TtTransformerBlock:
    def __init__(
        self,
        parameters: TtTransformerBlockParameters,
        *,
        num_heads: int,
        device,
    ) -> None:
        eps = 1e-6

        self._dual_attn = TtAttention(parameters.dual_attn, num_heads=num_heads, device=device)
        self._spatial_attn = (
            TtAttention(parameters.spatial_attn, num_heads=num_heads, device=device)
            if parameters.spatial_attn is not None
            else None
        )

        self._spatial_norm_1 = TtDistributedLayerNorm(parameters.spatial_norm_1, eps=eps)
        self._spatial_norm_2 = TtDistributedLayerNorm(parameters.spatial_norm_2, eps=eps)
        self._prompt_norm_1 = TtDistributedLayerNorm(parameters.prompt_norm_1, eps=eps)
        self._prompt_norm_2 = TtDistributedLayerNorm(TtLayerNormParameters(), eps=eps)

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

        # spatial = ttnn.all_gather(spatial, dim=-1)
        # prompt = ttnn.all_gather(prompt, dim=-1)
        spatial_scaled = spatial * (1 + spatial_scale) + spatial_shift
        # ttnn.deallocate(spatial)
        # spatial_scaled = ttnn.to_memory_config(spatial_scaled, ttnn.DRAM_MEMORY_CONFIG)
        # ttnn.deallocate(spatial_scaled)

        prompt_scaled = prompt * (1 + prompt_scale) + prompt_shift

        spatial_scaled = ttnn.all_gather(spatial_scaled, dim=-1)
        prompt_scaled = ttnn.all_gather(prompt_scaled, dim=-1)
        spatial_attn, prompt_attn = self._dual_attn(spatial=spatial_scaled, prompt=prompt_scaled, deallocate=True)
        # spatial_attn = ttnn.all_gather(spatial_attn, dim=-1)
        # prompt_attn = ttnn.all_gather(prompt_attn, dim=-1)

        spatial_attn_scaled = spatial_gate * spatial_attn
        prompt_attn_scaled = prompt_gate * prompt_attn if prompt_gate is not None else None

        # spatial_attn_scaled = ttnn.all_gather(spatial_attn_scaled, dim=-1)
        # prompt_attn_scaled = ttnn.all_gather(prompt_attn_scaled, dim=-1)

        ttnn.deallocate(spatial_attn)
        ttnn.deallocate(prompt_attn)
        return spatial_attn_scaled, prompt_attn_scaled

    def _spatial_ff_block(
        self, inp: ttnn.Tensor, *, gate: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor
    ) -> ttnn.Tensor:
        scaled = inp * (1 + scale) + shift
        scaled = ttnn.all_gather(scaled, dim=-1)
        result = gate * self._spatial_ff(scaled)
        # result = ttnn.all_gather(result, dim=-1)
        ttnn.deallocate(scaled)
        return result

    def _prompt_ff_block(
        self, inp: ttnn.Tensor, *, gate: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor
    ) -> ttnn.Tensor:
        assert self._prompt_ff is not None

        scaled = inp * (1 + scale) + shift
        scaled = ttnn.all_gather(scaled, dim=-1)
        result = gate * self._prompt_ff(scaled)
        # result = ttnn.all_gather(result, dim=-1)
        ttnn.deallocate(scaled)
        return result

    def __call__(  # noqa: PLR0915
        self, *, spatial: ttnn.Tensor, prompt: ttnn.Tensor, time_embed: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        t = ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        spatial_time = self._spatial_time_embed(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        prompt_time = self._prompt_time_embed(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        print(f"spatial_time.shape: {spatial_time.shape}")
        print(f"prompt_time.shape: {prompt_time.shape}")
        # spatial_time = ttnn.reshape(
        #     spatial_time, (1, spatial_time.shape[0], spatial_time.shape[1], spatial_time.shape[2])
        # )
        # prompt_time = ttnn.reshape(prompt_time, (1, prompt_time.shape[0], prompt_time.shape[1], prompt_time.shape[2]))
        # spatial_time = ttnn.all_gather(spatial_time, dim=len(spatial_time.shape) - 1)
        # prompt_time = ttnn.all_gather(prompt_time, dim=len(prompt_time.shape) - 1)
        # spatial_time = ttnn.reshape(spatial_time, (spatial_time.shape[1], spatial_time.shape[2], spatial_time.shape[3]))
        # prompt_time = ttnn.reshape(prompt_time, (prompt_time.shape[1], prompt_time.shape[2], prompt_time.shape[3]))
        # ttnn.deallocate(t)
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
            print(f"chunking prompt_time for context pre only: {prompt_time.shape}")
            [
                prompt_scale_attn,
                prompt_shift_attn,
            ] = chunk_time(prompt_time, 2)

            prompt_gate_attn = None
            prompt_shift_ff = None
            prompt_scale_ff = None
            prompt_gate_ff = None
        else:
            print(f"not context pre only: {prompt_time.shape}")
            [
                prompt_shift_attn,
                prompt_scale_attn,
                prompt_gate_attn,
                prompt_shift_ff,
                prompt_scale_ff,
                prompt_gate_ff,
            ] = chunk_time(prompt_time, 6)

        spatial_normed = self._spatial_norm_1(spatial)
        # spatial_normed = ttnn.all_gather(spatial_normed, dim=-1)
        # spatial = ttnn.all_gather(spatial, dim=-1)
        prompt_normed = self._prompt_norm_1(prompt)
        # prompt_normed = ttnn.all_gather(prompt_normed, dim=-1)
        # prompt = ttnn.all_gather(prompt, dim=-1)
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
            assert False, "Not supporting right now on Colman's branch"
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

        #        spatial_normed = ttnn.clone(spatial_normed, dtype=ttnn.bfloat8_b)

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
