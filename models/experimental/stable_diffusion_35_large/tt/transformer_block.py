# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from . import utils
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
    prompt_norm_2: TtLayerNormParameters
    spatial_norm_1: TtLayerNormParameters
    spatial_norm_2: TtLayerNormParameters
    prompt_ff: TtFeedForwardParameters | None
    spatial_ff: TtFeedForwardParameters
    distributed: bool

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        num_heads: int,
        unpadded_num_heads: int,
        hidden_dim_padding: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtTransformerBlockParameters:
        embedding_dim = state["norm1.linear.weight"].shape[1]

        def norm(state: dict[str, torch.Tensor]) -> TtLayerNormParameters:
            return TtLayerNormParameters.from_torch(
                state,
                dtype=dtype,
                device=device,
                weight_shape=[embedding_dim + hidden_dim_padding],
            )

        has_spatial_attn = has_substate(state, "attn2")
        has_ff_context = has_substate(state, "ff_context")
        spatial_time_embed_chunks = 9 if has_spatial_attn else 6
        prompt_time_embed_chunks = 2 if not has_ff_context else 6
        return cls(
            dual_attn=TtAttentionParameters.from_torch(
                substate(state, "attn"),
                num_heads=num_heads,
                unpadded_num_heads=unpadded_num_heads,
                hidden_dim_padding=hidden_dim_padding,
                dtype=dtype,
                device=device,
            ),
            spatial_attn=TtAttentionParameters.from_torch(
                substate(state, "attn2"),
                num_heads=num_heads,
                unpadded_num_heads=unpadded_num_heads,
                hidden_dim_padding=hidden_dim_padding,
                dtype=dtype,
                device=device,
            )
            if has_spatial_attn
            else None,
            spatial_norm_1=norm(substate(state, "norm1.norm")),
            spatial_norm_2=norm(substate(state, "norm2")),
            prompt_norm_1=norm(substate(state, "norm1_context.norm")),
            prompt_norm_2=norm({}),
            spatial_time_embed=TtLinearParameters.from_torch_time_embed(
                substate(state, "norm1.linear"),
                dtype=dtype,
                device=device,
                num_chunks=spatial_time_embed_chunks,
                hidden_dim_padding=hidden_dim_padding,
                unsqueeze_bias=True,
            ),
            prompt_time_embed=TtLinearParameters.from_torch_time_embed(
                substate(state, "norm1_context.linear"),
                dtype=dtype,
                device=device,
                num_chunks=prompt_time_embed_chunks,
                hidden_dim_padding=hidden_dim_padding,
                unsqueeze_bias=True,
            ),
            spatial_ff=TtFeedForwardParameters.from_torch(substate(state, "ff"), dtype=dtype, device=device),
            prompt_ff=TtFeedForwardParameters.from_torch(substate(state, "ff_context"), dtype=dtype, device=device)
            if has_ff_context
            else None,
            distributed=device.get_num_devices() > 1,
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

        self._spatial_norm_1 = TtLayerNorm(parameters.spatial_norm_1, eps=eps)
        self._spatial_norm_2 = TtLayerNorm(parameters.spatial_norm_2, eps=eps)
        self._prompt_norm_1 = TtLayerNorm(parameters.prompt_norm_1, eps=eps)
        self._prompt_norm_2 = TtLayerNorm(parameters.prompt_norm_2, eps=eps)

        self._spatial_ff = TtFeedForward(parameters.spatial_ff)
        self._prompt_ff = TtFeedForward(parameters.prompt_ff) if parameters.prompt_ff is not None else None

        self._spatial_time_embed = TtLinear(parameters.spatial_time_embed)
        self._prompt_time_embed = TtLinear(parameters.prompt_time_embed)

        self._context_pre_only = self._prompt_ff is None
        self._distributed = parameters.distributed

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
        N: int,
        L: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        spatial_scaled = spatial * (1 + spatial_scale) + spatial_shift
        prompt_scaled = prompt * (1 + prompt_scale) + prompt_shift
        if self._distributed:
            spatial_scaled = utils.all_gather(spatial_scaled, dim=-1)
            prompt_scaled = utils.all_gather(prompt_scaled, dim=-1)
        spatial_attn, prompt_attn = self._dual_attn(
            spatial=spatial_scaled, prompt=prompt_scaled, deallocate=True, N=N, L=L
        )
        spatial_attn_scaled = spatial_gate * spatial_attn
        prompt_attn_scaled = prompt_gate * prompt_attn if prompt_gate is not None else None
        ttnn.deallocate(spatial_attn)
        ttnn.deallocate(prompt_attn)
        return spatial_attn_scaled, prompt_attn_scaled

    def _spatial_ff_block(
        self, inp: ttnn.Tensor, *, gate: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor
    ) -> ttnn.Tensor:
        scaled = inp * (1 + scale) + shift
        if self._distributed:
            scaled = utils.all_gather(scaled, dim=-1)
        result = gate * self._spatial_ff(scaled)
        ttnn.deallocate(scaled)
        return result

    def _prompt_ff_block(
        self, inp: ttnn.Tensor, *, gate: ttnn.Tensor, scale: ttnn.Tensor, shift: ttnn.Tensor
    ) -> ttnn.Tensor:
        assert self._prompt_ff is not None

        scaled = inp * (1 + scale) + shift
        if self._distributed:
            scaled = utils.all_gather(scaled, dim=-1)
        result = gate * self._prompt_ff(scaled)
        ttnn.deallocate(scaled)
        return result

    def __call__(  # noqa: PLR0915
        self, *, spatial: ttnn.Tensor, prompt: ttnn.Tensor, time_embed: ttnn.Tensor, N: int, L: int
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        t = utils.silu(time_embed)
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
            # print(f"chunking prompt_time for context pre only: {prompt_time.shape}")
            [
                prompt_scale_attn,
                prompt_shift_attn,
            ] = chunk_time(prompt_time, 2)

            prompt_gate_attn = None
            prompt_shift_ff = None
            prompt_scale_ff = None
            prompt_gate_ff = None
        else:
            # print(f"not context pre only: {prompt_time.shape}")
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
        spatial_attn, prompt_attn = self._dual_attn_block(
            spatial=spatial_normed,
            prompt=prompt_normed,
            spatial_gate=spatial_gate_dual_attn,
            prompt_gate=prompt_gate_attn,
            prompt_scale=prompt_scale_attn,
            prompt_shift=prompt_shift_attn,
            spatial_scale=spatial_scale_dual_attn,
            spatial_shift=spatial_shift_dual_attn,
            N=N,
            L=L,
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
    size = t.shape[-1] // count
    return [t[:, :, :, i * size : (i + 1) * size] for i in range(count)]


def chunk_device_tensors(ttnn_tensor: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = ttnn_tensor.shape[-1] // count
    device_slices = []
    for i, device_tensor in enumerate(ttnn.get_device_tensors(ttnn_tensor)):
        device_slice = device_tensor[:, :, :, i * size : (i + 1) * size]
        device_slices.append(device_slice)
    return device_slices
