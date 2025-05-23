# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from . import utils
from .fun_attention import sd_joint_attention, TtAttentionParameters
from .fun_feed_forward import sd_feed_forward, TtFeedForwardParameters
from .fun_linear import sd_linear, TtLinearParameters
from .fun_normalization import sd_layer_norm, TtLayerNormParameters
from .parallel_config import DiTParallelConfig
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
    # TODO: Check that norm eps is same as reference for all norms
    eps: float = 1e-6

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
        parallel_config: DiTParallelConfig,
    ) -> TtTransformerBlockParameters:
        embedding_dim = state["norm1.linear.weight"].shape[1]

        def norm(state: dict[str, torch.Tensor]) -> TtLayerNormParameters:
            return TtLayerNormParameters.from_torch(
                state,
                dtype=dtype,
                device=device,
                weight_shape=[embedding_dim + hidden_dim_padding],
                eps=cls.eps,
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
                parallel_config=parallel_config,
            ),
            spatial_attn=TtAttentionParameters.from_torch(
                substate(state, "attn2"),
                num_heads=num_heads,
                unpadded_num_heads=unpadded_num_heads,
                hidden_dim_padding=hidden_dim_padding,
                dtype=dtype,
                device=device,
                parallel_config=parallel_config,
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
            spatial_ff=TtFeedForwardParameters.from_torch(
                substate(state, "ff"), dtype=dtype, device=device, parallel_config=parallel_config
            ),
            prompt_ff=TtFeedForwardParameters.from_torch(
                substate(state, "ff_context"), dtype=dtype, device=device, parallel_config=parallel_config
            )
            if has_ff_context
            else None,
            distributed=device.get_num_devices() > 1,
        )


def sd_spatial_attn_block(
    inp: ttnn.Tensor,
    parameters: TtAttentionParameters,
    parallel_config: DiTParallelConfig,
    *,
    num_heads: int,
    gate: ttnn.Tensor,
    scale: ttnn.Tensor,
    shift: ttnn.Tensor,
) -> ttnn.Tensor:
    # assert self._spatial_attn is not None
    assert parameters.spatial_attn is not None
    scaled = inp * (1 + scale) + shift
    attn, _ = sd_joint_attention(
        spatial=scaled, parameters=parameters, num_heads=num_heads, deallocate=True, parallel_config=parallel_config
    )

    result = gate * attn
    ttnn.deallocate(scaled)
    ttnn.deallocate(attn)
    return result


def sd_dual_attn_block(
    spatial: ttnn.Tensor,
    prompt: ttnn.Tensor,
    spatial_gate: ttnn.Tensor,
    prompt_gate: ttnn.Tensor | None,
    prompt_scale: ttnn.Tensor,
    prompt_shift: ttnn.Tensor,
    spatial_scale: ttnn.Tensor,
    spatial_shift: ttnn.Tensor,
    parameters: TtAttentionParameters,
    parallel_config: DiTParallelConfig,
    num_heads: int,
    N: int,
    L: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
    spatial_scaled = spatial * (1 + spatial_scale) + spatial_shift
    prompt_scaled = prompt * (1 + prompt_scale) + prompt_shift
    if parallel_config.tensor_parallel.factor > 1:
        spatial_scaled = utils.all_gather(spatial_scaled, dim=-1)
        prompt_scaled = utils.all_gather(prompt_scaled, dim=-1)
    spatial_attn, prompt_attn = sd_joint_attention(
        spatial=spatial_scaled,
        prompt=prompt_scaled,
        parameters=parameters,
        num_heads=num_heads,
        deallocate=True,
        N=N,
        L=L,
        parallel_config=parallel_config,
    )
    spatial_attn_scaled = spatial_gate * spatial_attn
    prompt_attn_scaled = prompt_gate * prompt_attn if prompt_gate is not None else None
    ttnn.deallocate(spatial_attn)
    ttnn.deallocate(prompt_attn)
    return spatial_attn_scaled, prompt_attn_scaled


def sd_gated_ff_block(
    inp: ttnn.Tensor,
    parameters: TtFeedForwardParameters,
    parallel_config: DiTParallelConfig,
    *,
    gate: ttnn.Tensor,
    scale: ttnn.Tensor,
    shift: ttnn.Tensor,
) -> ttnn.Tensor:
    scaled = inp * (1 + scale) + shift
    if parallel_config.tensor_parallel.factor > 1:
        scaled = utils.all_gather(scaled, dim=-1)
    result = gate * sd_feed_forward(scaled, parameters, parallel_config=parallel_config)
    ttnn.deallocate(scaled)
    return result


def sd_transformer_block(  # noqa: PLR0915
    *,
    spatial: ttnn.Tensor,
    prompt: ttnn.Tensor,
    time_embed: ttnn.Tensor,
    parameters: TtTransformerBlockParameters,
    parallel_config: DiTParallelConfig,
    num_heads: int,
    N: int,
    L: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
    t = ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    spatial_time = sd_linear(t, parameters.spatial_time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    prompt_time = sd_linear(t, parameters.prompt_time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t)
    if parameters.spatial_attn is not None:
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

    context_pre_only = parameters.prompt_ff is None
    if context_pre_only:
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

    spatial_normed = sd_layer_norm(spatial, parameters.spatial_norm_1)
    prompt_normed = sd_layer_norm(prompt, parameters.prompt_norm_1)
    spatial_attn, prompt_attn = sd_dual_attn_block(
        spatial=spatial_normed,
        prompt=prompt_normed,
        spatial_gate=spatial_gate_dual_attn,
        prompt_gate=prompt_gate_attn,
        prompt_scale=prompt_scale_attn,
        prompt_shift=prompt_shift_attn,
        spatial_scale=spatial_scale_dual_attn,
        spatial_shift=spatial_shift_dual_attn,
        parameters=parameters.dual_attn,
        parallel_config=parallel_config,
        num_heads=num_heads,
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

    if parameters.spatial_attn is not None:
        assert False, "Not supporting right now on Colman's branch"
        assert spatial_gate_attn is not None
        assert spatial_scale_attn is not None
        assert spatial_shift_attn is not None

        spatial += sd_spatial_attn_block(
            spatial_normed,
            parameters=parameters.spatial_attn,
            num_heads=num_heads,
            gate=spatial_gate_attn,
            scale=spatial_scale_attn,
            shift=spatial_shift_attn,
        )
        ttnn.deallocate(spatial_normed)
        ttnn.deallocate(spatial_gate_attn)
        ttnn.deallocate(spatial_scale_attn)
        ttnn.deallocate(spatial_shift_attn)

    spatial_normed = sd_layer_norm(spatial, parameters.spatial_norm_2)
    spatial += sd_gated_ff_block(
        spatial_normed,
        parameters=parameters.spatial_ff,
        parallel_config=parallel_config,
        gate=spatial_gate_ff,
        scale=spatial_scale_ff,
        shift=spatial_shift_ff,
    )
    ttnn.deallocate(spatial_normed)
    ttnn.deallocate(spatial_gate_ff)
    ttnn.deallocate(spatial_scale_ff)
    ttnn.deallocate(spatial_shift_ff)

    if context_pre_only:
        return spatial, None

    assert prompt_scale_ff is not None
    assert prompt_shift_ff is not None
    assert prompt_gate_ff is not None

    prompt += prompt_attn
    ttnn.deallocate(prompt_attn)
    prompt_normed = sd_layer_norm(prompt, parameters.prompt_norm_2)
    prompt += sd_gated_ff_block(
        prompt_normed,
        parameters=parameters.prompt_ff,
        parallel_config=parallel_config,
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
