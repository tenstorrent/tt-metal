# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Source lineage: HuggingFace PatchTST and PatchTST paper implementation details
# - https://huggingface.co/docs/transformers/en/model_doc/patchtst
# - https://github.com/huggingface/transformers/tree/main/src/transformers/models/patchtst
# - https://arxiv.org/abs/2211.14730

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.demos.wormhole.patchtst.reference.hf_reference import ReferenceArtifacts
from models.demos.wormhole.patchtst.tt.common import (
    MEMORY_CONFIG_BY_TIER,
    PatchTSTRuntimePolicy,
    TTLinear,
    build_linear,
    build_linear_from_state,
)


def _pad_last_dim_torch(tensor: torch.Tensor, padded_dim: int) -> torch.Tensor:
    pad = padded_dim - int(tensor.shape[-1])
    if pad <= 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad))


def _pad_last_dim_tt(tensor: ttnn.Tensor, padded_dim: int) -> ttnn.Tensor:
    pad = padded_dim - int(tensor.shape[-1])
    if pad <= 0:
        return tensor
    return ttnn.pad(tensor, padding=((0, 0), (0, 0), (0, pad)), value=0.0)


def _build_linear_padded_input(
    state: dict[str, torch.Tensor],
    prefix: str,
    padded_input_dim: int,
    *,
    device,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
) -> TTLinear:
    weight = state[f"{prefix}.weight"]
    padded_weight = torch.zeros((weight.shape[0], padded_input_dim), dtype=weight.dtype)
    padded_weight[:, : weight.shape[1]] = weight
    bias_key = f"{prefix}.bias"
    return build_linear(
        padded_weight,
        state[bias_key] if bias_key in state else None,
        device=device,
        dtype=dtype,
        memory_config=memory_config,
    )


@dataclass
class PatchTSTEmbedder:
    shared_projection: TTLinear | None
    channel_projections: list[TTLinear]

    def __call__(
        self,
        patch_input: torch.Tensor | ttnn.Tensor,
        share_embedding: bool,
        device,
        runtime: PatchTSTRuntimePolicy,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> ttnn.Tensor:
        mem_cfg = MEMORY_CONFIG_BY_TIER[runtime.activation_memory_tier]
        bsz, num_channels, num_patches, patch_length = [int(x) for x in patch_input.shape]
        padded_input_dim = int(math.ceil(patch_length / ttnn.TILE_SIZE) * ttnn.TILE_SIZE)

        if share_embedding:
            if self.shared_projection is None:
                raise ValueError("share_embedding=True requires a shared embedder projection.")
            if isinstance(patch_input, ttnn.Tensor):
                flat_tt = ttnn.reshape(patch_input, (bsz * num_channels, 1, num_patches, patch_length))
                padded_flat_tt = _pad_last_dim_tt(flat_tt, padded_input_dim)
            else:
                flat_tt = ttnn.from_torch(
                    _pad_last_dim_torch(
                        patch_input.reshape(bsz * num_channels, 1, num_patches, patch_length), padded_input_dim
                    ),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=mem_cfg,
                )
                padded_flat_tt = flat_tt
            out_tt = ttnn.linear(
                padded_flat_tt,
                self.shared_projection.weight,
                bias=self.shared_projection.bias,
                memory_config=mem_cfg,
                dtype=dtype,
            )
            if padded_flat_tt is not flat_tt:
                ttnn.deallocate(padded_flat_tt)
            if flat_tt is not patch_input:
                ttnn.deallocate(flat_tt)
            return ttnn.reshape(out_tt, (bsz, num_channels, num_patches, out_tt.shape[-1]))

        outputs = []
        for channel_idx in range(num_channels):
            projection = self.channel_projections[channel_idx] if self.channel_projections else self.shared_projection
            if projection is None:
                raise ValueError(
                    "Non-shared embedding requires per-channel projections or a shared projection fallback."
                )
            if isinstance(patch_input, ttnn.Tensor):
                channel_input = ttnn.slice(
                    patch_input, (0, channel_idx, 0, 0), (bsz, channel_idx + 1, num_patches, patch_length)
                )
                channel_tt = ttnn.reshape(channel_input, (bsz, 1, num_patches, patch_length))
                ttnn.deallocate(channel_input)
                padded_channel_tt = _pad_last_dim_tt(channel_tt, padded_input_dim)
            else:
                channel_tt = ttnn.from_torch(
                    _pad_last_dim_torch(patch_input[:, channel_idx : channel_idx + 1], padded_input_dim),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=mem_cfg,
                )
                padded_channel_tt = channel_tt
            outputs.append(
                ttnn.linear(
                    padded_channel_tt,
                    projection.weight,
                    bias=projection.bias,
                    memory_config=mem_cfg,
                    dtype=dtype,
                )
            )
            if padded_channel_tt is not channel_tt:
                ttnn.deallocate(padded_channel_tt)
            ttnn.deallocate(channel_tt)

        stacked = outputs[0]
        for tensor in outputs[1:]:
            combined = ttnn.concat([stacked, tensor], dim=1)
            ttnn.deallocate(stacked)
            stacked = combined
        return stacked

    def release(self) -> None:
        if self.shared_projection is not None:
            self.shared_projection.release()
        for projection in self.channel_projections:
            projection.release()


def build_embedder(
    reference: ReferenceArtifacts,
    *,
    device,
    dtype: ttnn.DataType = ttnn.bfloat16,
    memory_tier: str = "dram",
) -> PatchTSTEmbedder:
    state = reference.model.state_dict()
    memory_config = MEMORY_CONFIG_BY_TIER[memory_tier]
    patch_length = int(reference.config.patch_length)
    padded_patch_length = ((patch_length + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    embedder_prefix = "model.encoder.embedder.input_embedding"
    linear_builder = build_linear_from_state if padded_patch_length == patch_length else _build_linear_padded_input

    if bool(reference.config.share_embedding):
        return PatchTSTEmbedder(
            shared_projection=(
                linear_builder(state, embedder_prefix, device=device, dtype=dtype, memory_config=memory_config)
                if padded_patch_length == patch_length
                else linear_builder(
                    state,
                    embedder_prefix,
                    padded_patch_length,
                    device=device,
                    dtype=dtype,
                    memory_config=memory_config,
                )
            ),
            channel_projections=[],
        )

    projections = []
    channel_index = 0
    while f"{embedder_prefix}.{channel_index}.weight" in state:
        projections.append(
            linear_builder(
                state, f"{embedder_prefix}.{channel_index}", device=device, dtype=dtype, memory_config=memory_config
            )
            if padded_patch_length == patch_length
            else linear_builder(
                state,
                f"{embedder_prefix}.{channel_index}",
                padded_patch_length,
                device=device,
                dtype=dtype,
                memory_config=memory_config,
            )
        )
        channel_index += 1
    return PatchTSTEmbedder(shared_projection=None, channel_projections=projections)
