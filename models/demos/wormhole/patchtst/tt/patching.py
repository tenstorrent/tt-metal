# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Source lineage: HuggingFace PatchTST and PatchTST paper implementation details
# - https://huggingface.co/docs/transformers/en/model_doc/patchtst
# - https://github.com/huggingface/transformers/tree/main/src/transformers/models/patchtst
# - https://arxiv.org/abs/2211.14730

from __future__ import annotations

import torch

import ttnn


def _resolve_patch_geometry(
    runtime_context_length: int,
    context_length: int,
    patch_length: int,
    patch_stride: int,
) -> tuple[int, int, int]:
    runtime_context_length = int(runtime_context_length)
    if runtime_context_length != context_length:
        # Bring-up supports runtime context override (e.g. long-context stretch path).
        # We keep patch_length/stride fixed and derive patch windows from runtime sequence length.
        context_length = runtime_context_length
    if context_length <= patch_length:
        raise ValueError("context_length must be greater than patch_length")

    num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1
    new_sequence_length = patch_length + patch_stride * (num_patches - 1)
    sequence_start = context_length - new_sequence_length
    return context_length, num_patches, sequence_start


def patchify(past_values: torch.Tensor, context_length: int, patch_length: int, patch_stride: int) -> torch.Tensor:
    context_length, _, sequence_start = _resolve_patch_geometry(
        runtime_context_length=int(past_values.shape[-2]),
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
    )

    output = past_values[:, sequence_start:, :]
    output = output.unfold(dimension=-2, size=patch_length, step=patch_stride)
    output = output.transpose(-2, -3).contiguous()
    return output


def patchify_tt(past_values: ttnn.Tensor, context_length: int, patch_length: int, patch_stride: int) -> ttnn.Tensor:
    batch_size = int(past_values.shape[0])
    runtime_context_length = int(past_values.shape[1])
    num_channels = int(past_values.shape[2])
    _, num_patches, sequence_start = _resolve_patch_geometry(
        runtime_context_length=runtime_context_length,
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
    )

    pieces: list[ttnn.Tensor] = []
    for patch_idx in range(num_patches):
        start = sequence_start + patch_idx * patch_stride
        end = start + patch_length
        patch = ttnn.slice(
            past_values,
            (0, start, 0),
            (batch_size, end, num_channels),
        )
        patch = ttnn.permute(patch, (0, 2, 1))
        patch = ttnn.reshape(patch, (batch_size, num_channels, 1, patch_length))
        pieces.append(patch)

    if not pieces:
        raise ValueError("patchify_tt produced zero patches.")

    merged = pieces[0]
    for part in pieces[1:]:
        combined = ttnn.concat([merged, part], dim=2)
        ttnn.deallocate(merged)
        ttnn.deallocate(part)
        merged = combined
    return merged


def random_masking(
    patch_input: torch.Tensor,
    random_mask_ratio: float,
    mask_value: float = 0.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if random_mask_ratio < 0.0 or random_mask_ratio >= 1.0:
        raise ValueError(f"random_mask_ratio must be in [0, 1), got {random_mask_ratio}")

    bsz, channels, num_patches, patch_length = patch_input.shape
    len_keep = int(num_patches * (1.0 - random_mask_ratio))

    generator = None
    if seed is not None:
        generator = torch.Generator(device=patch_input.device.type)
        generator.manual_seed(int(seed))
    noise = torch.rand(bsz, channels, num_patches, device=patch_input.device, generator=generator)
    base_mask = torch.ones((bsz, channels, num_patches), device=patch_input.device)
    base_mask[:, :, :len_keep] = 0

    ids_shuffle = torch.argsort(noise, dim=-1)
    ids_restore = torch.argsort(ids_shuffle, dim=-1)
    mask = torch.gather(base_mask, dim=-1, index=ids_restore)
    expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, patch_length).to(dtype=torch.bool)

    masked = patch_input.masked_fill(expanded_mask, mask_value)
    return masked, mask.to(dtype=torch.bool)


def forecast_masking(
    patch_input: torch.Tensor,
    num_forecast_mask_patches: int | list[int] | tuple[int, ...],
    mask_value: float = 0.0,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(num_forecast_mask_patches, int):
        forecast_choices = [num_forecast_mask_patches]
    else:
        forecast_choices = [int(value) for value in num_forecast_mask_patches]

    if len(forecast_choices) == 0:
        raise ValueError("num_forecast_mask_patches cannot be empty for forecast masking")

    bsz, channels, num_patches, patch_length = patch_input.shape
    mask = torch.zeros((bsz, channels, num_patches), device=patch_input.device)

    groups: list[tuple[int, int]] = []
    total_assigned = 0
    for value in forecast_choices:
        if value <= 0 or value >= num_patches:
            raise ValueError(f"num_forecast_mask_patches value {value} must be > 0 and < num_patches ({num_patches})")
        count = int(bsz / len(forecast_choices))
        groups.append((value, count))
        total_assigned += count

    if total_assigned < bsz:
        value, count = groups[0]
        groups[0] = (value, count + (bsz - total_assigned))
    elif total_assigned > bsz:
        value, count = groups[-1]
        groups[-1] = (value, count - (total_assigned - bsz))

    start = 0
    for patch_count, count in groups:
        end = start + count
        mask[start:end, :, -patch_count:] = 1
        start = end

    generator = None
    if seed is not None:
        generator = torch.Generator(device=patch_input.device.type)
        generator.manual_seed(int(seed))
    perm = torch.randperm(mask.shape[0], device=patch_input.device, generator=generator)
    mask = mask[perm]
    expanded_mask = mask.unsqueeze(-1).expand(-1, -1, -1, patch_length).to(dtype=torch.bool)
    masked = patch_input.masked_fill(expanded_mask, mask_value)
    return masked, mask.to(dtype=torch.bool)


def apply_mask(
    patch_input: torch.Tensor,
    mask_type: str,
    random_mask_ratio: float,
    num_forecast_mask_patches: int | list[int] | tuple[int, ...],
    mask_value: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask_type == "random":
        return random_masking(
            patch_input,
            random_mask_ratio=random_mask_ratio,
            mask_value=mask_value,
            seed=seed,
        )
    if mask_type == "forecast":
        return forecast_masking(
            patch_input,
            num_forecast_mask_patches=num_forecast_mask_patches,
            mask_value=mask_value,
            seed=seed,
        )
    raise ValueError(f"Unsupported mask_type: {mask_type}")
