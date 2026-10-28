# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host front-end of the Kairos forward (torch fp32, python_env): KairosModel.encode up to the embedding input.

Mirrors the reference op for op (mask, truncate, instance norm, rfft features, router, dynamic patch division,
x_final), so the routing matches it exactly. `divide_patches_by_moe` is DynamicPatch._divide_patches_by_moe from
foundation-model-research/Kairos @ 16eb4b98 (Apache-2.0), copied line for line, with the gate call taken out.
"""
import torch
import torch.nn.functional as F

from models.experimental.kairos_forecast.tt.kairos_patch_utils import (
    _create_granularity_mask,
    _create_initial_position_mapping,
    _create_initial_setup,
    _divide_patches,
    _generate_x_final,
    _map_to_parent_blocks,
    _update_parent_mapping,
    _update_position_mapping,
)

CONTEXT, PATCH, N_REAL, LEVELS, S_FEAT = 2048, 128, 3, 3, 128


def instance_norm(x, eps=1e-5):
    loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
    scale = torch.nan_to_num(torch.nanmean((x - loc).square(), dim=-1, keepdim=True).sqrt(), nan=1.0)
    scale = torch.where(scale == 0, torch.abs(loc) + eps, scale)
    return (x - loc) / scale, (loc, scale)


def fft_features(context, mask, w, b):
    amp = torch.abs(torch.fft.rfft(torch.where(mask.bool(), context, torch.zeros_like(context)), dim=-1))
    return F.layer_norm(amp[..., :S_FEAT], (S_FEAT,), w, b)


def divide_patches_by_moe(x, mask, size, weights, indices):
    batch, patch_num, patch_len = x.shape
    expert_indices = indices.view(batch, patch_num, -1)  # [batch, patch_num, n_experts]
    weights = weights.view(batch, patch_num, -1)  # [batch, patch_num, n_experts]
    n_real_experts = N_REAL
    is_real_expert_mask = (expert_indices < n_real_experts)
    masked_indices = torch.where(is_real_expert_mask, expert_indices, -1)
    indices, _ = torch.max(masked_indices, dim=-1)  # [batch, patch_num]
    indices[indices == -1] = 0

    # Create initial parent mapping and blocks
    original_patches, original_mask, parent_mapping = _create_initial_setup(x, mask, size)

    # Save original expert indices for granularity mask generation
    original_expert_indices = expert_indices

    # Initialize position mapping for tracking patch positions within original patches
    position_mapping = _create_initial_position_mapping(x, size)

    # Apply division to both x and mask
    for i in range(LEVELS - 1):
        current_patch_num = x.size(1)
        to_divide = (indices > i).to(x.device)
        div_counts = to_divide.sum(dim=1)  # [B]

        new_x, new_size, weights, expert_indices = _divide_patches(x, size, to_divide, weights, expert_indices)
        new_mask, _ = _divide_patches(mask, size, to_divide)

        # Update parent mapping and position mapping
        new_parent_mapping = _update_parent_mapping(parent_mapping, to_divide, div_counts, x.device)
        new_position_mapping = _update_position_mapping(position_mapping, to_divide, div_counts, x.device)

        new_patch_nums = current_patch_num + div_counts
        total_elements = new_patch_nums.sum().item()

        expand_mask = torch.stack([
            torch.ones_like(to_divide),
            to_divide
        ], dim=-1).view(batch, -1)  # [B, 2L]

        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, 2).reshape(batch, -1)
        valid_indices = expanded_indices[expand_mask]

        assert expand_mask.sum().item() == total_elements, \
            f"Mask sum {expand_mask.sum().item()} vs total {total_elements}"
        assert valid_indices.numel() == total_elements, \
            f"Indices {valid_indices.numel()} vs total {total_elements}"

        max_len = new_patch_nums.max()
        indices = torch.full((batch, max_len), -1, dtype=torch.long, device=x.device)

        valid_pos_mask = (torch.arange(max_len, device=x.device)[None, :] < new_patch_nums[:, None])

        indices[valid_pos_mask] = valid_indices

        x, mask, size = new_x, new_mask, new_size
        parent_mapping = new_parent_mapping
        position_mapping = new_position_mapping

    parent_blocks = _map_to_parent_blocks(original_patches, parent_mapping, x.shape)
    parent_blocks_mask = _map_to_parent_blocks(original_mask, parent_mapping, mask.shape)
    granularity_mask = _create_granularity_mask(original_expert_indices, parent_mapping, position_mapping,
                                                x.shape, patch_len)
    # Generate x_final with rearranged features
    x_final = _generate_x_final(parent_blocks, parent_blocks_mask, granularity_mask)

    return new_x, new_mask, new_size, weights, expert_indices, x_final


def front_end(past_target, router, W):
    """past_target [B, T] (NaN = missing). Returns (front, routing): the dict the device forward takes, and the
    gate's own outputs. router is a RouterState; its bias is used, then updated, as the reference's gate does."""
    mask = (~torch.isnan(past_target)).to(past_target.dtype)
    if past_target.shape[-1] > CONTEXT:
        past_target, mask = past_target[..., -CONTEXT:], mask[..., -CONTEXT:]
    context = torch.where(mask > 0.0, past_target, torch.nan)
    context, loc_scale = instance_norm(context)
    s_features = fft_features(context, mask, W["fft_norm.weight"], W["fft_norm.bias"])
    px, pm = context.unfold(-1, PATCH, PATCH), mask.unfold(-1, PATCH, PATCH)
    size = torch.full(px.shape[:-1], PATCH, dtype=torch.int64)
    B = px.shape[0]
    w, idx, scores = router.route(px.reshape(-1, PATCH), W["patch.moe.gate.weight"])
    router.update(w, idx)
    new_x, new_mask, new_size, ew, ei, x_final = divide_patches_by_moe(px, pm, size, w, idx)
    N = new_size.shape[1]
    front = {
        "s_features": s_features,
        "x_final": x_final.view(N_REAL, B, N, -1),
        "weights": ew,
        "indices": ei,
        "attn_mask": torch.nan_to_num(new_mask, nan=0.0).sum(dim=-1) > 0,
        "loc_scale": loc_scale,
    }
    return front, {"weights": w, "indices": idx, "scores": scores, "bias_after": router.bias.clone()}
