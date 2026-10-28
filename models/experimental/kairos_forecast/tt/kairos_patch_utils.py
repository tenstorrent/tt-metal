# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Vendored unchanged from foundation-model-research/Kairos @ 16eb4b98, tsfm/model/kairos/patch_utils.py
# (Apache-2.0). Host-side index logic of the dynamic patching; the port must match it exactly.
from typing import Optional, Tuple
import torch


def _divide_patches(
        x: torch.Tensor, size: torch.Tensor, to_divide: torch.Tensor, weights: Optional[torch.Tensor] = None,
        expert_indices: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    batch, patch_len, patch_size = x.shape

    # Calculate indices for the new positions
    div_counts = to_divide.sum(dim=1)  # [batch]
    if div_counts.max().item() == 0:
        if weights is not None:
            return x, size, weights, expert_indices
        else:
            return x, size
    new_patch_len: int = patch_len + div_counts.max().item()  # type: ignore

    # Create position mapping for each batch
    batch_idx = torch.arange(batch, device=x.device)[:, None].expand(-1, patch_len)  # [batch, patch_len]

    # Create indices for scattered elements
    base_idx = torch.arange(patch_len, device=x.device)[None, :].expand(batch, -1)  # [batch, patch_len]
    offset = torch.cumsum(to_divide.float(), dim=1).long()  # [batch, patch_len]
    new_positions = base_idx + offset  # [batch, patch_len]

    # Initialize output tensors
    new_x = torch.zeros(batch, new_patch_len, patch_size, device=x.device, dtype=x.dtype)
    new_size = torch.zeros(batch, new_patch_len, device=size.device, dtype=size.dtype)
    if weights is not None:
        new_weights_shape = (batch, new_patch_len) + weights.shape[2:]
        new_weights = torch.zeros(new_weights_shape, dtype=weights.dtype, device=weights.device)
    if expert_indices is not None:
        new_expert_indices_shape = (batch, new_patch_len) + expert_indices.shape[2:]
        new_expert_indices = torch.zeros(new_expert_indices_shape, dtype=expert_indices.dtype,
                                            device=expert_indices.device)

    # Scatter undivided patches
    undivided = ~to_divide
    new_x[batch_idx[undivided], new_positions[undivided]] = x[undivided]
    new_size[batch_idx[undivided], new_positions[undivided]] = size[undivided]
    if weights is not None:
        new_weights[batch_idx[undivided], new_positions[undivided]] = weights[undivided]
    if expert_indices is not None:
        new_expert_indices[batch_idx[undivided], new_positions[undivided]] = expert_indices[undivided]

    # Scatter divided patches
    divided = to_divide
    # Get the sizes for divided patches
    div_sizes = size[divided].div(2, rounding_mode="floor")

    # First half of divided patches
    first_half_idx = torch.arange(patch_size, device=x.device)[None, :] < div_sizes[:, None]
    new_x[batch_idx[divided], new_positions[divided] - 1] = torch.where(
        first_half_idx, x[divided], torch.zeros_like(x[divided])
    )
    new_size[batch_idx[divided], new_positions[divided] - 1] = div_sizes
    if weights is not None:
        new_weights[batch_idx[divided], new_positions[divided] - 1] = weights[divided]
    if expert_indices is not None:
        new_expert_indices[batch_idx[divided], new_positions[divided] - 1] = expert_indices[divided]

    # Second half of divided patches
    second_half_idx = (torch.arange(patch_size, device=x.device)[None, :] >= div_sizes[:, None]) & (
        torch.arange(patch_size, device=x.device)[None, :] < size[divided][:, None]
    )
    second_half_values = torch.where(second_half_idx, x[divided], torch.zeros_like(x[divided]))
    new_x[batch_idx[divided], new_positions[divided]] = torch.roll(
        second_half_values, -div_sizes.max().item(), dims=1
    )
    new_size[batch_idx[divided], new_positions[divided]] = size[divided] - div_sizes
    if weights is not None:
        new_weights[batch_idx[divided], new_positions[divided]] = weights[divided]
        if expert_indices is not None:
            new_expert_indices[batch_idx[divided], new_positions[divided]] = expert_indices[divided]
            return new_x, new_size, new_weights, new_expert_indices
        return new_x, new_size, new_weights, None
    else:
        return new_x, new_size

def _create_initial_setup(x: torch.Tensor, mask: torch.Tensor, size: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create initial original patches, masks and parent mapping.
    Assumes patch_num == num_original_patches (one patch per original patch).

    Args:
        x: [batch, patch_num, patch_len] - current patches
        mask: [batch, patch_num, patch_len] - current masks
        size: [batch, num_original_patches] - sizes of original patches

    Returns:
        original_patches: [batch, num_original_patches, patch_len] - reconstructed original patches
        original_mask: [batch, num_original_patches, patch_len] - reconstructed original masks
        parent_mapping: [batch, patch_num] - mapping from current patches to original patch indices
    """
    batch, patch_num, patch_len = x.shape
    num_original_patches = size.shape[1]

    # Verify expected structure
    assert patch_num == num_original_patches, f"patch_num {patch_num} != num_original_patches {num_original_patches}"

    # x and mask already represent one patch per original patch
    original_patches = x.clone()
    original_mask = mask.clone()

    # Create simple parent mapping: each patch maps to itself
    parent_mapping = torch.arange(patch_num, device=x.device).unsqueeze(0).expand(batch, -1)

    return original_patches, original_mask, parent_mapping

def _create_initial_position_mapping(x: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """
    Create initial position mapping for tracking patch positions within original patches.

    Args:
        x: [batch, patch_num, patch_len] - current patches
        size: [batch, num_original_patches] - sizes of original patches

    Returns:
        position_mapping: [batch, patch_num, 2] - [start_pos, end_pos] within original patch
    """
    batch, patch_num, patch_len = x.shape

    # Initially, each patch covers the entire original patch space
    # position_mapping stores [start_pos, end_pos] within the original patch
    position_mapping = torch.zeros((batch, patch_num, 2), dtype=torch.long, device=x.device)

    # Each patch initially spans the full range [0, patch_len)
    position_mapping[:, :, 0] = 0  # start_pos = 0
    position_mapping[:, :, 1] = patch_len  # end_pos = patch_len

    return position_mapping

def _update_parent_mapping(parent_mapping: torch.Tensor, to_divide: torch.Tensor,
                            div_counts: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Update parent mapping for divided patches.
    """
    batch, current_patch_num = parent_mapping.shape
    new_patch_nums = current_patch_num + div_counts
    max_new_patch_num = new_patch_nums.max()

    # Create repeat counts: 1 for non-divided patches, 2 for divided patches
    repeat_counts = 1 + to_divide.long()  # [batch, current_patch_num]

    # Initialize output tensor
    new_parent_mapping = torch.full((batch, max_new_patch_num), -1, dtype=torch.long, device=device)

    # Calculate positions for each element
    positions = torch.cumsum(repeat_counts, dim=1) - repeat_counts  # Starting position for each element
    batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1,
                                                                        current_patch_num)  # [batch, current_patch_num]

    # Place first copy (always exists)
    valid_mask_1 = positions < max_new_patch_num
    new_parent_mapping[batch_idx[valid_mask_1], positions[valid_mask_1]] = parent_mapping[valid_mask_1]

    # Place second copy (only for divided patches)
    second_positions = positions + 1
    valid_mask_2 = to_divide.bool() & (second_positions < max_new_patch_num)
    new_parent_mapping[batch_idx[valid_mask_2], second_positions[valid_mask_2]] = parent_mapping[valid_mask_2]

    return new_parent_mapping

def _update_position_mapping(position_mapping: torch.Tensor, to_divide: torch.Tensor,
                                div_counts: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Update position mapping for divided patches.
    """
    batch, current_patch_num, _ = position_mapping.shape
    new_patch_nums = current_patch_num + div_counts
    max_new_patch_num = new_patch_nums.max()

    # Create repeat counts: 1 for non-divided patches, 2 for divided patches
    repeat_counts = 1 + to_divide.long()  # [batch, current_patch_num]

    # Initialize output tensor
    new_position_mapping = torch.zeros((batch, max_new_patch_num, 2), dtype=torch.long, device=device)

    # Calculate positions for each element
    positions = torch.cumsum(repeat_counts, dim=1) - repeat_counts  # Starting position for each element
    batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1,
                                                                        current_patch_num)  # [batch, current_patch_num]

    # For non-divided patches: keep original position mapping
    valid_mask_1 = positions < max_new_patch_num
    new_position_mapping[batch_idx[valid_mask_1], positions[valid_mask_1]] = position_mapping[valid_mask_1]

    # For divided patches: split the position range
    second_positions = positions + 1
    valid_mask_2 = to_divide.bool() & (second_positions < max_new_patch_num)

    if valid_mask_2.any():
        # Get original start and end positions - correct indexing
        divided_position_mapping = position_mapping[valid_mask_2]  # [num_divided, 2]
        orig_start = divided_position_mapping[:, 0]  # [num_divided]
        orig_end = divided_position_mapping[:, 1]  # [num_divided]
        mid_pos = (orig_start + orig_end) // 2

        # Get batch and position indices for divided patches
        batch_indices_divided = batch_idx[valid_mask_2]  # [num_divided]
        first_positions_divided = positions[valid_mask_2]  # [num_divided]
        second_positions_divided = second_positions[valid_mask_2]  # [num_divided]

        # First half: [start, mid]
        new_position_mapping[batch_indices_divided, first_positions_divided, 0] = orig_start
        new_position_mapping[batch_indices_divided, first_positions_divided, 1] = mid_pos

        # Second half: [mid, end]
        new_position_mapping[batch_indices_divided, second_positions_divided, 0] = mid_pos
        new_position_mapping[batch_indices_divided, second_positions_divided, 1] = orig_end

    return new_position_mapping

def _map_to_parent_blocks(original_patches: torch.Tensor, parent_mapping: torch.Tensor,
                            target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Map parent indices to actual parent blocks.

    Args:
        original_patches: [batch, num_original_patches, patch_len] - original patch representations
        parent_mapping: [batch, new_patch_num] - mapping from new patches to original patch indices
        target_shape: (batch, new_patch_num, patch_len) - target shape

    Returns:
        parent_blocks: [batch, new_patch_num, patch_len] - parent blocks for each new patch
    """
    batch, new_patch_num, patch_len = target_shape

    # Create mask for valid indices
    valid_mask = parent_mapping >= 0  # [batch, new_patch_num]

    # Clamp negative indices to 0 to avoid indexing errors
    safe_parent_mapping = torch.clamp(parent_mapping, min=0)  # [batch, new_patch_num]

    # Use advanced indexing to gather parent blocks
    batch_indices = torch.arange(batch, device=original_patches.device).view(-1, 1)  # [batch, 1]
    parent_blocks = original_patches[batch_indices, safe_parent_mapping]  # [batch, new_patch_num, patch_len]

    # Zero out invalid positions using broadcasting
    parent_blocks = parent_blocks * valid_mask.unsqueeze(-1).float()

    return parent_blocks

def _create_granularity_mask(original_expert_indices: torch.Tensor, parent_mapping: torch.Tensor,
                                position_mapping: torch.Tensor, target_shape: Tuple[int, int, int],
                                original_patch_len: int) -> torch.Tensor:
    """
    Create granularity mask.

    Args:
        original_expert_indices: [batch, num_original_patches, n_experts] - expert indices for original patches
        parent_mapping: [batch, new_patch_num] - mapping from new patches to original patch indices
        position_mapping: [batch, new_patch_num, 2] - position ranges within original patches
        target_shape: (batch, new_patch_num, patch_len) - target shape
        original_patch_len: length of original patches

    Returns:
        granularity_mask: [batch, new_patch_num, max_granularity, original_patch_len] - granularity masks
    """
    batch, new_patch_num, patch_len = target_shape

    # Determine max granularity from expert indices
    max_granularity = original_expert_indices.shape[-1]

    granularity_mask = torch.zeros((batch, new_patch_num, max_granularity, original_patch_len),
                                    dtype=torch.float32, device=parent_mapping.device)

    # Create mask for valid patches
    valid_mask = parent_mapping >= 0  # [batch, new_patch_num]
    safe_parent_mapping = torch.clamp(parent_mapping, min=0)  # [batch, new_patch_num]

    # Get expert indices for each patch's parent
    batch_indices = torch.arange(batch, device=parent_mapping.device).view(-1, 1)  # [batch, 1]
    patch_expert_indices = original_expert_indices[
        batch_indices, safe_parent_mapping]  # [batch, new_patch_num, n_experts]

    # Get position ranges for each patch
    if hasattr(position_mapping, 'shape') and len(position_mapping.shape) == 3:
        start_positions = position_mapping[:, :, 0]  # [batch, new_patch_num]
        end_positions = position_mapping[:, :, 1]  # [batch, new_patch_num]
    else:
        # Fallback: compute positions based on patch structure
        start_positions, end_positions = _compute_patch_positions(
            parent_mapping, new_patch_num, original_patch_len, batch)

    # For each granularity level, generate masks
    for granularity in range(max_granularity):
        # Check which patches have this granularity level
        has_granularity = (patch_expert_indices == granularity).any(dim=-1) & valid_mask  # [batch, new_patch_num]

        if has_granularity.any():
            # Generate mask pattern based on granularity level
            mask_pattern = _generate_granularity_pattern(
                granularity, start_positions, end_positions, original_patch_len,
                has_granularity, max_granularity)

            granularity_mask[:, :, granularity, :] = mask_pattern

    return granularity_mask

def _compute_patch_positions(parent_mapping: torch.Tensor, new_patch_num: int,
                                original_patch_len: int, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute patch positions.
    """
    device = parent_mapping.device

    # For simplicity in this implementation, assume equal division within parents
    # This works for the given example structure

    # Initialize positions - fallback to simple equal division
    start_positions = torch.zeros((batch, new_patch_num), dtype=torch.long, device=device)
    end_positions = torch.full((batch, new_patch_num), original_patch_len, dtype=torch.long, device=device)

    # Valid mask
    valid_mask = parent_mapping >= 0

    if valid_mask.any():
        # Create a mask for each possible parent value using broadcasting
        max_parent = parent_mapping.max().item() if parent_mapping.numel() > 0 else 0
        parent_values = torch.arange(max_parent + 1, device=device)  # [max_parent + 1]

        # Broadcasting: [batch, new_patch_num, 1] == [max_parent + 1]
        parent_masks = (parent_mapping.unsqueeze(-1) == parent_values.unsqueeze(0).unsqueeze(
            0))  # [batch, new_patch_num, max_parent + 1]

        # Count patches per parent per batch - [batch, max_parent + 1]
        patches_per_parent = parent_masks.sum(dim=1)

        # Create cumulative indices within each parent group using broadcasting
        # For each parent, create relative positions
        cumsum_masks = torch.cumsum(parent_masks.float(), dim=1)  # [batch, new_patch_num, max_parent + 1]
        relative_indices = (cumsum_masks - 1) * parent_masks.float()  # [batch, new_patch_num, max_parent + 1]

        # Calculate patch sizes per parent - [batch, max_parent + 1]
        patch_sizes = original_patch_len // patches_per_parent.clamp(min=1)

        # Calculate start positions using broadcasting - [batch, new_patch_num, max_parent + 1]
        start_positions_expanded = (relative_indices * patch_sizes.unsqueeze(1)).long()
        end_positions_expanded = start_positions_expanded + patch_sizes.unsqueeze(1)

        # Sum across parent dimension (only one will be non-zero per patch)
        final_start_positions = start_positions_expanded.sum(dim=-1)  # [batch, new_patch_num]
        final_end_positions = end_positions_expanded.sum(dim=-1)  # [batch, new_patch_num]

        # Apply only to valid patches
        start_positions = torch.where(valid_mask, final_start_positions, start_positions)
        end_positions = torch.where(valid_mask, final_end_positions.clamp(max=original_patch_len), end_positions)

    return start_positions, end_positions

def _generate_granularity_pattern(granularity: int, start_positions: torch.Tensor,
                                    end_positions: torch.Tensor, original_patch_len: int,
                                    has_granularity: torch.Tensor, max_granularity: int) -> torch.Tensor:
    """
    Generate granularity pattern.
    """
    batch, new_patch_num = has_granularity.shape
    device = has_granularity.device

    mask_pattern = torch.zeros((batch, new_patch_num, original_patch_len), dtype=torch.float32, device=device)

    if has_granularity.any():
        if granularity == 0:
            # Coarsest granularity: activate entire patch
            mask_pattern[has_granularity] = 1.0

        else:
            # For other granularities, calculate activation length
            activation_length = original_patch_len // (2 ** granularity)
            activation_length = max(1, activation_length)  # Ensure at least 1

            # Get patches that have this granularity
            batch_idx, patch_idx = torch.where(has_granularity)

            if len(batch_idx) > 0:
                # Get start positions for patches with this granularity
                start_pos = start_positions[batch_idx, patch_idx]  # [num_patches_with_granularity]

                # Calculate slot indices for all patches
                slot_indices = start_pos // activation_length  # [num_patches_with_granularity]

                # Calculate activation ranges for all patches
                activation_starts = slot_indices * activation_length  # [num_patches_with_granularity]
                activation_ends = torch.clamp(activation_starts + activation_length,
                                                max=original_patch_len)  # [num_patches_with_granularity]

                # Use broadcasting to create range masks
                # Create position grid: [1, original_patch_len]
                pos_grid = torch.arange(original_patch_len, device=device).unsqueeze(0)  # [1, original_patch_len]

                # Expand activation ranges: [num_patches_with_granularity, 1]
                starts_expanded = activation_starts.unsqueeze(1)  # [num_patches_with_granularity, 1]
                ends_expanded = activation_ends.unsqueeze(1)  # [num_patches_with_granularity, 1]

                # Create range masks using broadcasting: [num_patches_with_granularity, original_patch_len]
                range_masks = (pos_grid >= starts_expanded) & (pos_grid < ends_expanded)

                # Apply range masks to corresponding positions in mask_pattern
                mask_pattern[batch_idx, patch_idx] = range_masks.float()

    return mask_pattern

def _generate_x_final(parent_blocks: torch.Tensor, parent_blocks_mask: torch.Tensor,
                        granularity_mask: torch.Tensor) -> torch.Tensor:
    """
    Generate x_final with rearranged features.

    Args:
        parent_blocks: [batch, patch_num, patch_len]
        parent_blocks_mask: [batch, patch_num, patch_len]
        granularity_mask: [batch, patch_num, max_granularity, patch_len]

    Returns:
        x_final: [max_granularity, batch*patch_num, max_feat_size * 2]
    """
    batch, patch_num, patch_len = parent_blocks.shape
    max_granularity = granularity_mask.shape[2]
    total_patches = batch * patch_num

    # Assume we have predefined feature sizes for each granularity level
    # This should be consistent with self.in_features_ls
    feat_sizes = [patch_len // (2 ** i) for i in range(max_granularity)]
    feat_sizes = [max(1, size) for size in feat_sizes]  # Ensure at least 1
    max_feat_size = max(feat_sizes) if feat_sizes else patch_len

    # Initialize output tensor
    x_final = torch.zeros((max_granularity, total_patches, max_feat_size * 2),
                            dtype=parent_blocks.dtype, device=parent_blocks.device)

    # Flatten spatial dimensions for easier processing
    parent_blocks_flat = parent_blocks.reshape(total_patches, patch_len)  # [total_patches, patch_len]
    parent_blocks_mask_flat = parent_blocks_mask.reshape(total_patches, patch_len)  # [total_patches, patch_len]
    granularity_mask_flat = granularity_mask.reshape(total_patches, max_granularity,
                                                        patch_len)  # [total_patches, max_granularity, patch_len]

    # Process each granularity level
    for granularity in range(max_granularity):
        feat_size = feat_sizes[granularity]

        # Get granularity mask for this level
        current_granularity_mask = granularity_mask_flat[:, granularity, :]  # [total_patches, patch_len]

        # Apply granularity mask to parent blocks and parent blocks mask
        masked_parent_blocks = parent_blocks_flat * current_granularity_mask  # [total_patches, patch_len]
        masked_parent_blocks_mask = parent_blocks_mask_flat * current_granularity_mask  # [total_patches, patch_len]

        # Rearrange each part separately to maintain structure
        rearranged_parent_blocks = _rearrange_effective_values(
            masked_parent_blocks, current_granularity_mask, feat_size, max_feat_size)

        rearranged_parent_blocks_mask = _rearrange_effective_values(
            masked_parent_blocks_mask, current_granularity_mask, feat_size, max_feat_size)

        # Combine rearranged parts: [parent_blocks | parent_blocks_mask]
        rearranged_features = torch.cat([rearranged_parent_blocks, rearranged_parent_blocks_mask], dim=-1)

        # Store in x_final
        x_final[granularity] = rearranged_features

    return x_final

def _rearrange_effective_values(features: torch.Tensor, mask: torch.Tensor,
                                target_feat_size: int, max_feat_size: int) -> torch.Tensor:
    """
    Rearrange effective values to front positions.

    Args:
        features: [total_patches, feat_dim] - input features (single part: either parent_blocks or parent_blocks_mask)
        mask: [total_patches, feat_dim] - mask indicating effective positions
        target_feat_size: actual feature size for this granularity
        max_feat_size: maximum feature size across all granularities (for output shape consistency)

    Returns:
        rearranged: [total_patches, max_feat_size] - rearranged features (single part)
    """
    total_patches, feat_dim = features.shape
    device = features.device

    # Initialize output with max_feat_size to ensure consistent shape
    rearranged = torch.zeros((total_patches, max_feat_size), dtype=features.dtype, device=device)

    # Find effective positions
    effective_mask = mask > 0  # [total_patches, feat_dim]

    # Create position indices for sorting
    position_indices = torch.arange(feat_dim, device=device).unsqueeze(0).expand(total_patches,
                                                                                    -1)  # [total_patches, feat_dim]

    # Sort to get effective positions first - use large value for invalid positions
    masked_positions = torch.where(effective_mask, position_indices.float(),
                                    float('inf'))  # [total_patches, feat_dim]
    sorted_positions, sort_indices = torch.sort(masked_positions, dim=1)  # [total_patches, feat_dim]

    # Count effective values per patch
    effective_counts = effective_mask.sum(dim=1)  # [total_patches]

    # Create output position indices using broadcasting - limit to target_feat_size
    output_indices = torch.arange(target_feat_size, device=device).unsqueeze(0)  # [1, target_feat_size]

    # Determine which output positions should be filled
    take_counts = torch.clamp(effective_counts, max=target_feat_size).unsqueeze(1)  # [total_patches, 1]
    valid_output_mask = output_indices < take_counts  # [total_patches, target_feat_size]

    # Get all valid (batch, output_pos) pairs
    batch_indices, output_positions = torch.where(valid_output_mask)  # [num_valid], [num_valid]

    if len(batch_indices) > 0:
        # Get corresponding source
        source_indices = sort_indices[batch_indices, output_positions]  # [num_valid]

        # Gather source values
        source_values = features[batch_indices, source_indices]  # [num_valid]

        # Place values in output (only in the first target_feat_size positions)
        rearranged[batch_indices, output_positions] = source_values

    return rearranged