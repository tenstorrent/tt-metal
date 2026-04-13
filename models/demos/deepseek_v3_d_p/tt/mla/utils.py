# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch


def create_balanced_chunk_order(sp_factor: int) -> list[int]:
    """Create balanced chunk order for sequence reordering.

    For sp_factor=4, creates 2*4=8 chunks with order: 0,7,1,6,2,5,3,4
    This interleaves chunks from start and end to balance workload.
    """
    num_chunks = 2 * sp_factor
    balanced_order = []

    left = 0
    right = num_chunks - 1

    for i in range(num_chunks):
        if i % 2 == 0:
            balanced_order.append(left)
            left += 1
        else:
            balanced_order.append(right)
            right -= 1

    return balanced_order


def reorder_tensor_chunks(tensor: torch.Tensor, chunk_order: list[int], seq_dim: int = 2) -> torch.Tensor:
    """Reorder tensor chunks along sequence dimension according to chunk_order."""
    seq_len = tensor.shape[seq_dim]
    num_chunks = len(chunk_order)
    chunk_size = seq_len // num_chunks

    # Split into chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        if seq_dim == 2:
            chunks.append(tensor[:, :, start:end, :])
        else:
            raise NotImplementedError(f"Reordering for seq_dim={seq_dim} not implemented")

    # Reorder chunks according to chunk_order
    reordered_chunks = [chunks[i] for i in chunk_order]

    # Concatenate reordered chunks
    return torch.cat(reordered_chunks, dim=seq_dim)


def reverse_reorder_tensor_chunks(tensor: torch.Tensor, chunk_order: list[int], seq_dim: int = 2) -> torch.Tensor:
    """Reverse the chunk reordering to restore original order."""
    # Create inverse permutation
    inverse_order = [0] * len(chunk_order)
    for new_pos, orig_pos in enumerate(chunk_order):
        inverse_order[orig_pos] = new_pos

    return reorder_tensor_chunks(tensor, inverse_order, seq_dim)


def global_to_local_token_id(global_token_id: int, sp_factor: int, seq_len: int) -> tuple[int, int]:
    """Convert a global token ID to a device ID and local token ID under zigzag (striped) attention.

    In zigzag attention, the sequence is split into 2*sp_factor chunks. Device k holds
    chunks k and (2*sp_factor - 1 - k), balancing causal attention workload across devices.

    Args:
        global_token_id: The global token position across the full sequence.
        sp_factor: Number of devices in the sequence parallel group.
        seq_len: Total sequence length across all devices.

    Returns:
        A tuple of (device_id, local_token_id).
    """
    num_chunks = 2 * sp_factor
    chunk_size = seq_len // num_chunks
    chunk_id = global_token_id // chunk_size
    offset_in_chunk = global_token_id % chunk_size

    if chunk_id < sp_factor:
        device_id = chunk_id
        local_token_id = offset_in_chunk
    else:
        device_id = num_chunks - 1 - chunk_id
        local_token_id = chunk_size + offset_in_chunk

    return device_id, local_token_id
