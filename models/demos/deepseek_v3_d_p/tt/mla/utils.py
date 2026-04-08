# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn


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


def zero_cache_padding_zigzag(
    kvpe_cache: ttnn.Tensor,
    global_end_token: int,
    sp_factor: int,
    seq_len: int,
    decode_chunk_align: int = 128,
):
    """Zero-pad KV cache for migration alignment under zigzag attention layout.

    After prefill writes valid tokens to the cache, this function zeroes the
    padding region from global_end_token up to the next decode_chunk_align
    boundary. It handles the zigzag token mapping to dispatch per-device
    zero_cache_range calls with correct local token ranges.

    Args:
        kvpe_cache: The mesh KV cache tensor (one shard per SP device).
        global_end_token: First invalid global token position (effective seq_len).
        sp_factor: Number of devices in the sequence parallel group.
        seq_len: Total sequence length across all devices.
        decode_chunk_align: Decode chunk alignment in tokens (default 128).
    """
    padded_end = math.ceil(global_end_token / decode_chunk_align) * decode_chunk_align
    padded_end = min(padded_end, seq_len)

    if global_end_token >= padded_end:
        return

    chunk_size = seq_len // (2 * sp_factor)
    tile_size = 32  # KV cache tile height

    # Accumulate local token ranges per device
    # Each entry: [local_start, local_end) — we track min start and max end per device
    device_ranges: dict[int, tuple[int, int]] = {}

    # Walk the global padding region in tile-sized steps (32 tokens)
    # Round global_end_token down to tile boundary for the start
    global_start = (global_end_token // tile_size) * tile_size
    for global_tok in range(global_start, padded_end, tile_size):
        device_id, local_token_id = global_to_local_token_id(global_tok, sp_factor, seq_len)
        local_tile_start = (local_token_id // tile_size) * tile_size
        local_tile_end = local_tile_start + tile_size

        if device_id not in device_ranges:
            device_ranges[device_id] = (local_tile_start, local_tile_end)
        else:
            prev_start, prev_end = device_ranges[device_id]
            device_ranges[device_id] = (min(prev_start, local_tile_start), max(prev_end, local_tile_end))

    # Dispatch per-device zero_cache_range
    device_tensors = ttnn.get_device_tensors(kvpe_cache)
    for device_id, (local_start, local_end) in device_ranges.items():
        ttnn.kv_cache.zero_cache_range(device_tensors[device_id], local_start, local_end)
