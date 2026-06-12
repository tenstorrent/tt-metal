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


def reorder_tensor_chunks(tensor: torch.Tensor, chunk_order: list[int], seq_dim: int = -2) -> torch.Tensor:
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
        elif seq_dim == -2:
            chunks.append(tensor[..., start:end, :])
        else:
            raise NotImplementedError(f"Reordering for seq_dim={seq_dim} not implemented")

    # Reorder chunks according to chunk_order
    reordered_chunks = [chunks[i] for i in chunk_order]

    # Concatenate reordered chunks
    return torch.cat(reordered_chunks, dim=seq_dim)


def reverse_reorder_tensor_chunks(tensor: torch.Tensor, chunk_order: list[int], seq_dim: int = -2) -> torch.Tensor:
    """Reverse the chunk reordering to restore original order."""
    # Create inverse permutation
    inverse_order = [0] * len(chunk_order)
    for new_pos, orig_pos in enumerate(chunk_order):
        inverse_order[orig_pos] = new_pos

    return reorder_tensor_chunks(tensor, inverse_order, seq_dim)


def block_cyclic_reorder(matrix: torch.Tensor, chunk_local: int, sp_factor: int, seq_dim: int = 2) -> torch.Tensor:
    """Reorder a [.., seq, ..] matrix into block-cyclic order keyed by `chunk_local`.

    Splits the sequence into blocks of `chunk_local` rows and concatenates them so that device c's
    contiguous shard (after a plain SP shard over `seq_dim`) holds blocks c, c+sp, c+2sp, ... — the
    same block-cyclic layout the per-chip KV cache writes into. This makes the indexed-RoPE op's
    contiguous, `update_idxt`-offset read of each device's cos/sin shard land on the right global
    positions, including the boundary chip's older-then-wrap rows.
    """
    seq_len = matrix.shape[seq_dim]
    assert seq_len % chunk_local == 0, f"seq_len {seq_len} must be a multiple of chunk_local {chunk_local}"
    num_blocks = seq_len // chunk_local
    assert num_blocks % sp_factor == 0, f"num_blocks {num_blocks} must be a multiple of sp_factor {sp_factor}"
    blocks = list(torch.split(matrix, chunk_local, dim=seq_dim))
    order = [b for c in range(sp_factor) for b in range(c, num_blocks, sp_factor)]
    return torch.cat([blocks[b] for b in order], dim=seq_dim)


def rotated_chip_positions(kv_actual_isl: int, sp: int, chunk_local: int) -> list[list[int]]:
    """Per-chip global position for each chip-local row after the server's KV-pad-aware rotation,
    mirroring the writer kernel EXACTLY. positions[c][r] is the global token position carried by
    chip c's r-th rotated row.

    Slab-aware: each chip writes chunk_local rows starting at update_idxt, and the cache cell at
    local row lr on chip c holds global position (lr // chunk_local)*chunk_size_global +
    c*chunk_local + (lr % chunk_local). The union over all (c, r) tiles
    [kv_actual_isl, kv_actual_isl + chunk_size_global) exactly, so the first new_actual_isl
    positions are the contiguous valid range and the rest are pad (masked by causality) -- no
    separate new_actual_isl plumbing needed.
    """
    chunk_size_global = sp * chunk_local
    boundary_slab = kv_actual_isl // chunk_size_global
    boundary_chip = (kv_actual_isl // chunk_local) % sp
    boundary_offset = kv_actual_isl % chunk_local

    positions = [[0] * chunk_local for _ in range(sp)]
    for c in range(sp):
        if c < boundary_chip:
            update_idxt = (boundary_slab + 1) * chunk_local
        elif c == boundary_chip:
            update_idxt = boundary_slab * chunk_local + boundary_offset
        else:
            update_idxt = boundary_slab * chunk_local
        for r in range(chunk_local):
            lr = update_idxt + r
            positions[c][r] = (lr // chunk_local) * chunk_size_global + c * chunk_local + (lr % chunk_local)
    return positions


def blockcyclic_positions(sp: int, chunk_size_global: int, seq_len_cache: int) -> torch.Tensor:
    """Global natural position held by each block-cyclic shard row (device-major: an SP-contiguous
    split of the cache's seq dim yields each chip's rows).

    Shard row r -> chip c = r // seq_len_local, local row lr = r % seq_len_local, and that row holds
    global position (lr // chunk_local) * chunk_size_global + c * chunk_local + (lr % chunk_local) --
    the inverse of the update_padded_kv_cache writer. Returns a [seq_len_cache] index tensor.
    """
    seq_len_local = seq_len_cache // sp
    chunk_local = chunk_size_global // sp
    c = torch.arange(sp).repeat_interleave(seq_len_local)
    lr = torch.arange(seq_len_local).repeat(sp)
    slab, off = lr // chunk_local, lr % chunk_local
    return slab * chunk_size_global + c * chunk_local + off


def blockcyclic_cache_host(
    kv_natural: torch.Tensor, sp: int, chunk_size_global: int, seq_len_cache: int, kvpe_dim: int
) -> torch.Tensor:
    """Arrange a natural-order [seq, kvpe] KV tensor into the device cache's block-cyclic slab-major
    layout so that an SP-contiguous shard of dim 2 gives each chip its own rows. Rows whose global
    position is beyond the provided KV (e.g. the not-yet-written new chunk) stay zero. Returns
    [1, 1, seq_len_cache, kvpe_dim].
    """
    prior_len = kv_natural.shape[0]
    p = blockcyclic_positions(sp, chunk_size_global, seq_len_cache)
    out = torch.zeros(seq_len_cache, kvpe_dim, dtype=kv_natural.dtype)
    valid = p < prior_len
    out[valid] = kv_natural[p[valid]]
    return out.reshape(1, 1, seq_len_cache, kvpe_dim)


def global_to_local_token_id(
    global_token_id: int,
    sp_factor: int,
    seq_len: int,
    is_balanced: bool = True,
) -> tuple[int, int]:
    """Convert a global token ID to a device ID and local token ID.

    Args:
        global_token_id: The global token position across the full sequence.
        sp_factor: Number of devices in the sequence parallel group.
        seq_len: Total sequence length across all devices.
        is_balanced: If True (default), uses zigzag (striped) attention where the sequence
            is split into 2*sp_factor chunks and device k holds chunks k and (2*sp_factor - 1 - k),
            balancing causal attention workload. If False, uses sequential distribution where
            the sequence is split into sp_factor chunks and device k holds chunk k.

    Returns:
        A tuple of (device_id, local_token_id).
    """
    if is_balanced:
        # Zigzag/balanced: num_chunks = 2 * sp_factor
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
    else:
        # Sequential: num_chunks = sp_factor
        num_chunks = sp_factor
        chunk_size = seq_len // num_chunks
        chunk_id = global_token_id // chunk_size
        offset_in_chunk = global_token_id % chunk_size
        device_id = chunk_id
        local_token_id = offset_in_chunk

    return device_id, local_token_id
