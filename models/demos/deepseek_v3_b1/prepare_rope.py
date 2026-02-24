# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Prepare RoPE (Rotary Position Embedding) tensors for DeepSeek V3 B1 ops.

Provides the transformation matrix and cos/sin tables in the layout expected by
the pre-SDPA fused op and RopeSingleCore. See test_pre_sdpa.py and test_rope.py
for how these tensors are consumed.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3.tt.rope import get_rot_transformation_mat  # re-exported for callers

# RoPE dimensions (match test_pre_sdpa / pre-SDPA op)
ROPE_HEAD_DIM = 64
ROPE_TRANS_TILE = (32, 32)  # ttnn.TILE_SIZE x ttnn.TILE_SIZE

# Grid constants for pre-SDPA trans_mat (match test_pre_sdpa)
QNOPE_GRID_COLS = 8
QROPE_GRID_COLS = 4
MATMUL2_GRID_Y = 8
KV_CACHE_BRANCH_START_OFFSET = (0, 8)


def get_rope_trans_mat_core_range_set(device_grid_size: ttnn.CoreCoord) -> ttnn.CoreRangeSet:
    """Build the core range set for the RoPE transformation matrix (pre-SDPA layout).

    Combines qrope_grid (QROPE_GRID_COLS x MATMUL2_GRID_Y) and kv_cache_branch_rope_crs (2 cores).
    Layout matches test_pre_sdpa.py so each core holds one [32, 32] shard.
    """
    qrope_grid = ttnn.CoreRange(
        ttnn.CoreCoord(QNOPE_GRID_COLS, 0),
        ttnn.CoreCoord(QNOPE_GRID_COLS + QROPE_GRID_COLS - 1, MATMUL2_GRID_Y - 1),
    )
    kv_ox, kv_oy = KV_CACHE_BRANCH_START_OFFSET
    kv_cache_branch_rope_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(8 + kv_ox, kv_oy),
                ttnn.CoreCoord(8 + kv_ox, 1 + kv_oy),
            )
        }
    )
    return kv_cache_branch_rope_crs.merge(ttnn.CoreRangeSet({qrope_grid}))


def create_rope_trans_mat_tensor(device) -> ttnn.Tensor:
    """Build the RoPE transformation matrix in pre-SDPA layout.

    Returns a ttnn tensor: HEIGHT_SHARDED L1, one (32, 32) shard per core over
    qrope_grid + kv_cache_branch_rope_crs, replicated on mesh. Layout matches
    the pre-SDPA fused op (test_pre_sdpa.py).
    """
    device_grid_size = device.compute_with_storage_grid_size()
    trans_mat_crs = get_rope_trans_mat_core_range_set(device_grid_size)
    num_cores = trans_mat_crs.num_cores()

    trans_mat = get_rot_transformation_mat().repeat(1, 1, num_cores, 1)
    trans_shard_spec = ttnn.ShardSpec(
        trans_mat_crs,
        (ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    trans_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, trans_shard_spec)
    trans_tile = ttnn.Tile((ttnn.TILE_SIZE, ttnn.TILE_SIZE))
    return ttnn.from_torch(
        trans_mat.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=trans_mem_config,
        tile=trans_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def get_cos_sin_torch(
    max_seq_len: int,
    head_dim: int = ROPE_HEAD_DIM,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos and sin tensors for RoPE in Meta-style format.

    Shape: (cos, sin) each [1, 1, max_seq_len, head_dim].
    Formula matches test_pre_sdpa.py and test_rope.py (base=10000, inv_freq, outer).
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)
    sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).flatten(-2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return cos, sin


def create_rope_cos_sin_ttnn(
    device,
    max_seq_len: int,
    head_dim: int = ROPE_HEAD_DIM,
    base: float = 10000.0,
    tile: ttnn.Tile | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Build RoPE cos/sin as ttnn tensors in pre-SDPA layout.

    Returns (ttnn_cos, ttnn_sin): DRAM INTERLEAVED, replicated on mesh, shape
    [1, 1, max_seq_len, head_dim]. Same layout used for QRoPE and KRoPE in
    test_pre_sdpa.py (both can use this single pair).
    """
    if tile is None:
        tile = ttnn.Tile([1, 32])
    cos_torch, sin_torch = get_cos_sin_torch(max_seq_len, head_dim=head_dim, base=base)
    dram_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    ttnn_cos = ttnn.from_torch(
        cos_torch.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    ttnn_sin = ttnn.from_torch(
        sin_torch.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_mem_config,
        tile=tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    return ttnn_cos, ttnn_sin
