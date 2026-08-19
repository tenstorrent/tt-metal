# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
DRAM-sharded matmul helpers for Qwen3-TTS decode-mode projections.

Mirrors the pattern used in models/tt_transformers/tt/model_config.py:
  * weights:       width-sharded across the device's DRAM banks (e.g. 12 banks on
                   wormhole_b0). N is padded up to a multiple of TILE * dram_cores.
  * activation:    width-sharded in L1 across the compute grid along K.
  * program:       MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig.

This keeps the activation in L1, parallelises weight DRAM reads across all banks,
and removes the implicit DRAM read of in0 every matmul that the interleaved 1D
multicast path performs.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

import ttnn

TILE = ttnn.TILE_SIZE  # 32


def _largest_divisor(n: int, max_divisor: int = 8) -> int:
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def find_grid_k_n(K_tiles: int, N_tiles: int, max_rows: int = 10, max_cols: int = 13) -> Tuple[int, int]:
    """Largest core grid (rows, cols) where num_cores divides BOTH K_tiles and N_tiles.

    Both divisibility constraints are needed: K for `in0_block_w`, N for the
    width-sharded output (each core must own a tile-aligned N slice).

    Default max grid bumped to 13×10 to support Blackhole's 130-core compute grid;
    callers on Wormhole get the same code path (constraints just cap at 64 anyway).
    """
    max_cores = max_rows * max_cols
    candidates = [c for c in range(1, max_cores + 1) if K_tiles % c == 0 and N_tiles % c == 0]
    candidates.sort(reverse=True)
    for cores in candidates:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise AssertionError(f"No grid divides both K={K_tiles} and N={N_tiles} tiles within {max_rows}x{max_cols}.")


def pad_n_for_dram_align(n: int, dram_cores: int) -> int:
    """Pad N up to a multiple of TILE*dram_cores so the weight shards evenly across DRAM banks."""
    align = TILE * dram_cores
    return ((n + align - 1) // align) * align


def dram_sharded_weight_memcfg(k: int, n_padded: int, device) -> ttnn.MemoryConfig:
    """Build a WIDTH_SHARDED DRAM memory config for a [k, n_padded] weight."""
    dram_grid_size = device.dram_grid_size()
    dram_cores = dram_grid_size.x
    assert dram_grid_size.y == 1, "DRAM-sharded weights assume y=1 dram grid"
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    assert n_padded % dram_cores == 0, f"N={n_padded} not divisible by dram_cores={dram_cores}"
    shard_spec = ttnn.ShardSpec(dram_grid, (k, n_padded // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def width_sharded_l1_memcfg(m_tiles: int, k_tiles: int, num_cores_x: int, num_cores_y: int) -> ttnn.MemoryConfig:
    """Width-shard a [1,1,m_tiles*32, k_tiles*32] activation across the compute grid (row-major)."""
    num_cores = num_cores_x * num_cores_y
    assert k_tiles % num_cores == 0, f"K_tiles={k_tiles} not divisible by num_cores={num_cores}"
    core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
    )
    shard_spec = ttnn.ShardSpec(
        core_grid,
        (m_tiles * TILE, (k_tiles // num_cores) * TILE),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def dram_sharded_program_config(
    m: int, k: int, n: int, num_cores: int, fused_activation=None
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    assert k % (TILE * num_cores) == 0, f"K={k} not divisible by TILE*num_cores={TILE*num_cores}"
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_largest_divisor(k // (TILE * num_cores)),
        per_core_M=math.ceil(m / TILE),
        per_core_N=math.ceil(n / (TILE * num_cores)),
        fused_activation=fused_activation,
    )


def build_dram_sharded_weight(
    weight_torch_2d: torch.Tensor,
    device,
    dtype=ttnn.bfloat16,
) -> Tuple[ttnn.Tensor, int, int]:
    """
    Build a DRAM-sharded weight tensor from a 2D torch weight [K, N_unpadded].

    Returns (ttnn_tensor, k, n_padded). Caller drives the matching program config.
    """
    assert weight_torch_2d.dim() == 2
    k, n_unpadded = weight_torch_2d.shape
    dram_cores = device.dram_grid_size().x
    n_padded = pad_n_for_dram_align(n_unpadded, dram_cores)
    if n_padded != n_unpadded:
        pad_w = torch.zeros(k, n_padded - n_unpadded, dtype=weight_torch_2d.dtype)
        weight_torch_2d = torch.cat([weight_torch_2d, pad_w], dim=1).contiguous()
    weight_4d = weight_torch_2d.unsqueeze(0).unsqueeze(0).contiguous()
    memcfg = dram_sharded_weight_memcfg(k, n_padded, device)
    tt = ttnn.from_torch(
        weight_4d,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memcfg,
    )
    return tt, k, n_padded


def build_dram_sharded_weight_tp(
    weight_kn_full: "torch.Tensor",
    device,
    tp_size: int,
    split_dim: int,
    dtype=ttnn.bfloat16,
) -> Tuple["ttnn.Tensor", int, int]:
    """Build DRAM-sharded per-chip weight for TP>1, distributed across a mesh.

    split_dim=1 (column-parallel): weight [K, N_full], each chip gets [K, N_full/tp].
    split_dim=0 (row-parallel):    weight [K_full, N], each chip gets [K_full/tp, N].

    Returns (ttnn_tensor, k_per_chip, n_padded_per_chip).
    """
    import torch as _torch

    dram_cores = device.dram_grid_size().x

    if split_dim == 1:
        K = int(weight_kn_full.shape[0])
        N_per_chip = int(weight_kn_full.shape[1]) // tp_size
        n_padded = pad_n_for_dram_align(N_per_chip, dram_cores)
        if n_padded == N_per_chip:
            host = weight_kn_full.contiguous()  # [K, tp*N_per_chip] — already aligned
        else:
            chunks = list(_torch.chunk(weight_kn_full, tp_size, dim=1))
            pads = [_torch.zeros(K, n_padded, dtype=chunks[0].dtype) for _ in chunks]
            for i, c in enumerate(chunks):
                pads[i][:, : c.shape[1]] = c
            host = _torch.cat(pads, dim=1)  # [K, tp*n_padded]
        memcfg = dram_sharded_weight_memcfg(K, n_padded, device)
        tt = ttnn.from_torch(
            host,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memcfg,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
        )
        return tt, K, n_padded

    else:  # split_dim == 0 (row-parallel)
        K_per_chip = int(weight_kn_full.shape[0]) // tp_size
        N = int(weight_kn_full.shape[1])
        n_padded = pad_n_for_dram_align(N, dram_cores)
        if n_padded == N:
            host = weight_kn_full.contiguous()  # [K_full, N]
        else:
            host = _torch.zeros(int(weight_kn_full.shape[0]), n_padded, dtype=weight_kn_full.dtype)
            host[:, :N] = weight_kn_full
        memcfg = dram_sharded_weight_memcfg(K_per_chip, n_padded, device)
        tt = ttnn.from_torch(
            host,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memcfg,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
        )
        return tt, K_per_chip, n_padded
