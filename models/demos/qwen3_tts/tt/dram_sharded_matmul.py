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


def _wormhole_mesh_on_mmio_multi_chip_board(device) -> bool:
    """N300-style board: multiple WH chips share one host MMIO while ``open_device`` still yields a 1x1 mesh."""
    try:
        if device.arch() != ttnn.device.Arch.WORMHOLE_B0:
            return False
        n = int(ttnn.device.GetNumAvailableDevices())
        if n < 2:
            return False
        get_ids = getattr(device, "get_device_ids", None)
        if not callable(get_ids):
            return False
        mmio_parent = [int(ttnn.device.GetPCIeDeviceID(i)) for i in range(n)]
        peer_count: dict[int, int] = {}
        for m in mmio_parent:
            peer_count[m] = peer_count.get(m, 0) + 1
        return any(peer_count.get(int(ttnn.device.GetPCIeDeviceID(int(cid))), 0) > 1 for cid in get_ids())
    except Exception:
        return False


def mesh_dram_shard_decode_matmul_ok(device) -> bool:
    """Allow DRAM-sharded decode matmul only for safe single-logical-device paths (see N300 MMIO note above)."""
    try:
        get_n = getattr(device, "get_num_devices", None)
        if callable(get_n) and int(get_n()) > 1:
            return False
        sh = device.shape
        if int(sh[0]) * int(sh[1]) > 1:
            return False
    except Exception:
        pass
    return not _wormhole_mesh_on_mmio_multi_chip_board(device)


def _largest_divisor(n: int, max_divisor: int = 8) -> int:
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def find_grid_k_n(
    K_tiles: int,
    N_tiles: int,
    max_rows: int = 10,
    max_cols: int = 13,
    *,
    K2_tiles: int | None = None,
    N2_tiles: int | None = None,
) -> Tuple[int, int]:
    """Largest grid whose core count divides K/N tile axes (and optionally a second K/N pair for decode MLP).

    On Wormhole pass ``max_rows`` / ``max_cols`` from ``compute_with_storage_grid_size()`` so shards fit
    harvested grids (often 8×7). Defaults suit large grids (e.g. Blackhole).
    """
    if (K2_tiles is None) ^ (N2_tiles is None):
        raise ValueError("K2_tiles and N2_tiles must be passed together or not at all")

    def divides_all(c: int) -> bool:
        if K_tiles % c or N_tiles % c:
            return False
        if K2_tiles is not None:
            assert N2_tiles is not None
            return K2_tiles % c == 0 and N2_tiles % c == 0
        return True

    max_cores = max_rows * max_cols
    candidates = [c for c in range(1, max_cores + 1) if divides_all(c)]
    candidates.sort(reverse=True)
    for cores in candidates:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    extra = f", K2={K2_tiles}, N2={N2_tiles}" if K2_tiles is not None else ""
    raise AssertionError(f"No grid divides K={K_tiles}, N={N_tiles}{extra} tiles within {max_rows}x{max_cols}.")


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
