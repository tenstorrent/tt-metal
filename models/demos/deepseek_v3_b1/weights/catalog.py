# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek **artifact targets** and layout constants (**ArtifactTarget** / ``TensorTarget``).

Source key naming and HF conventions for fingerprint **SourceSelection** are implied by the
``prepare_*`` call sites in :mod:`models.demos.deepseek_v3_b1.weights.adapter`. Generic cache types
live in :mod:`models.demos.deepseek_v3_b1.tensor_cache`.
"""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import DOWN_PROJ_SINGLE_DEVICE_SPEC, BlitzDecodeWeights
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions as D
from models.demos.deepseek_v3_b1.tensor_cache import (
    ReplicateMeshMapper,
    Shard2dMeshMapper,
    ShardMeshMapper,
    TensorTarget,
)
from models.demos.deepseek_v3_b1.weights.types import _MTP_NUM_DRAM_BANKS

# Bump when any standalone tensor preprocessing logic changes to invalidate caches.
CURRENT_TRANSFORM_VERSION = 1

# MoE sender core: hardcoded grid (13, 10) so cache layout is consistent across slow/fast dispatch.
MOE_SENDER_GRID_SIZE = (13, 10)
_GATE_BIAS_TILE = ttnn.Tile([16, 16])

_GATE_BIAS_SENDER_CORE = ttnn.CoreCoord(MOE_SENDER_GRID_SIZE[0] - 1, MOE_SENDER_GRID_SIZE[1] - 1)
_GATE_BIAS_SENDER_CORE_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(_GATE_BIAS_SENDER_CORE, _GATE_BIAS_SENDER_CORE)])

_LM_HEAD_K = D.HIDDEN_SIZE
_LM_HEAD_VOCAB_SIZE = D.VOCAB_SIZE
_LM_HEAD_MATMUL_CORE_GRID = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
    ]
)
_LM_HEAD_B_TILE = ttnn.Tile([32, 32])
_LM_HEAD_A_TILE = ttnn.Tile([1, 32])
_LM_HEAD_N_PER_CORE = 160
_LM_HEAD_MCAST_CORE = ttnn.CoreCoord(10, 9)
_LM_HEAD_MCAST_CORE_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(_LM_HEAD_MCAST_CORE, _LM_HEAD_MCAST_CORE)])

_NORM_MEM_CONFIG = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(_LM_HEAD_MCAST_CORE_GRID, (1, _LM_HEAD_K), ttnn.ShardOrientation.ROW_MAJOR),
)


def gate_bias_target(layer_idx: int) -> TensorTarget:
    return TensorTarget(
        name=f"gate_bias_layer{layer_idx}",
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_GATE_BIAS_SENDER_CORE_GRID, (16, 16), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile_shape=(16, 16),
    )


EMBEDDING_TARGET = TensorTarget(
    name="embedding",
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

LM_HEAD_TARGET = TensorTarget(
    name="lm_head",
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_LM_HEAD_MATMUL_CORE_GRID, (_LM_HEAD_K, _LM_HEAD_N_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR),
    ),
    mesh_mapper_config=ShardMeshMapper(dim=1),
)

FINAL_NORM_TARGET = TensorTarget(
    name="final_norm",
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=_NORM_MEM_CONFIG,
    tile_shape=(1, 32),
)


def mtp_norm_target(name: str) -> TensorTarget:
    return TensorTarget(
        name=name,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_NORM_MEM_CONFIG,
        tile_shape=(1, 32),
    )


def mtp_eh_proj_target(K: int, N: int) -> TensorTarget:
    n_per_bank = N // _MTP_NUM_DRAM_BANKS
    eh_shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_MTP_NUM_DRAM_BANKS - 1, 0))}
    )
    return TensorTarget(
        name="mtp_eh_projection",
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(eh_shard_grid, (K, n_per_bank), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )


def shared_down_tensor_target(bdw: BlitzDecodeWeights) -> TensorTarget:
    """TensorTarget for shared expert down (L1 WIDTH_SHARDED on matmul cores, bfloat4_b)."""
    dp_spec = DOWN_PROJ_SINGLE_DEVICE_SPEC
    K_down_per_device = 256
    N_per_core = 64
    matmul_core_grid = dp_spec.build_matmul_core_grid()
    dp_shard_spec = ttnn.ShardSpec(matmul_core_grid, (K_down_per_device, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    dp_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, dp_shard_spec)
    moe_tp = bdw.moe_tp
    if moe_tp == 1:
        mmc = ReplicateMeshMapper()
    else:
        mmc = Shard2dMeshMapper(dims=(0, 1))
    return TensorTarget(
        name="shared_down_proj",
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dp_mem,
        tile_shape=(32, 32),
        mesh_mapper_config=mmc,
    )


def moe_routed_expert_tensor_target(name: str, K: int, N: int, device) -> TensorTarget:
    """TensorTarget for one MoE routed expert projection (DRAM WIDTH_SHARDED, bfloat4_b)."""
    tile_w = 32
    num_banks = device.dram_grid_size().x
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    per_core_N = N_padded // num_banks
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
            )
        }
    )
    shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    return TensorTarget(
        name=name,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
        tile_shape=(32, 32),
        mesh_mapper_config=ReplicateMeshMapper(),
    )


def dense_routed_stacked_tensor_target(name: str, K: int, N: int, device) -> TensorTarget:
    """TensorTarget for dense MLP routed projection (all experts stacked on mesh)."""
    tile_w = 32
    num_banks = device.dram_grid_size().x
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    per_core_N = N_padded // num_banks
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
            )
        }
    )
    shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    return TensorTarget(
        name=name,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem_config,
        tile_shape=(32, 32),
        mesh_mapper_config=Shard2dMeshMapper(dims=(0, 1)),
    )
