# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for fused Input Gather + Gated Local Reduce + Down Projection.

Fuses:
  1. Input gather: pull [1,32] tiles from source cores to (12,9)
  2. Gated reduce: SiLU(sum(group1)) * sum(group2) → [1, K]
  3. Mcast: broadcast [1, K] to 130-core grid
  4. Matmul: [1, K] x [K, N_per_core] on 112 cores
  5. Output gather: collect to [1, N] on sender core

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_gated_local_reduce_down_proj.py -v -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj
from models.demos.deepseek_v3_b1.fused_ops.face_view_utils import can_use_face_view
from models.demos.deepseek_v3_b1.fused_ops.gated_local_reduce_down_proj.op import GatedLocalReduceDownProjOp
from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp


@pytest.mark.parametrize(
    "tiles_per_k, K, N_per_core, weights_dtype",
    [
        (2, 256, 32, ttnn.bfloat8_b),  # 2 collections of 8, K=256, N=3584
        (2, 128, 32, ttnn.bfloat8_b),  # 2 collections of 4, K=128
        (4, 256, 32, ttnn.bfloat8_b),  # 4 collections of 8, K=256
        (2, 256, 64, ttnn.bfloat8_b),  # 2 collections, larger output N=7168
        (2, 256, 64, ttnn.bfloat4_b),  # bfloat4 weights
        (8, 256, 64, ttnn.bfloat8_b),  # 8 collections, 64+64 A/B grid, N=7168
    ],
)
def test_gated_local_reduce_down_proj(device, tiles_per_k, K, N_per_core, weights_dtype):
    """Test fused input gather + gated local reduce + down projection."""

    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 13x10")

    N = N_per_core * DownProj.NUM_MATMUL_CORES  # 112 cores
    k_num_tiles = K // 32
    M = 1
    num_sources_per_group = tiles_per_k * k_num_tiles

    face_view = can_use_face_view(1, 32, tiles_per_k, k_num_tiles)

    # Use A/B grid when we need 64 sources per group (includes DRAM/phantom cores)
    use_ab_grid = num_sources_per_group == 64

    logger.info("=" * 70)
    logger.info("Testing Fused InputGather+GatedLocalReduce+DownProj:")
    logger.info(f"  tiles_per_k={tiles_per_k}, K={K}, N={N}, N_per_core={N_per_core}")
    logger.info(f"  k_num_tiles={k_num_tiles}, sources_per_group={num_sources_per_group}")
    logger.info(f"  weights_dtype={weights_dtype}, face_view={face_view}")
    logger.info(f"  use_ab_grid={use_ab_grid}")
    logger.info("=" * 70)

    # Tile definitions
    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    # ========================================================================
    # Select source cores
    # ========================================================================
    matmul_core_grid = DownProj.build_matmul_core_grid()
    matmul_cores_list = ttnn.corerange_to_cores(matmul_core_grid)

    if use_ab_grid:
        # 64+64 layout using DRAM workers, phantoms, and matmul cores
        g1_source_cores, g2_source_cores = SharedExpertOp.build_ab_grids()
        logger.info(f"Using A/B grid: {len(g1_source_cores)} A + {len(g2_source_cores)} B source cores")
    else:
        # Small cases: pick source cores from matmul cores only
        total_sources = 2 * num_sources_per_group
        assert total_sources <= len(
            matmul_cores_list
        ), f"Need {total_sources} source cores but only {len(matmul_cores_list)} matmul cores"

        g1_source_cores = matmul_cores_list[:num_sources_per_group]
        g2_source_cores = matmul_cores_list[num_sources_per_group:total_sources]

    g1_source_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in g1_source_cores])
    g2_source_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in g2_source_cores])

    # ========================================================================
    # Create source data: each source core gets one [1, 32] tile
    # Layout: tiles_per_k collections of k_num_tiles tiles each
    # Source tensor rows 0..k_num_tiles-1 → collection 0
    # Source tensor rows k_num_tiles..2*k_num_tiles-1 → collection 1, etc.
    # ========================================================================
    torch.manual_seed(42)

    # Generate collection vectors [tiles_per_k, 1, K] for golden
    g1_collections = [torch.randn((M, K), dtype=torch.bfloat16) for _ in range(tiles_per_k)]
    g2_collections = [torch.randn((M, K), dtype=torch.bfloat16) for _ in range(tiles_per_k)]

    # Build source tensors: [num_sources_per_group, 32]
    # Collection c, K-position k → source row c * k_num_tiles + k
    g1_src_rows = []
    for c in range(tiles_per_k):
        for k in range(k_num_tiles):
            g1_src_rows.append(g1_collections[c][:, k * 32 : (k + 1) * 32])
    torch_g1_src = torch.cat(g1_src_rows, dim=0)  # [num_sources, 32]

    g2_src_rows = []
    for c in range(tiles_per_k):
        for k in range(k_num_tiles):
            g2_src_rows.append(g2_collections[c][:, k * 32 : (k + 1) * 32])
    torch_g2_src = torch.cat(g2_src_rows, dim=0)  # [num_sources, 32]

    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)
    torch_add_input = torch.randn((M, N), dtype=torch.bfloat16)

    # Golden reference
    torch_expected = GatedLocalReduceDownProjOp.golden(
        [t.float() for t in g1_collections],
        [t.float() for t in g2_collections],
        torch_weights.float(),
        torch_add_input.float(),
    ).bfloat16()
    logger.info(f"Golden output shape: {torch_expected.shape}")

    # ========================================================================
    # Source tensors: HEIGHT_SHARDED, 1 tile per core
    # ========================================================================
    mcast_gather_core = DownProj.MCAST_GATHER_CORE
    sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_gather_core, mcast_gather_core)])

    g1_src_shard_spec = ttnn.ShardSpec(g1_source_grid, (M, 32), ttnn.ShardOrientation.ROW_MAJOR)
    g1_src_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, g1_src_shard_spec)

    ttnn_g1_src = ttnn.from_torch(
        torch_g1_src,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=g1_src_mem,
        tile=a_tile,
    )

    g2_src_shard_spec = ttnn.ShardSpec(g2_source_grid, (M, 32), ttnn.ShardOrientation.ROW_MAJOR)
    g2_src_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, g2_src_shard_spec)

    ttnn_g2_src = ttnn.from_torch(
        torch_g2_src,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=g2_src_mem,
        tile=a_tile,
    )
    logger.info(f"Created source tensors: {num_sources_per_group} cores per group")

    # ========================================================================
    # Gather destination tensors on sender core (pre-allocated, filled by gather)
    # ========================================================================
    dst_shard_shape = (num_sources_per_group * M, 32)
    dst_shard_spec = ttnn.ShardSpec(sender_core_grid, dst_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    dst_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, dst_shard_spec)

    ttnn_g1_dst = ttnn.from_torch(
        torch.zeros(num_sources_per_group * M, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dst_mem,
        tile=a_tile,
    )
    ttnn_g2_dst = ttnn.from_torch(
        torch.zeros(num_sources_per_group * M, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dst_mem,
        tile=a_tile,
    )
    logger.info(
        f"Created gather dest tensors: shard {dst_shard_shape} on ({mcast_gather_core.x},{mcast_gather_core.y})"
    )

    # ========================================================================
    # Weights: WIDTH_SHARDED across 112 matmul cores
    # ========================================================================
    weights_shard_spec = ttnn.ShardSpec(matmul_core_grid, (K, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    weights_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights_shard_spec)

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem,
        tile=b_tile,
    )
    logger.info(f"Created weights tensor: shard ({K}, {N_per_core}) on {DownProj.NUM_MATMUL_CORES} cores")

    # ========================================================================
    # Output: HEIGHT_SHARDED on sender core
    # ========================================================================
    output_shard_spec = ttnn.ShardSpec(sender_core_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem,
        tile=out_tile,
    )

    # ========================================================================
    # Add input (bias): HEIGHT_SHARDED on sender core
    # ========================================================================
    add_input_shard_spec = ttnn.ShardSpec(sender_core_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    add_input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, add_input_shard_spec)

    ttnn_add_input = ttnn.from_torch(
        torch_add_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=add_input_mem,
        tile=out_tile,
    )

    # ========================================================================
    # Run fused operation
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Running fused input gather + gated local reduce + down projection ...")

    ttnn_result = GatedLocalReduceDownProjOp.op(
        ttnn_g1_src,
        ttnn_g2_src,
        ttnn_g1_dst,
        ttnn_g2_dst,
        ttnn_weights,
        ttnn_add_input,
        ttnn_output,
        tiles_per_k,
        k_num_tiles,
    )

    output_torch = ttnn.to_torch(ttnn_result)
    logger.info(f"Output shape: {output_torch.shape}")

    expected_shape = (M, N)
    assert output_torch.shape == expected_shape, f"Expected {expected_shape}, got {output_torch.shape}"

    pcc_threshold = 0.97
    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(f"PCC comparison: {pcc_message}")

    assert passing, f"PCC check failed: {pcc_message}"
    logger.info("=" * 70)
    logger.info("Fused InputGather+GatedLocalReduce+DownProj test PASSED!")
    logger.info("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
