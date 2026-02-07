# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for KNSlicedMatmul micro-op.

Verifies: act[k_offset..k_offset+k_per_core] @ weights[k_per_core, out_w] -> output[1, out_w]

Tests both single-core (out_w > 1) and multi-core (k_offset > 0) configurations.

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_kn_sliced_matmul.py -v -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.kn_sliced_matmul.op import KNSlicedMatmulOp


@pytest.mark.parametrize(
    "k_per_core, out_w",
    [
        (4, 1),  # Original: single output tile
        (4, 2),  # Two output tiles
        (8, 2),  # Larger K, two output tiles
        (4, 4),  # Four output tiles
    ],
)
def test_kn_sliced_matmul_single_core(device, k_per_core, out_w):
    """Test KNSlicedMatmul on a single core with configurable out_w."""

    M = 1
    act_tile = ttnn.Tile([M, 32])
    weight_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    K = k_per_core * 32
    N = out_w * 32

    logger.info(f"Testing KNSlicedMatmul: [{M}, {K}] x [{K}, {N}] -> [{M}, {N}], out_w={out_w}")

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})

    torch.manual_seed(42)
    torch_act = torch.randn((M, K), dtype=torch.bfloat16)
    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)
    torch_expected = (torch_act.float() @ torch_weights.float()).bfloat16()

    act_shard = ttnn.ShardSpec(core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
    act_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, act_shard)
    ttnn_act = ttnn.from_torch(
        torch_act, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=act_mem, tile=act_tile
    )

    weights_shard = ttnn.ShardSpec(core_grid, (K, N), ttnn.ShardOrientation.ROW_MAJOR)
    weights_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights_shard)
    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem,
        tile=weight_tile,
    )

    out_shard = ttnn.ShardSpec(core_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem,
        tile=out_tile,
    )

    KNSlicedMatmulOp.op(ttnn_act, ttnn_weights, ttnn_output, k_per_core, out_w)

    output_torch = ttnn.to_torch(ttnn_output)
    assert output_torch.shape == (M, N)

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.99)
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"PCC check failed: {pcc_message}"


@pytest.mark.parametrize(
    "num_cores, k_per_core, out_w",
    [
        (2, 4, 1),  # 2 cores, K-parallel, out_w=1
        (4, 2, 1),  # 4 cores, K-parallel, out_w=1
        (2, 4, 2),  # 2 cores, K-parallel, out_w=2
    ],
)
def test_kn_sliced_matmul_k_offset(device, num_cores, k_per_core, out_w):
    """Test KNSlicedMatmul with k_offset > 0 across multiple cores.

    Each core computes a K-slice partial: act[k_offset..k_offset+k_per_core] @ weights.
    Verifies each core's output independently against the golden partial.
    """

    M = 1
    act_tile = ttnn.Tile([M, 32])
    weight_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    K_total = num_cores * k_per_core * 32
    N = out_w * 32

    logger.info(
        f"Testing KNSlicedMatmul k_offset: {num_cores} cores, "
        f"K_total={K_total}, k_per_core={k_per_core * 32}, out_w={out_w}"
    )

    cores = [ttnn.CoreCoord(i, 0) for i in range(num_cores)]
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    torch.manual_seed(42)
    torch_act = torch.randn((M, K_total), dtype=torch.bfloat16)
    torch_weights_per_core = [torch.randn((k_per_core * 32, N), dtype=torch.bfloat16) for _ in range(num_cores)]

    # Golden: each core's partial result
    torch_partials = []
    for i in range(num_cores):
        k_start = i * k_per_core * 32
        k_end = k_start + k_per_core * 32
        partial = (torch_act[:, k_start:k_end].float() @ torch_weights_per_core[i].float()).bfloat16()
        torch_partials.append(partial)

    # Activation: replicated on each core via HEIGHT_SHARDED with full [1, K_total] shard.
    # Each core uses k_offset to index into its slice of the shared activation buffer.
    act_shard = ttnn.ShardSpec(core_grid, (M, K_total), ttnn.ShardOrientation.ROW_MAJOR)
    act_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, act_shard)
    ttnn_act = ttnn.from_torch(
        torch_act.repeat(num_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=act_mem,
        tile=act_tile,
    )

    # Weights: HEIGHT_SHARDED, each core gets [k_per_core * 32, N]
    torch_weights_stacked = torch.cat(torch_weights_per_core, dim=0)
    weights_shard = ttnn.ShardSpec(core_grid, (k_per_core * 32, N), ttnn.ShardOrientation.ROW_MAJOR)
    weights_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, weights_shard)
    ttnn_weights = ttnn.from_torch(
        torch_weights_stacked,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem,
        tile=weight_tile,
    )

    # Output: HEIGHT_SHARDED, each core produces [M, N]
    out_shard = ttnn.ShardSpec(core_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard)
    ttnn_output = ttnn.from_torch(
        torch.zeros((num_cores * M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem,
        tile=out_tile,
    )

    core_k_offsets = [(cores[i], i * k_per_core) for i in range(num_cores)]
    KNSlicedMatmulOp.op(ttnn_act, ttnn_weights, ttnn_output, k_per_core, out_w, core_k_offsets)

    output_torch = ttnn.to_torch(ttnn_output)
    assert output_torch.shape == (num_cores * M, N)

    for i in range(num_cores):
        core_output = output_torch[i * M : (i + 1) * M, :]
        passing, pcc_message = comp_pcc(torch_partials[i], core_output, 0.99)
        logger.info(f"Core {i} (k_offset={i * k_per_core}): PCC={pcc_message}")
        assert passing, f"Core {i} PCC check failed: {pcc_message}"

    logger.info(f"PASSED: {num_cores} cores, k_per_core={k_per_core}, out_w={out_w}")
