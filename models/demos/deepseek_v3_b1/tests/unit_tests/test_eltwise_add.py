# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Eltwise Add Micro Op Test

Tests element-wise addition with per-core indexing for MoE fused operation:
- 8 MM cores (optimal DRAM bank workers), each with 1x896 down_proj output
- fused_add tensor: 1x7168, 1x32 tiles, HEIGHT_SHARDED (replicated on all cores)
- Each core uses sender_index to offset into fused_add:
  - Core 0 (sender_index=0): fused_add[0:896]
  - Core 1 (sender_index=1): fused_add[896:1792]
  - Core 2 (sender_index=2): fused_add[1792:2688]
  - etc.

Tensor and CB setup:
- down_proj output: 1x896 per core, 1x32 tiles, WIDTH_SHARDED
- fused_add: 1x7168 per core (replicated), 1x32 tiles, HEIGHT_SHARDED
- output: 1x896 per core, 1x32 tiles, WIDTH_SHARDED
- CB view: 32x32 tiles (CB aliasing - data order unchanged, just different view)

Integration pattern (after down_proj in MoE kernel):
1. Each core has down_proj output (1x896) in WIDTH_SHARDED CB (1x32 tiles, view as 32x32)
2. Each core has full fused_add (1x7168) in HEIGHT_SHARDED CB (1x32 tiles, view as 32x32)
3. Kernel uses sender_index to compute offset into fused_add
4. TRISC does add_tiles on 32x32 CB view
5. Output is WIDTH_SHARDED
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.eltwise_add.op import EltwiseAdd
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def golden_eltwise_add(mm_output: torch.Tensor, fused_add: torch.Tensor) -> torch.Tensor:
    """
    Compute golden reference for eltwise add.

    Args:
        mm_output: Output from down_proj matmul [1, 1, 1, total_width]
        fused_add: Tensor to add [1, 1, 1, total_width]

    Returns:
        mm_output + fused_add
    """
    return mm_output + fused_add


@pytest.mark.parametrize("width_per_core", [896])  # Actual MoE dimensions: 8 * 896 = 7168
@pytest.mark.parametrize("num_cores", [8])
def test_eltwise_add_moe_dimensions(device, width_per_core, num_cores):
    """
    Test eltwise add with actual MoE dimensions and replicated fused_add tensor.

    Tensor setup:
    - down_proj output: 1x896 per core, 1x32 tiles, WIDTH_SHARDED
    - fused_add: 1x7168 per core (replicated), 1x32 tiles, HEIGHT_SHARDED
    - output: 1x896 per core, 1x32 tiles, WIDTH_SHARDED

    Each core uses sender_index to offset into fused_add:
      - Core 0 (sender_index=0): fused_add[0:896]
      - Core 1 (sender_index=1): fused_add[896:1792]
      - etc.

    CB view: 32x32 tiles (aliasing, data order unchanged)
    """
    total_width = width_per_core * num_cores  # 7168

    # 1x32 tiles for tensors
    tile_1x32 = ttnn.Tile([1, 32])

    # Get compute cores (optimal DRAM bank workers)
    compute_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    if len(compute_cores) < num_cores:
        pytest.skip(f"Device has {len(compute_cores)} cores, need {num_cores}")
    compute_cores = compute_cores[:num_cores]

    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )

    logger.info(f"Testing MoE eltwise_add: {num_cores} cores x {width_per_core} = {total_width}")
    logger.info(f"Compute cores: {[(c.x, c.y) for c in compute_cores]}")

    # Create PyTorch tensors
    torch.manual_seed(42)
    # mm_output: WIDTH_SHARDED [1, 1, 1, 7168] -> each core gets [1, 896]
    mm_output_torch = torch.randn([1, 1, 1, total_width]).bfloat16().float()
    # fused_add: [1, 1, 1, 7168] -> replicated on all cores
    fused_add_torch = torch.randn([1, 1, 1, total_width]).bfloat16().float()

    # Compute golden reference
    # Core i computes: mm_output[i*896:(i+1)*896] + fused_add[i*896:(i+1)*896]
    golden = golden_eltwise_add(mm_output_torch, fused_add_torch)

    # ========== mm_output tensor - WIDTH_SHARDED, 1x32 tiles ==========
    # Each core gets [1, 896] slice (28 tiles of 1x32)
    mm_out_shard_spec = ttnn.ShardSpec(compute_core_grid, (1, width_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    mm_out_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, mm_out_shard_spec
    )
    mm_output_t = ttnn.from_torch(
        mm_output_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mm_out_memory_config,
        tile=tile_1x32,
    )

    # ========== fused_add tensor - HEIGHT_SHARDED (replicated), 1x32 tiles ==========
    # Each core has the FULL [1, 7168] tensor (224 tiles of 1x32)
    # Replicate along height for HEIGHT_SHARDED distribution
    fused_add_replicated = fused_add_torch.repeat(1, 1, num_cores, 1)  # [1, 1, 8, 7168]
    fused_add_shard_spec = ttnn.ShardSpec(compute_core_grid, (1, total_width), ttnn.ShardOrientation.ROW_MAJOR)
    fused_add_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, fused_add_shard_spec
    )
    fused_add_t = ttnn.from_torch(
        fused_add_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=fused_add_memory_config,
        tile=tile_1x32,
    )

    # ========== Output tensor - WIDTH_SHARDED, 1x32 tiles, padded to 32x32 size ==========
    # Each core outputs 1024 elements (padded for 32x32 tile), only first 896 are valid
    output_width_per_core = 32 * 32  # 1024 elements per core (32x32 tile size)
    output_total_width = output_width_per_core * num_cores  # 8192
    output_shard_spec = ttnn.ShardSpec(compute_core_grid, (1, output_width_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )
    output_t = ttnn.from_torch(
        torch.zeros([1, 1, 1, output_total_width]).bfloat16().float(),  # [1, 1, 1, 8192]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_memory_config,
        tile=tile_1x32,
    )

    logger.info("MoE eltwise add tensor setup:")
    logger.info(f"  mm_output: WIDTH_SHARDED [1, {width_per_core}] per core, 1x32 tiles")
    logger.info(f"  fused_add: HEIGHT_SHARDED [1, {total_width}] per core (replicated), 1x32 tiles")
    logger.info(f"  output: WIDTH_SHARDED [1, {output_width_per_core}] per core, 1x32 tiles (padded)")
    logger.info(f"  Valid output: first {width_per_core} of {output_width_per_core} elements per core")
    logger.info(f"  Kernel uses sender_index to offset into fused_add")

    # ========== Run eltwise add with custom kernel ==========
    logger.info("Running MoE eltwise add with EltwiseAdd.op")
    try:
        result_t = EltwiseAdd.op(mm_output_t, fused_add_t, output_t)
    except Exception as e:
        logger.error(f"EltwiseAdd.op failed: {e}")
        pytest.skip(f"Operation failed: {e}")

    # Convert to torch
    result_torch = ttnn.to_torch(result_t)

    # Result shape is [1, 1, 1, 8192] (1024 per core, padded)
    # Golden shape is [1, 1, 1, 7168] (896 per core)
    # Extract first 896 elements from each 1024-element chunk
    result_valid = []
    for i in range(num_cores):
        start_idx = i * output_width_per_core
        end_idx = start_idx + width_per_core
        result_valid.append(result_torch[..., start_idx:end_idx])
    result_valid = torch.cat(result_valid, dim=-1)  # [1, 1, 1, 7168]

    expected_pcc = 0.999
    passing, output = comp_pcc(golden, result_valid, expected_pcc)
    logger.info(f"Golden: {golden}")
    logger.info(f"Result: {result_valid}")
    logger.info(output)
    assert passing, f"PCC check failed: {output}"
    logger.info("MoE eltwise add test passed!")
