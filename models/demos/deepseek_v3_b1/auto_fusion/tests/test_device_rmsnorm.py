# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device test: auto-fused single RMSNorm kernel.

Validates that an auto-generated fused kernel containing a single RMSNorm
produces identical output to the standalone RMSNorm micro-op.

Requires Blackhole device (RMSNorm uses experimental mul_reduce_scalar API).

Run with: python -m pytest models/demos/deepseek_v3_b1/auto_fusion/tests/test_device_rmsnorm.py -xvs
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph
from models.demos.deepseek_v3_b1.auto_fusion.specs.rmsnorm import RMSNORM
from models.demos.deepseek_v3_b1.micro_ops.rmsnorm.op import RMSNormSingleCore
from models.demos.deepseek_v3_b1.utils import float_to_uint32


def _is_blackhole(device) -> bool:
    """Check if the device is a Blackhole architecture."""
    try:
        arch = device.arch()
        return "blackhole" in str(arch).lower()
    except Exception:
        return False


@pytest.mark.parametrize("width", [7168, 1536, 512])
@pytest.mark.parametrize("epsilon", [1e-6])
def test_auto_fused_rmsnorm_matches_standalone(device, width, epsilon):
    """
    Test that auto-fused RMSNorm produces the same output as standalone RMSNorm.

    Strategy:
    1. Run the standalone RMSNorm micro-op and record the output (Blackhole only)
    2. Build and run an auto-fused kernel with the same RMSNorm
    3. Compare outputs against golden (and standalone on Blackhole)

    Note: RMSNorm uses experimental mul_reduce_scalar API which is only
    available on Blackhole. This test is skipped on other architectures.
    """
    if not _is_blackhole(device):
        pytest.skip("RMSNorm uses experimental mul_reduce_scalar API (Blackhole only)")

    shape = (1, width)
    tile = ttnn.Tile([1, 32])
    FULL_32x32_TILE = ttnn.Tile((32, 32))
    HALF_16x32_TILE = ttnn.Tile((16, 32))
    is_16x32 = (width // 32) % 32 != 0
    interpreted_tile = HALF_16x32_TILE if is_16x32 else FULL_32x32_TILE
    tile_height, tile_width = interpreted_tile.tile_shape
    num_tiles = (shape[0] * shape[1]) // (tile_height * tile_width)
    numel = shape[0] * shape[1]

    # Create torch tensors
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape, dtype=torch.bfloat16)

    # Golden reference
    torch_expected = RMSNormSingleCore.golden(torch_input, torch_gamma, epsilon=epsilon)

    # Shard spec: single core
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        (shape[0], width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    # === Run standalone RMSNorm for reference ===
    ttnn_input_ref = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config, tile=tile
    )
    ttnn_gamma_ref = ttnn.from_torch(
        torch_gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config, tile=tile
    )
    ttnn_output_ref = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    standalone_result = RMSNormSingleCore.op(
        ttnn_input_ref,
        ttnn_gamma_ref,
        ttnn_output_ref,
        epsilon=epsilon,
        numel=numel,
        fp32_dest_acc_en=False,
    )
    standalone_torch = ttnn.to_torch(standalone_result)[:, :width]

    # === Build and run auto-fused RMSNorm ===
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    # Create fresh tensors for fused run
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config, tile=tile
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config, tile=tile
    )
    ttnn_output = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    # Build the fusion graph
    g = FusionGraph()
    g.add(
        "rmsnorm",
        RMSNORM,
        cores=core_grid,
        ct_args={
            "fp32_acc": 0,
            "num_tiles": num_tiles,
            "rsqrt_fast_approx": 0,
            "input_num_pages": num_tiles,
            "gamma_num_pages": num_tiles,
            "epsilon": float_to_uint32(epsilon),
            "scalar": float_to_uint32(1.0 / math.sqrt(float(numel))),
        },
    )

    # Build and run
    fused_op = g.build(
        device,
        io_tensors={
            ("rmsnorm", "input"): ttnn_input,
            ("rmsnorm", "gamma"): ttnn_gamma,
            ("rmsnorm", "output"): ttnn_output,
        },
    )

    logger.info(f"Generated kernel path: {fused_op.kernel_path}")
    fused_result = fused_op.run()
    fused_torch = ttnn.to_torch(fused_result)[:, :width]

    # === Validate against golden ===
    passing_golden, pcc_golden = comp_pcc(torch_expected, fused_torch, 0.999)
    logger.info(f"Fused vs golden: {pcc_golden}")

    # === Validate against standalone ===
    passing_standalone, pcc_standalone = comp_pcc(standalone_torch, fused_torch, 0.999)
    logger.info(f"Fused vs standalone: {pcc_standalone}")

    assert passing_golden, f"Fused RMSNorm doesn't match golden: {pcc_golden}"
    assert passing_standalone, f"Fused RMSNorm doesn't match standalone: {pcc_standalone}"
    logger.info(f"PASSED: width={width}")
