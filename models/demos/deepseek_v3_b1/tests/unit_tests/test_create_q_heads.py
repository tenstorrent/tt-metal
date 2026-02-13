# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.create_q_heads.op import CreateQHeads

"""
TTNN Create Q Heads Test
Tests create Q heads operation from 12x8 sender cores to 4x2 receiver cores.

Sender layout:
  - Qnope cores (cols 0-7): 8x8 = 64 cores, [1, 512] per core
  - Qrope cores (cols 8-11): 4x8 = 32 cores, [2, 64] per core (2 heads of 64 elements each)

Each receiver core at (rx, ry) collects from sender row:
  - ry=1: rows 0-3 (rx=0→row0, rx=1→row1, rx=2→row2, rx=3→row3)
  - ry=2: rows 4-7 (rx=0→row4, rx=1→row5, rx=2→row6, rx=3→row7)

Per-receiver data layout (8 heads per receiver):
  - Head 0: qnope[col=0] (512) + qrope[col=8, head0] (64) = 576 elements
  - Head 1: qnope[col=1] (512) + qrope[col=8, head1] (64) = 576 elements
  - Head 2: qnope[col=2] (512) + qrope[col=9, head0] (64) = 576 elements
  - Head 3: qnope[col=3] (512) + qrope[col=9, head1] (64) = 576 elements
  - Head 4: qnope[col=4] (512) + qrope[col=10, head0] (64) = 576 elements
  - Head 5: qnope[col=5] (512) + qrope[col=10, head1] (64) = 576 elements
  - Head 6: qnope[col=6] (512) + qrope[col=11, head0] (64) = 576 elements
  - Head 7: qnope[col=7] (512) + qrope[col=11, head1] (64) = 576 elements
  - Total: 8 * 576 = 4608 elements per receiver
"""


@pytest.mark.parametrize(
    "qnope_shard_shape, qrope_shard_shape, noc",
    [
        ((1, 512), (2, 64), ttnn.NOC.NOC_0),  # Force NOC0
        ((1, 512), (2, 64), ttnn.NOC.NOC_1),  # Force NOC1
        ((1, 512), (2, 64), None),  # Auto NOC routing - selects best single NOC
    ],
)
def test_create_q_heads(device, qnope_shard_shape, qrope_shard_shape, noc):
    """Test TTNN create Q heads operation from 12x8 cores to 4x2 cores"""
    # Qnope: 8 columns x 8 rows (cols 0-7)
    qnope_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))
    # Qrope: 4 columns x 8 rows (cols 8-11)
    qrope_grid = ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 7))
    # Receiver: 4 columns x 2 rows at (0-3, 1-2)
    receiver_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(3, 2))

    # Check device grid size
    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 12 or device_grid.y < 8:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 12x8 sender grid")

    # Calculate dimensions
    qnope_rows = qnope_grid.end.y - qnope_grid.start.y + 1  # 8
    qnope_cols = qnope_grid.end.x - qnope_grid.start.x + 1  # 8
    qrope_rows = qrope_grid.end.y - qrope_grid.start.y + 1  # 8
    qrope_cols = qrope_grid.end.x - qrope_grid.start.x + 1  # 4
    receiver_rows = receiver_grid.end.y - receiver_grid.start.y + 1  # 2
    receiver_cols = receiver_grid.end.x - receiver_grid.start.x + 1  # 4

    num_qnope_cores = qnope_rows * qnope_cols  # 64
    num_qrope_cores = qrope_rows * qrope_cols  # 32
    num_receivers = receiver_rows * receiver_cols  # 8

    # Shard sizes
    qnope_elements_per_core = qnope_shard_shape[0] * qnope_shard_shape[1]  # 512
    qrope_elements_per_head = qrope_shard_shape[1]  # 64

    # Head size = qnope (512) + qrope (64) = 576 elements
    head_elements = qnope_elements_per_core + qrope_elements_per_head  # 576
    num_heads_per_receiver = qnope_cols  # 8 heads per receiver (matches qnope_cols)

    # For BLOCK_SHARDED, tensor shape = (grid_rows * shard_height, grid_cols * shard_width)
    # Qnope tensor: 8 rows x 8 cols, shard (1, 512) → tensor shape (8, 4096)
    qnope_tensor_shape = (qnope_rows * qnope_shard_shape[0], qnope_cols * qnope_shard_shape[1])
    # Qrope tensor: 8 rows x 4 cols, shard (2, 64) → tensor shape (16, 256)
    qrope_tensor_shape = (qrope_rows * qrope_shard_shape[0], qrope_cols * qrope_shard_shape[1])
    # Output tensor: 2 rows x 4 cols, shard (8, 576) → tensor shape (16, 2304)
    output_shard_shape = (num_heads_per_receiver, head_elements)  # (8, 576)
    output_tensor_shape = (receiver_rows * output_shard_shape[0], receiver_cols * output_shard_shape[1])  # (16, 2304)

    logger.info(f"Qnope: {num_qnope_cores} cores, {qnope_shard_shape} per core, tensor shape {qnope_tensor_shape}")
    logger.info(f"Qrope: {num_qrope_cores} cores, {qrope_shard_shape} per core, tensor shape {qrope_tensor_shape}")
    logger.info(f"Output: {num_receivers} receivers, {output_shard_shape} per core, tensor shape {output_tensor_shape}")

    input_tile = ttnn.Tile([1, 32])
    output_tile = ttnn.Tile([8, 32])

    # Create PyTorch tensors with correct 2D shapes for BLOCK_SHARDED
    torch.manual_seed(42)
    torch_qnope = torch.randn(qnope_tensor_shape, dtype=torch.bfloat16)
    torch_qrope = torch.randn(qrope_tensor_shape, dtype=torch.bfloat16)

    # Compute expected output using golden function
    torch_expected = CreateQHeads.golden(torch_qnope, torch_qrope, qnope_grid, qrope_grid, receiver_grid)

    logger.info(f"Expected output shape: {torch_expected.shape}")
    logger.info(f"Expected output tensor shape for comparison: {output_tensor_shape}")

    # Create qnope tensor sharded on 8x8 grid
    qnope_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({qnope_grid}),
        qnope_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    qnope_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, qnope_shard_spec)

    ttnn_qnope = ttnn.from_torch(
        torch_qnope,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=qnope_mem_config,
        tile=input_tile,
    )

    # Create qrope tensor sharded on 4x8 grid
    qrope_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({qrope_grid}),
        qrope_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    qrope_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, qrope_shard_spec)

    ttnn_qrope = ttnn.from_torch(
        torch_qrope,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=qrope_mem_config,
        tile=input_tile,
    )

    # Create output tensor sharded on 4x2 grid
    # Each receiver has 8 heads of 576 elements = 4608 total elements
    # output_shard_shape already defined above as (8, 576)
    output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({receiver_grid}),
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_interm = torch.zeros(output_tensor_shape, dtype=torch.bfloat16)  # (16, 2304)
    torch_output = torch.zeros(output_tensor_shape, dtype=torch.bfloat16)  # (16, 2304)
    ttnn_interm = ttnn.from_torch(
        torch_interm,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    logger.info(f"Created tensors: qnope={ttnn_qnope.shape}, qrope={ttnn_qrope.shape}, output={ttnn_output.shape}")

    # Run create Q heads operation
    logger.info("Running create Q heads operation...")
    ttnn_result = CreateQHeads.op(ttnn_qnope, ttnn_qrope, ttnn_interm, ttnn_output)

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == output_tensor_shape, f"Expected shape {output_tensor_shape}, got {output_torch.shape}"

    # Verify that the output matches the expected
    logger.info("Verifying create Q heads results...")

    # Reshape expected to match output shape for comparison
    torch_expected_reshaped = torch_expected.reshape(output_tensor_shape)

    # Check if outputs match
    if torch.equal(output_torch, torch_expected_reshaped):
        logger.info("Create Q heads test passed!")
    else:
        # Print diff for debugging
        diff = (output_torch - torch_expected_reshaped).abs()
        max_diff = diff.max().item()
        num_mismatches = (diff > 0).sum().item()
        total_elements = output_torch.numel()
        logger.error(f"Output mismatch! Max diff: {max_diff}, Num mismatches: {num_mismatches}/{total_elements}")

        # Show first mismatch location for debugging
        mismatch_indices = torch.where(diff > 0)
        if len(mismatch_indices[0]) > 0:
            first_row = mismatch_indices[0][0].item()
            first_col = mismatch_indices[1][0].item()
            logger.error(
                f"First mismatch at ({first_row}, {first_col}): expected {torch_expected_reshaped[first_row, first_col]}, got {output_torch[first_row, first_col]}"
            )

        assert False, "Output tensor does not match expected tensor"
