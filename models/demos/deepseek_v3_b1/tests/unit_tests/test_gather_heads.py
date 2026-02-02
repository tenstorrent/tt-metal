# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Gather Heads Test
Tests gather heads operation from 6x4 sender cores to 2x2 receiver cores.

Sender layout:
  - Qnope cores (cols 0-3): 4x4 = 16 cores, [1, 512] per core
  - Qrope cores (cols 4-5): 2x4 = 8 cores, [2, 64] per core (2 heads of 64 elements each)

Each receiver core at (rx, ry) collects from sender row:
  - ry=1: rows 0-1 (rx=0→row0, rx=1→row1)
  - ry=2: rows 2-3 (rx=0→row2, rx=1→row3)

Per-receiver data layout (4 heads per receiver):
  - Head 0: qnope[col=0] (512) + qrope[col=4, head0] (64) = 576 elements
  - Head 1: qnope[col=1] (512) + qrope[col=4, head1] (64) = 576 elements
  - Head 2: qnope[col=2] (512) + qrope[col=5, head0] (64) = 576 elements
  - Head 3: qnope[col=3] (512) + qrope[col=5, head1] (64) = 576 elements
  - Total: 4 * 576 = 2304 elements per receiver
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.gather_heads.op import GatherHeads


def golden_gather_heads(qnope_input, qrope_input, qnope_grid, qrope_grid, receiver_grid):
    """
    PyTorch reference implementation of gather heads.

    Args:
        qnope_input: (4, 2048) tensor - BLOCK_SHARDED across 4x4 grid with shard (1, 512)
        qrope_input: (8, 128) tensor - BLOCK_SHARDED across 4x2 grid with shard (2, 64)
        qnope_grid: 4x4 grid
        qrope_grid: 2x4 grid
        receiver_grid: 2x2 grid

    Returns:
        Output tensor with gathered data for each receiver (4 heads each of 576 elements)
    """
    # Extract dimensions
    qnope_rows = qnope_grid.end.y - qnope_grid.start.y + 1  # 4
    qnope_cols = qnope_grid.end.x - qnope_grid.start.x + 1  # 4
    qrope_rows = qrope_grid.end.y - qrope_grid.start.y + 1  # 4
    qrope_cols = qrope_grid.end.x - qrope_grid.start.x + 1  # 2

    receiver_rows = receiver_grid.end.y - receiver_grid.start.y + 1  # 2
    receiver_cols = receiver_grid.end.x - receiver_grid.start.x + 1  # 2

    # Per-core shard sizes (from tensor shapes and grid)
    qnope_shard_h = qnope_input.shape[0] // qnope_rows  # 1
    qnope_shard_w = qnope_input.shape[1] // qnope_cols  # 512
    qrope_shard_h = qrope_input.shape[0] // qrope_rows  # 2
    qrope_shard_w = qrope_input.shape[1] // qrope_cols  # 64

    qnope_elements = qnope_shard_h * qnope_shard_w  # 512
    qrope_elements_per_head = qrope_shard_w  # 64

    # Per-head size = qnope (512) + qrope (64) = 576
    head_elements = qnope_elements + qrope_elements_per_head  # 576

    # Per-receiver: 4 heads (qnope_cols)
    num_heads_per_receiver = qnope_cols

    # For BLOCK_SHARDED with shard (shard_h, shard_w) on grid (rows, cols):
    # Tensor is (rows * shard_h, cols * shard_w)
    # Shard at grid position (r, c) covers tensor region:
    #   rows: [r * shard_h, (r+1) * shard_h)
    #   cols: [c * shard_w, (c+1) * shard_w)

    # Create output tensor in BLOCK_SHARDED layout
    # For BLOCK_SHARDED with shard (4, 576) on 2x2 grid:
    # - Tensor shape: (2*4, 2*576) = (8, 1152)
    # - Each receiver's shard is 4 rows x 576 cols
    # - Head N is at row N within the shard (row N, cols 0-575)
    output = torch.zeros(receiver_rows * num_heads_per_receiver, receiver_cols * head_elements, dtype=qnope_input.dtype)

    # Gather: each receiver gets data from one sender row
    # Receiver grid is ROW_MAJOR: (0,1), (1,1), (0,2), (1,2)
    # These map to sender rows 0, 1, 2, 3
    for ry_idx in range(receiver_rows):
        for rx_idx in range(receiver_cols):
            # Receiver at grid position (rx_idx, ry_idx)
            # Sender row mapping: (ry_idx == 0) ? rx_idx : (rx_idx + 2)
            sender_row = rx_idx if ry_idx == 0 else (rx_idx + 2)

            # Build 4 heads for this receiver
            for head_idx in range(num_heads_per_receiver):
                qnope_col = head_idx  # 0-3
                qrope_col = head_idx // 2  # 0-1 (each qrope col serves 2 heads)
                qrope_head = head_idx % 2  # 0 or 1

                # Get qnope data for this head from shard at (sender_row, qnope_col)
                qnope_row_start = sender_row * qnope_shard_h
                qnope_row_end = qnope_row_start + qnope_shard_h
                qnope_col_start = qnope_col * qnope_shard_w
                qnope_col_end = qnope_col_start + qnope_shard_w
                qnope_data = qnope_input[qnope_row_start:qnope_row_end, qnope_col_start:qnope_col_end].flatten()

                # Get qrope data for this head from shard at (sender_row, qrope_col)
                # qrope shard is (2, 64), we need head qrope_head (row 0 or 1)
                qrope_row_start = sender_row * qrope_shard_h + qrope_head
                qrope_col_start = qrope_col * qrope_shard_w
                qrope_col_end = qrope_col_start + qrope_shard_w
                qrope_data = qrope_input[qrope_row_start, qrope_col_start:qrope_col_end]

                # Place in output using BLOCK_SHARDED layout:
                # - Receiver at (rx_idx, ry_idx) owns rows [ry_idx*8, (ry_idx+1)*8) and cols [rx_idx*576, (rx_idx+1)*576)
                # - Head N is at row (ry_idx*8 + N), cols [rx_idx*576, rx_idx*576 + 576)
                out_row = ry_idx * num_heads_per_receiver + head_idx
                out_col_start = rx_idx * head_elements
                output[out_row, out_col_start : out_col_start + qnope_elements] = qnope_data
                output[out_row, out_col_start + qnope_elements : out_col_start + head_elements] = qrope_data

    return output.reshape(1, -1)  # Flatten to [1, total_elements]


@pytest.mark.parametrize(
    "qnope_shard_shape, qrope_shard_shape, noc",
    [
        ((1, 512), (2, 64), ttnn.NOC.NOC_0),  # Force NOC0
        ((1, 512), (2, 64), ttnn.NOC.NOC_1),  # Force NOC1
        ((1, 512), (2, 64), None),  # Auto NOC routing - selects best single NOC
    ],
)
def test_gather_heads(device, qnope_shard_shape, qrope_shard_shape, noc):
    """Test TTNN gather heads operation from 6x4 cores to 2x2 cores"""
    # Define grids
    # Qnope: 4 columns x 4 rows (cols 0-3)
    qnope_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))
    # Qrope: 2 columns x 4 rows (cols 4-5)
    qrope_grid = ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 3))
    # Receiver: 2 columns x 2 rows at (0-1, 1-2)
    receiver_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 2))

    # Check device grid size
    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 6 or device_grid.y < 4:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 6x4 sender grid")

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
    num_heads_per_receiver = qnope_cols  # Number of heads per receiver equals number of qnope columns

    # For BLOCK_SHARDED, tensor shape = (grid_rows * shard_height, grid_cols * shard_width)
    # Qnope tensor: 4 rows x 4 cols, shard (1, 512) → tensor shape (4, 2048)
    qnope_tensor_shape = (qnope_rows * qnope_shard_shape[0], qnope_cols * qnope_shard_shape[1])
    # Qrope tensor: 4 rows x 2 cols, shard (2, 64) → tensor shape (8, 128)
    qrope_tensor_shape = (qrope_rows * qrope_shard_shape[0], qrope_cols * qrope_shard_shape[1])
    # Output tensor: 2 rows x 2 cols, shard (4, 576) → tensor shape (8, 1152)
    output_shard_shape = (num_heads_per_receiver, head_elements)  # (4, 576)
    output_tensor_shape = (receiver_rows * output_shard_shape[0], receiver_cols * output_shard_shape[1])

    logger.info(f"Qnope: {num_qnope_cores} cores, {qnope_shard_shape} per core, tensor shape {qnope_tensor_shape}")
    logger.info(f"Qrope: {num_qrope_cores} cores, {qrope_shard_shape} per core, tensor shape {qrope_tensor_shape}")
    logger.info(f"Output: {num_receivers} receivers, {output_shard_shape} per core, tensor shape {output_tensor_shape}")

    # Use tile with 1x32 shape for this test
    tile = ttnn.Tile([1, 32])

    # Create PyTorch tensors with correct 2D shapes for BLOCK_SHARDED
    torch.manual_seed(42)
    torch_qnope = torch.randn(qnope_tensor_shape, dtype=torch.bfloat16)
    torch_qrope = torch.randn(qrope_tensor_shape, dtype=torch.bfloat16)

    # Compute expected output using golden function
    torch_expected = golden_gather_heads(torch_qnope, torch_qrope, qnope_grid, qrope_grid, receiver_grid)

    logger.info(f"Expected output shape: {torch_expected.shape}")
    logger.info(f"Expected output tensor shape for comparison: {output_tensor_shape}")

    # Create qnope tensor sharded on 4x4 grid
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
        tile=tile,
    )

    # Create qrope tensor sharded on 2x4 grid
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
        tile=tile,
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

    torch_output = torch.zeros(output_tensor_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )

    logger.info(f"Created tensors: qnope={ttnn_qnope.shape}, qrope={ttnn_qrope.shape}, output={ttnn_output.shape}")
    logger.info(f"NOC mode: {noc if noc else 'Auto'}")

    # Run gather heads operation
    logger.info("Running gather heads operation...")
    ttnn_result = GatherHeads.op(ttnn_qnope, ttnn_qrope, ttnn_output, noc)

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == output_tensor_shape, f"Expected shape {output_tensor_shape}, got {output_torch.shape}"

    # Verify that the output matches the expected
    logger.info("Verifying gather heads results...")

    # Reshape expected to match output shape for comparison
    torch_expected_reshaped = torch_expected.reshape(output_tensor_shape)

    # Check if outputs match
    if torch.equal(output_torch, torch_expected_reshaped):
        logger.info("Gather heads test passed!")
    else:
        # Print diff for debugging
        diff = (output_torch - torch_expected_reshaped).abs()
        max_diff = diff.max().item()
        num_mismatches = (diff > 0).sum().item()
        total_elements = output_torch.numel()
        logger.error(f"Output mismatch! Max diff: {max_diff}, Num mismatches: {num_mismatches}/{total_elements}")

        # Check first few elements of each head for receiver 0
        logger.info("=== Receiver 0 head data comparison ===")
        for head in range(8):
            head_start = head * 576
            qnope_slice = slice(head_start, head_start + 4)  # First 4 qnope elements
            qrope_slice = slice(head_start + 512, head_start + 516)  # First 4 qrope elements
            logger.info(
                f"Head {head} qnope[0:4]: expected={torch_expected_reshaped[0, qnope_slice].tolist()}, got={output_torch[0, qnope_slice].tolist()}"
            )
            logger.info(
                f"Head {head} qrope[0:4]: expected={torch_expected_reshaped[0, qrope_slice].tolist()}, got={output_torch[0, qrope_slice].tolist()}"
            )

        # Find first mismatch location
        mismatch_indices = torch.where(diff > 0)
        if len(mismatch_indices[0]) > 0:
            first_row = mismatch_indices[0][0].item()
            first_col = mismatch_indices[1][0].item()
            logger.error(
                f"First mismatch at ({first_row}, {first_col}): expected {torch_expected_reshaped[first_row, first_col]}, got {output_torch[first_row, first_col]}"
            )

        assert False, "Output tensor does not match expected tensor"
