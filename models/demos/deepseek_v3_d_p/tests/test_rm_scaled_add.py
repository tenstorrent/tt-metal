# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from tracy import signpost
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize(
    "shape",
    [
        (1, 7168),  # Target use case (7 tiles)
        (1, 1024),  # Single tile
        (1, 15360),  # 15 tiles - tests chunked processing beyond 8 DSTs
    ],
)
@pytest.mark.parametrize(
    "scale",
    [
        0.5,
        1.0,
        2.0,
        -0.3,
    ],
)
def test_rm_scaled_add(device, shape, scale):
    """
    Test the rm_scaled_add operation: output = A + B * scale

    This experimental operation treats row-major tensors as tile-formatted data
    for efficient FPU processing.
    """
    torch.manual_seed(42)

    # Create input tensors
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    # Compute expected output
    expected = torch_a + torch_b * scale

    logger.info(f"Testing rm_scaled_add with shape={shape}, scale={scale}")
    logger.info(f"Input A range: [{torch_a.min():.4f}, {torch_a.max():.4f}]")
    logger.info(f"Input B range: [{torch_b.min():.4f}, {torch_b.max():.4f}]")
    logger.info(f"Expected range: [{expected.min():.4f}, {expected.max():.4f}]")

    # Convert to ttnn tensors
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Run the operation with signposts for Tracy profiling
    signpost(f"rm_scaled_add_{shape[0]}x{shape[1]}_scale{scale}_start")
    ttnn_output = ttnn.experimental.rm_scaled_add(ttnn_a, ttnn_b, scale)
    signpost(f"rm_scaled_add_{shape[0]}x{shape[1]}_scale{scale}_stop")

    # Convert back to torch
    output = ttnn.to_torch(ttnn_output)

    logger.info(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Compare with expected
    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    logger.info(f"PCC: {pcc}")

    assert passing, f"PCC {pcc} is below threshold 0.99"


@pytest.mark.parametrize(
    "scale",
    [
        0.5,
        1.0,
        2.0,
    ],
)
def test_scaled_add_with_ttnn_ops(device, scale):
    """
    Test A = A + B * scale using existing ttnn operations on (32, 224) tile tensor.

    This serves as a reference/comparison for the rm_scaled_add experimental op.
    Uses TILE_LAYOUT which is the standard format for ttnn operations.
    """
    torch.manual_seed(42)

    shape = (32, 224)  # 7168 elements = 7 tiles when viewed as 32x224

    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    expected = torch_a + torch_b * scale

    logger.info(f"Testing ttnn ops scaled add with shape={shape}, scale={scale}")

    # Use TILE_LAYOUT for standard ttnn operations
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # A = A + B * scale using standard ttnn ops with signposts for Tracy profiling
    signpost(f"ttnn_ops_scaled_add_32x224_scale{scale}_start")
    ttnn_scaled_b = ttnn.multiply(ttnn_b, scale)
    ttnn_output = ttnn.add(ttnn_a, ttnn_scaled_b)
    signpost(f"ttnn_ops_scaled_add_32x224_scale{scale}_stop")

    output = ttnn.to_torch(ttnn_output)

    passing, pcc = comp_pcc(expected, output, pcc=0.99)
    logger.info(f"PCC: {pcc}")

    assert passing, f"PCC {pcc} is below threshold 0.99"
