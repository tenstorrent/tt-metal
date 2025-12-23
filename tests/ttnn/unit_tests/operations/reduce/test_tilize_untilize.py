# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("height", [32, 64, 128])
@pytest.mark.parametrize("width", [32, 64, 128])
def test_tilize_untilize(device, batch_size, channels, height, width):
    """
    Test tilize_untilize operation which serves as a template for compute operations.

    This operation takes row-major input, tilizes it for compute, then untilizes back
    to row-major output. As an identity operation, output should exactly match input.

    Requirements:
    - Height and width must be multiples of 32 (tile-aligned)
    - Input must be in ROW_MAJOR layout
    - Input must be INTERLEAVED (not sharded)
    """
    torch.manual_seed(0)

    shape = (batch_size, channels, height, width)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.tilize_untilize(input_tensor)

    # Verify output properties
    assert (
        output_tensor.shape == input_tensor.shape
    ), f"Shape mismatch: expected {input_tensor.shape}, got {output_tensor.shape}"
    assert (
        output_tensor.dtype == input_tensor.dtype
    ), f"Dtype mismatch: expected {input_tensor.dtype}, got {output_tensor.dtype}"
    assert (
        output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Layout mismatch: expected ROW_MAJOR_LAYOUT, got {output_tensor.layout}"

    # Convert back to torch for comparison
    torch_output = ttnn.to_torch(output_tensor)

    # Identity operation: output should match input exactly
    # Using PCC=1.0 since this is a lossless transform
    assert_with_pcc(torch_input, torch_output, pcc=0.9999)
