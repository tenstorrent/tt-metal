# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_equal


def test_untilize_with_padded_input(device):
    """Demonstrate that untilize on a padded tensor produces a buffer with padding.

    The corruption is 'silent' because to_torch() and print() respect logical_shape
    and only show the logical portion. But the underlying buffer is larger than expected,
    containing zero-padded rows/columns that downstream ops may misinterpret.
    """
    # Use randn so values are distinct and misalignment would be obvious
    torch_tensor = torch.randn(1, 1, 33, 33, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    print(f"\nInput logical_shape: {tt_tensor.shape}")
    print(f"Input padded_shape:  {tt_tensor.padded_shape}")

    # Without fix: untilize produces ROW_MAJOR tensor with padding in the buffer
    untilized = ttnn.untilize(tt_tensor)

    print(f"\nAfter untilize:")
    print(f"  logical_shape: {untilized.shape}")
    print(f"  padded_shape:  {untilized.padded_shape}")
    print(f"  layout:        {untilized.layout}")

    # Check if the fix is active: if padded_shape == logical_shape, our fix worked
    has_padding = untilized.shape != untilized.padded_shape
    print(f"  output has padding in buffer: {has_padding}")

    if has_padding:
        # The buffer is LARGER than logical shape implies.
        # to_torch() hides this by extracting only the logical portion.
        torch_output = ttnn.to_torch(untilized)
        print(f"  to_torch() shape: {torch_output.shape}")  # Shows [1,1,33,33] — looks fine!
        print(
            f"  BUT the actual buffer has {untilized.padded_shape} = "
            f"{untilized.padded_shape[-2] * untilized.padded_shape[-1]} elements per batch"
        )
        print(f"  while logical data is only " f"{untilized.shape[-2] * untilized.shape[-1]} elements per batch")
        print("\n  silent corruption: the buffer has extra padding data")
        print("  that downstream ops may misinterpret, causing PCC failures.")
    else:
        # Fix is active: output has no padding, buffer is exactly 33×33
        torch_output = ttnn.to_torch(untilized)
        print(f"  to_torch() shape: {torch_output.shape}")
        print("   No padding in output buffer (clean data)")

    # Verify correctness
    assert_equal(torch_tensor, ttnn.to_torch(untilized))
