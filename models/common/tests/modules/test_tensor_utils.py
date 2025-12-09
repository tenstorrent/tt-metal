# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for tensor utility functions in models.common.modules.tensor_utils.
"""

import pytest
import torch


def test_pad_dim_to_size():
    """Test the pad_dim_to_size utility function."""
    from models.common.modules.tensor_utils import pad_dim_to_size

    # Test padding on last dimension
    x = torch.randn(1, 1, 32, 100)
    padded = pad_dim_to_size(x, dim=-1, size=128)
    assert padded.shape == (1, 1, 32, 128)

    # Original data should be preserved
    assert torch.allclose(padded[:, :, :, :100], x)

    # Padding should be zeros
    assert torch.allclose(padded[:, :, :, 100:], torch.zeros(1, 1, 32, 28))

    # Test no padding needed
    x2 = torch.randn(1, 1, 32, 128)
    padded2 = pad_dim_to_size(x2, dim=-1, size=128)
    assert torch.equal(padded2, x2)

    # Test padding on different dimension
    x3 = torch.randn(1, 1, 24, 128)
    padded3 = pad_dim_to_size(x3, dim=-2, size=32)
    assert padded3.shape == (1, 1, 32, 128)

    # Test error when target size is smaller
    with pytest.raises(ValueError, match="smaller than current size"):
        pad_dim_to_size(x, dim=-1, size=50)


def test_pad_dim_to_size_positive_dim():
    """Test pad_dim_to_size with positive dimension index."""
    from models.common.modules.tensor_utils import pad_dim_to_size

    x = torch.randn(2, 3, 4, 5)

    # Pad dim=0
    padded = pad_dim_to_size(x, dim=0, size=4)
    assert padded.shape == (4, 3, 4, 5)
    assert torch.equal(padded[:2], x)

    # Pad dim=1
    padded = pad_dim_to_size(x, dim=1, size=8)
    assert padded.shape == (2, 8, 4, 5)
    assert torch.equal(padded[:, :3], x)


if __name__ == "__main__":
    test_pad_dim_to_size()
    print("  ✓ test_pad_dim_to_size")

    test_pad_dim_to_size_positive_dim()
    print("  ✓ test_pad_dim_to_size_positive_dim")

    print("\nAll tensor_utils tests passed! ✓")
