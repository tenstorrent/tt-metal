# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


# def test_manual_seed(device):
#     """Test manual_seed with explicit keyword arguments for device and seeds (integer scalar)."""
#     ttnn.manual_seed(seeds=42, device=device)


def test_manual_seed_with_user_id(device):
    """Test manual_seed with both seed and user_id as integer scalars using keyword arguments."""
    ttnn.manual_seed(seeds=42, device=device, user_ids=7)


# def test_manual_short(device):
#     """Test manual_seed using positional arguments with only seed (shorthand syntax)."""
#     ttnn.manual_seed(42, device=device)


# def test_manual_tensors(device):
#     """Test manual_seed with tensor inputs for both seeds and user_ids, verifying tensor-based API."""
#     seed_tensor = ttnn.from_torch(torch.Tensor([42]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device)
#     user_id_tensor = ttnn.from_torch(torch.Tensor([7]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device)
#     ttnn.manual_seed(seeds=seed_tensor, device=device, user_ids=user_id_tensor)


# def test_manual_tensors_wrong_config(device):
#     """Test that manual_seed raises ValueError when mixing tensor seeds with scalar user_ids (invalid configuration)."""
#     seed_tensor = ttnn.from_torch(torch.Tensor([42]), dtype=ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device)
#     with pytest.raises(Exception, match="Seeds were provided as a tensor, so user_ids must not be provided as a scalar."):
#         ttnn.manual_seed(seeds=seed_tensor, device=device, user_ids=7)
