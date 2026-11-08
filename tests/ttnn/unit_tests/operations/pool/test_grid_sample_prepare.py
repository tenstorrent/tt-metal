# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
@pytest.mark.parametrize("input_shape", [[1, 40000, 64, 32], [1, 64, 40000, 32]])
def test_prepare_grid_sample_grid_rejects_coordinate_overflow(input_shape, mode, expect_error):
    # Precomputed grid coordinates are stored as int16 bit patterns, so a row or column index above INT16_MAX
    # used to be clamped to 32767 and the kernel sampled the wrong pixel with no error.
    grid = ttnn.from_torch(torch.tensor([[[[1.0, 1.0]]]], dtype=torch.float32), dtype=ttnn.float32)
    with expect_error(RuntimeError, "at most 32768"):
        ttnn.prepare_grid_sample_grid(grid, input_shape, mode=mode, align_corners=True)


def test_grid_sample_last_row_at_max_supported_height(device):
    # H - 1 == INT16_MAX is the largest representable coordinate, and sampling the bottom row is exact.
    input_h = 32768
    torch_input = torch.zeros(1, input_h, 1, 32, dtype=torch.bfloat16)
    torch_input[0, input_h - 1, 0, :] = 1.0
    grid_torch = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)  # x = left column, y = bottom row
    reference = torch.nn.functional.grid_sample(
        torch_input.permute(0, 3, 1, 2).float(), grid_torch, mode="nearest", align_corners=True
    ).permute(0, 2, 3, 1)

    prepared = ttnn.prepare_grid_sample_grid(
        ttnn.from_torch(grid_torch, dtype=ttnn.float32), list(torch_input.shape), mode="nearest", align_corners=True
    )
    coords = ttnn.to_torch(prepared).contiguous().view(torch.int16)
    assert coords[0, 0, 0, 0].item() == input_h - 1
    assert coords[0, 0, 0, 1].item() == 0

    output = ttnn.grid_sample(
        ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        ttnn.to_device(prepared, device),
        mode="nearest",
        align_corners=True,
        use_precomputed_grid=True,
    )
    assert torch.equal(ttnn.to_torch(output).float(), reference.float())
