# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize("shape", [(0, 8, 8, 16), (1, 0, 8, 16)])
def test_fold_empty(device, shape):
    # An empty input folds to an empty output; the DRAM factory divided by its zero patch count.
    torch_input = torch.zeros(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.fold(ttnn_input, 2, 2)
    assert tuple(output.shape) == (shape[0], shape[1] // 2, shape[2] // 2, shape[3] * 4)
    assert output.logical_volume() == 0


def test_fold_nonempty_control(device):
    # Non-empty inputs keep the pixel-unshuffle result.
    x = torch.randn(1, 8, 8, 16, dtype=torch.bfloat16)
    output = ttnn.to_torch(ttnn.fold(ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device), 2, 2)).float()
    reference = x.float().reshape(1, 4, 2, 4, 2, 16).permute(0, 1, 3, 2, 4, 5).reshape(1, 4, 4, 64)
    assert torch.equal(output.flatten().sort().values, reference.flatten().sort().values)
