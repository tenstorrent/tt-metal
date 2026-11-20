# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("shape", [(1, 8, 8, 0), (0, 8, 8, 32)])
def test_rotate_empty(device, shape):
    # An empty input rotates to an empty output; the factories divided by its dims.
    torch_input = torch.zeros(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.rotate(ttnn_input, 45.0)
    assert tuple(output.shape) == shape
    assert output.logical_volume() == 0


def test_rotate_nonempty_control(device):
    # A zero-degree rotation of a non-empty input is the identity.
    x = torch.randn(1, 8, 8, 32, dtype=torch.bfloat16)
    output = ttnn.to_torch(ttnn.rotate(ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device), 0.0))
    assert_equal(output, x)
