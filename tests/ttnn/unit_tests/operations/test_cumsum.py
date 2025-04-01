# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

# Also TODO:
# 1) test with preallocated output
# 2) test with output.dtype != input.dtype
# 3) test with misc. tensor layouts


@pytest.mark.parametrize("size", [[0, 0, 0], [2, 3, 4]])
@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cumsum(size, dim, dtype):
    torch_input_tensor = torch.rand(size, dtype=dtype)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    output_tensor = ttnn.cumsum(input_tensor, dim=dim, dtype=dtype)

    # TODO: Finish implementation of ttnn.cumsum() and test against torch.cumsum()

    assert output_tensor.shape == (size,)
