# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("dim,", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]])
@pytest.mark.parametrize("dtype", [(torch.bfloat16, ttnn.bfloat16)])
def test_cumprod(dim, shape, dtype, device):
    torch.manual_seed(22041997)

    input_tensor = torch.randn(*shape, dtype=dtype[0])
    result_tensor_torch = torch.cumprod(input_tensor, 0)
    ttnn_tensor = ttnn.from_torch(input_tensor, dtype[1], layout=ttnn.Layout.TILE, device=device)
    result_tensor = ttnn.experimental.cumprod(ttnn_tensor, 0)

    assert ttnn_tensor.shape == result_tensor.shape
    assert ttnn_tensor.dtype == result_tensor.dtype
    assert input_tensor.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == result_tensor.shape
