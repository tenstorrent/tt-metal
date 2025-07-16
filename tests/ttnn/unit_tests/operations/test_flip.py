# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import itertools

from tests.ttnn.utils_for_testing import assert_with_pcc


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("c", [1, 2])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_flip_rm(device, n, c, h, w, dtype):
    torch.manual_seed(2005)
    shape = (n, c, h, w)
    torch_tensor = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_tensor, (2, 3))
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = ttnn.flip(tt_tensor, (2, 3))
    assert_with_pcc(torch_output, ttnn.to_torch(tt_output), 0.9999)


@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_flip_tiled(device, n, c, h, w, dtype):
    torch.manual_seed(2005)
    shape = (n, c, h, w)
    torch_tensor = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_tensor, (2, 3))
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.flip(tt_tensor, (2, 3))
    assert_with_pcc(torch_output, ttnn.to_torch(tt_output), 0.9999)
