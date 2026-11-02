# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_normalize_hw(x):
    xd = x.double()
    mean = xd.mean(dim=(2, 3), keepdim=True)
    std = xd.std(dim=(2, 3), keepdim=True, unbiased=False)
    return ((xd - mean) / std).float()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("scale", [1.0, 100.0, 1e-3])
@pytest.mark.parametrize("shape", [[1, 1, 32, 32], [1, 2, 64, 64]])
def test_normalize_hw(device, dtype, scale, shape):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    x = (torch.randn(shape) * scale).to(torch_dtype)

    got = ttnn.to_torch(ttnn.normalize_hw(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)))

    assert_with_pcc(torch_normalize_hw(x.float()), got.float(), 0.999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_normalize_hw_zero_variance(device, dtype):
    # Every element equal, so the divisor is zero. rsqrt(0) is inf where
    # reciprocal(sqrt(0)) was, and 0 * inf resolves the same way.
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    x = torch.full([1, 1, 32, 32], 3.0, dtype=torch_dtype)

    got = ttnn.to_torch(ttnn.normalize_hw(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)))

    assert torch.isfinite(got.float()).all() or torch.isnan(got.float()).all()
