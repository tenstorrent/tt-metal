# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# The H and W reduction accumulates each batch's tiles into one DEST tile, one block call per batch on one init, with odd and even tile counts.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 8, 96, 96), id="1x8x96x96"),
        pytest.param((2, 3, 64, 160), id="2x3x64x160"),
        pytest.param((4, 1, 32, 672), id="4x1x32x672"),
        pytest.param((1, 3, 128, 128), id="1x3x128x128"),
        pytest.param((1, 5, 32, 64), id="1x5x32x64"),
    ],
)
@pytest.mark.parametrize("op", ["sum", "mean"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_reduce_hw_batches(device, shape, op, dtype):
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = getattr(ttnn, op)(x, dim=[2, 3], keepdim=True)
    expected = getattr(torch, op)(t, dim=(2, 3), keepdim=True)
    assert_with_pcc(expected, ttnn.to_torch(out).to(torch.float32), 0.999)
