# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.div honours the dtype it is given.

The four non-int32 call sites passed std::nullopt into the dtype slot while a
live output_dtype sat in scope, so the requested dtype was dropped and the
result came back in the lhs dtype.
"""

import pytest
import torch
import ttnn

SHAPE = (1, 1, 32, 32)


def _t(v, dtype, torch_dtype, device):
    return ttnn.from_torch(torch.full(SHAPE, v, dtype=torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize(
    "dtype, torch_dtype, requested",
    [
        (ttnn.bfloat16, torch.bfloat16, ttnn.float32),
        (ttnn.float32, torch.float32, ttnn.bfloat16),
        (ttnn.bfloat16, torch.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
def test_div_returns_the_requested_dtype(device, dtype, torch_dtype, requested, rounding_mode):
    a, b = _t(7, dtype, torch_dtype, device), _t(2, dtype, torch_dtype, device)
    for got in (
        ttnn.div(a, b, rounding_mode=rounding_mode, dtype=requested),
        ttnn.div(a, 2, rounding_mode=rounding_mode, dtype=requested),
    ):
        assert got.dtype == requested, f"asked {requested}, got {got.dtype}"
        assert ttnn.to_torch(got).float().flatten()[0].item() == (3.5 if rounding_mode is None else 3)


@pytest.mark.parametrize("dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
@pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
def test_div_without_dtype_still_follows_the_lhs(device, dtype, torch_dtype, rounding_mode):
    a, b = _t(7, dtype, torch_dtype, device), _t(2, dtype, torch_dtype, device)
    assert ttnn.div(a, b, rounding_mode=rounding_mode).dtype == dtype
    assert ttnn.div(a, 2, rounding_mode=rounding_mode).dtype == dtype
