# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tril and triu build their 0/1 mask in the input's own dtype.

A bfloat16 mask against an integer input promoted the multiply to bfloat16, so the
kept elements came back rounded to 8 mantissa bits and the returned dtype was
BFLOAT16 rather than the input's.
"""

import pytest
import torch
import ttnn

SHAPE = (1, 1, 64, 64)
N = SHAPE[2] * SHAPE[3]


def _t(x, dtype, device):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("start", [1, 100000, 2**30])
@pytest.mark.parametrize("diagonal", [0, 1, -1, 5, -7])
@pytest.mark.parametrize("op, golden", [("tril", torch.tril), ("triu", torch.triu)])
def test_integer_inputs_keep_their_dtype_and_their_value(device, dtype, start, diagonal, op, golden):
    x = torch.arange(start, start + N, dtype=torch.int32).reshape(SHAPE)
    got = getattr(ttnn, op)(_t(x, dtype, device), diagonal=diagonal)
    assert got.dtype == dtype, f"{op} on {dtype} returned {got.dtype}"
    assert torch.equal(ttnn.to_torch(got).to(torch.int64), golden(x, diagonal=diagonal).to(torch.int64))


@pytest.mark.parametrize("dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
@pytest.mark.parametrize("diagonal", [0, 1, -1])
@pytest.mark.parametrize("op, golden", [("tril", torch.tril), ("triu", torch.triu)])
def test_float_inputs_are_unchanged(device, dtype, torch_dtype, diagonal, op, golden):
    torch.manual_seed(0)
    x = (torch.rand(SHAPE) * 8 - 4).to(torch_dtype)
    got = ttnn.to_torch(getattr(ttnn, op)(_t(x, dtype, device), diagonal=diagonal))
    assert got.dtype == torch_dtype
    assert torch.equal(got.float(), golden(x.float(), diagonal=diagonal))
