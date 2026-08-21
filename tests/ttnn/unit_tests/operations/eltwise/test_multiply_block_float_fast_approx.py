# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""multiply honours an explicit fast_and_approximate_mode on block-float operands.

Block-float still defaults to the FPU path. What changed is that a caller who passes
False now gets the SFPU path, where before the argument was discarded — while divide,
bound by the same helper, honoured it for the same dtypes.
"""

import pytest
import torch
import ttnn

SHAPE = (1, 1, 256, 256)
ALL_BF16 = torch.arange(0, 65536, dtype=torch.int64).to(torch.int32).to(torch.int16).view(torch.bfloat16)


def _t(x, dtype, device):
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    return ttnn.from_torch(x.to(torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _differ(a, b):
    x, y = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
    return ((x != y) & ~(torch.isnan(x) & torch.isnan(y))).sum().item()


def test_block_float_multiply_still_defaults_to_the_fast_path(device):
    a = _t(ALL_BF16.reshape(SHAPE), ttnn.bfloat8_b, device)
    b = _t(torch.full(SHAPE, 1.5), ttnn.bfloat8_b, device)
    assert _differ(ttnn.multiply(a, b), ttnn.multiply(a, b, fast_and_approximate_mode=True)) == 0


def test_block_float_multiply_honours_an_explicit_false(device):
    a = _t(ALL_BF16.reshape(SHAPE), ttnn.bfloat8_b, device)
    b = _t(torch.full(SHAPE, 1.5), ttnn.bfloat8_b, device)
    moved = _differ(ttnn.multiply(a, b, fast_and_approximate_mode=False), ttnn.multiply(a, b))
    assert moved > 0, "an explicit False is still being discarded for block-float operands"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_non_block_float_defaults_are_unchanged(device, dtype):
    a = _t(ALL_BF16.reshape(SHAPE), dtype, device)
    b = _t(torch.full(SHAPE, 1.5), dtype, device)
    assert _differ(ttnn.multiply(a, b), ttnn.multiply(a, b, fast_and_approximate_mode=False)) == 0


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32])
def test_divide_default_is_unchanged(device, dtype):
    a = _t(ALL_BF16.reshape(SHAPE), dtype, device)
    b = _t(torch.full(SHAPE, 1.5), dtype, device)
    assert _differ(ttnn.divide(a, b), ttnn.divide(a, b, fast_and_approximate_mode=False)) == 0


def test_block_float_scalar_overload_honours_an_explicit_false(device):
    a = _t(ALL_BF16.reshape(SHAPE), ttnn.bfloat8_b, device)
    moved = _differ(ttnn.multiply(a, 2.0, fast_and_approximate_mode=False), ttnn.multiply(a, 2.0))
    assert moved > 0, "an explicit False is still being discarded for the scalar overload"
