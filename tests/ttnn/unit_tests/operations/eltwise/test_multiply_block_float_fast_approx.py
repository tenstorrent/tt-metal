# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""multiply honours an explicit fast_and_approximate_mode on block-float operands.

Block-float still defaults to the FPU path. What changed is that a caller who passes
False now gets the SFPU path. Before this change the argument was discarded, while
divide, bound by the same helper, honoured it for the same dtypes.

The explicit-False cases assert that a second program is compiled, not that the output
moves. Whether the two paths disagree on any given value is architecture and dtype
dependent. On Wormhole the scalar-first FPU kernel takes SrcA's format from the scalar
circular buffer instead of from the block-float tensor, so that operand order already
computes at bfloat16 precision and the SFPU path returns the same answer, and on that
architecture bfloat4_b agrees between the two paths in every operand order. A new
program-cache entry is the signal that holds everywhere, and reaching the SFPU path at
all is what the fix changed.
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


def _programs_added_by_an_explicit_false(device, call):
    """call(mode) runs the op, mode None meaning the argument is not passed at all.

    Returns how many program-cache entries an explicit False adds, after proving that
    repeating the default call adds none.
    """
    counter = device.cache_entries_counter
    call(None)

    counter.reset()
    with counter.measure():
        call(None)
    assert counter.total == 0, "the default call is not being cached"

    counter.reset()
    with counter.measure():
        call(False)
    return counter.total


def test_block_float_multiply_still_defaults_to_the_fast_path(device):
    a = _t(ALL_BF16.reshape(SHAPE), ttnn.bfloat8_b, device)
    b = _t(torch.full(SHAPE, 1.5), ttnn.bfloat8_b, device)
    assert _differ(ttnn.multiply(a, b), ttnn.multiply(a, b, fast_and_approximate_mode=True)) == 0


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_block_float_multiply_honours_an_explicit_false(device, dtype):
    a = _t(ALL_BF16.reshape(SHAPE), dtype, device)
    b = _t(torch.full(SHAPE, 1.5), dtype, device)

    def call(mode):
        return ttnn.multiply(a, b) if mode is None else ttnn.multiply(a, b, fast_and_approximate_mode=mode)

    added = _programs_added_by_an_explicit_false(device, call)
    assert added == 1, "an explicit False is still being discarded for block-float operands"


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


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_block_float_scalar_overload_honours_an_explicit_false(device, dtype):
    a = _t(ALL_BF16.reshape(SHAPE), dtype, device)

    def call(mode):
        return ttnn.multiply(a, 2.0) if mode is None else ttnn.multiply(a, 2.0, fast_and_approximate_mode=mode)

    added = _programs_added_by_an_explicit_false(device, call)
    assert added == 1, "an explicit False is still being discarded for the scalar overload"


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_block_float_scalar_first_overload_honours_an_explicit_false(device, dtype):
    a = _t(ALL_BF16.reshape(SHAPE), dtype, device)
    assert _differ(ttnn.multiply(2.0, a), ttnn.multiply(2.0, a, fast_and_approximate_mode=True)) == 0

    def call(mode):
        return ttnn.multiply(2.0, a) if mode is None else ttnn.multiply(2.0, a, fast_and_approximate_mode=mode)

    added = _programs_added_by_an_explicit_false(device, call)
    assert added == 1, "an explicit False is still being discarded for the scalar-first overload"
