# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Full block-float permutation sweep for ttnn.permute (see #48813).

ttnn::permute typecasts bfloat8_b to bfloat16 and back when a permutation has no native
block-float pattern; bfloat4_b never gets that fallback. This sweeps every 4-D permutation
for both block-float types, on an aligned and an unaligned shape.
"""

import itertools

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

_PERMS = list(itertools.permutations(range(4)))

_SHAPES = [(2, 3, 32, 64), (2, 3, 32, 60)]

# PCC floor per dtype. bfloat4_b carries a 3-bit mantissa, so even a perfectly ordered
# result sits well below the bfloat8_b floor; both values are measured, not guessed.
_PCC = {ttnn.bfloat8_b: 0.9999, ttnn.bfloat4_b: 0.99}

# bfloat4_b permutations that abort with a misaligned NoC read instead of returning a wrong
# result. They cannot be xfail: the process dies and takes the rest of the session with it.
_BFP4_MISALIGNED = {
    (0, 2, 3, 1),
    (0, 3, 2, 1),
    (1, 2, 3, 0),
    (1, 3, 2, 0),
    (2, 0, 3, 1),
    (2, 1, 3, 0),
    (2, 3, 0, 1),
    (2, 3, 1, 0),
    (3, 0, 2, 1),
    (3, 1, 2, 0),
    (3, 2, 0, 1),
    (3, 2, 1, 0),
}

# bfloat4_b permutations that stay within the PCC floor without the bfloat16 fallback.
_BFP4_EXPECTED_PASS = {(0, 1, 2, 3), (1, 0, 2, 3)}


def _params():
    for dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        for shape in _SHAPES:
            for perm in _PERMS:
                marks = ()
                if dtype == ttnn.bfloat4_b and perm not in _BFP4_MISALIGNED and perm not in _BFP4_EXPECTED_PASS:
                    marks = (pytest.mark.xfail(reason="#48813: bfloat4_b has no bfloat16 permute fallback"),)
                name = "bfloat8_b" if dtype == ttnn.bfloat8_b else "bfloat4_b"
                pid = f"{name}-{'x'.join(map(str, shape))}-{''.join(map(str, perm))}"
                yield pytest.param(dtype, shape, perm, marks=marks, id=pid)


@pytest.mark.parametrize("dtype, shape, perm", list(_params()))
def test_permute_block_float(device, dtype, shape, perm):
    if dtype == ttnn.bfloat4_b and perm in _BFP4_MISALIGNED:
        pytest.skip("#48813: bfloat4_b permute aborts on a misaligned NoC read for this permutation")

    torch.manual_seed(2005)
    torch_input = torch.randn(*shape)
    torch_output = torch.permute(torch_input, perm)

    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    tt_output = ttnn.to_torch(ttnn.permute(tt_input, perm))

    assert_with_pcc(torch_output, tt_output, _PCC[dtype])
