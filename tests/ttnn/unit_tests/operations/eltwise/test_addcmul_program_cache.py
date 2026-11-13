# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The ternary ops key their program on width and broadcast pattern, not on height: addcmul must not compile per length."""

import pytest
import torch

import ttnn


def _addcmul(device, shapes, value=0.5):
    tensors = [torch.randn(shape) for shape in shapes]
    tt = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in tensors]
    out = ttnn.to_torch(ttnn.addcmul(*tt, value=value)).float()
    a, b, c = (t.to(torch.bfloat16).float() for t in tensors)
    torch.testing.assert_close(out, a + value * b * c, rtol=1.6e-2, atol=1e-1)


@pytest.mark.parametrize("width", [1024])
def test_addcmul_program_cache_is_row_free(device, width):
    from tests.ttnn.utils_for_testing import assert_program_cache_flat

    def run(rows):
        _addcmul(device, [(1, 1, rows, width)] * 3)

    assert assert_program_cache_flat(device, run, [4096, 2048, 96, 32]) == 1


def test_addcmul_program_cache_separates_row_broadcast(device):
    """A single-row operand is still its own program."""
    from tests.ttnn.utils_for_testing import assert_program_cache_flat

    def run(case):
        rows, bcast_rows = case
        _addcmul(device, [(1, 1, rows, 1024), (1, 1, rows, 1024), (1, 1, bcast_rows, 1024)])

    assert_program_cache_flat(device, run, [(2048, 2048), (1024, 1024), (2048, 1)], miss_indices=(2,))
