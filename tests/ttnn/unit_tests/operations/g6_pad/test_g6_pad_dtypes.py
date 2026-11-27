# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""32-bit dtypes, rank 2 and L1 placement on the TILE regimes (verifier drift fix of SUPPORTED).

Production cases only (BB-22), plus int32 on a production TILE shape: int32 is in TARGET and the
TILE kernels are elem_bytes-generic, but no golden TILE cell carries it. Inputs use the full
integer range and a poisoned tile padding; the gate is bit equality.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_pad import g6_pad

from .test_g6_pad import to_torch_pad

_TORCH = {
    ttnn.float32: torch.float32,
    ttnn.int32: torch.int32,
    ttnn.uint32: torch.uint32,
    ttnn.bfloat16: torch.bfloat16,
}
_VIEW = {2: torch.int16, 4: torch.int32}

CASES = [
    # (shape, padding, value, dtype, memory)
    ((5, 12, 128, 128), ((0, 3), (0, 0), (0, 0), (0, 0)), 0, ttnn.float32, "dram"),  # p1, leading, tile_aligned
    ((1, 3844, 12), ((0, 0), (0, 28), (0, 0)), 0.0, ttnn.float32, "dram"),  # case 249, extension
    ((1, 196), ((0, 60),), 0, ttnn.uint32, "dram"),  # case 63, rank 2, padded_dims w
    ((1, 207), ((0, 49),), 0, ttnn.uint32, "dram"),  # case 70
    ((1, 196, 128), ((0, 60), (0, 0)), -1, ttnn.int32, "dram"),  # int32 TILE, new_storage
    ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 24), (0, 0)), -7, ttnn.int32, "dram"),  # int32 TILE, extension
    ((1, 1, 2560), ((0, 0), (0, 31), (0, 0)), 0.0, ttnn.bfloat16, "l1"),  # case 299, L1
]


def _input(shape, dtype):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int64).to(torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**32, shape, dtype=torch.int64).to(torch.uint32)
    return (torch.randn(shape) * 100).to(_TORCH[dtype])


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{'x'.join(map(str, c[0]))}-{c[3]}-{c[4]}")
def test_g6_pad_dtypes(device, case):
    torch.manual_seed(7)
    shape, padding, value, dtype, memory = case
    x = _input(shape, dtype)
    expected = torch.nn.functional.pad(x, to_torch_pad(padding), mode="constant", value=value)
    mem = ttnn.L1_MEMORY_CONFIG if memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
    x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    x_tt = ttnn.fill_implicit_tile_padding(x_tt, 77)
    out = g6_pad(x_tt, padding, value=value)
    assert out.dtype == dtype and out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config().buffer_type == mem.buffer_type
    got = ttnn.to_torch(out)
    assert list(got.shape) == list(expected.shape)
    view = _VIEW[expected.element_size()]
    assert torch.equal(got.contiguous().view(view), expected.contiguous().view(view))
