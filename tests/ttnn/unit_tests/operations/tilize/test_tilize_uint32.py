# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — uint32 integer passthrough for tilize.

Integer tilize is a pure passthrough: the tilize LLK reorders bytes with no
arithmetic and no cast, so the read-back tensor must be BIT-EXACT to the input
values (comp_equal, not PCC). Integers only pair with the same integer dtype —
int<->float crosses are out of contract, but under the registry model they are
pruned by INVALID in feature_spec.py (test-side skip) rather than refused by the
op, so the op is intentionally agnostic to them and this suite does not exercise
them.

The refinement TARGET (feature_spec.py) is uint32; uint32 stands in for the
integer passthrough family, with uint16 / int32 covered here and in
test_regression.py. All three go through the identical (non-fast) tilize path.
"""

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize


# rank 2/3/4, single-tile, multi-tile, non-square.
SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 96, 32),  # non-square (tall)
    (1, 1, 32, 96),  # non-square (wide)
    (2, 3, 64, 64),  # multi-batch, rank 4
    (128, 256),  # rank 2
    (4, 32, 64),  # rank 3
]

# Integer passthrough dtypes. uint16/int32 use int32 as the torch source dtype
# (torch has no uint32/uint16); a small unsigned-safe range keeps int32 negative
# values in test_regression's domain while staying unambiguous here.
INT_DTYPES = [ttnn.uint32, ttnn.uint16, ttnn.int32]


def _torch_int_ref(dtype, shape):
    torch.manual_seed(42)
    lo = -1000 if dtype == ttnn.int32 else 0
    return torch.randint(lo, 1000, shape, dtype=torch.int32)


@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("use_multicore", [False, True])
def test_tilize_int_identity(device, dtype, shape, use_multicore):
    """tilize(RM int) -> TILE int: bit-exact identity, single & multi-core."""
    torch_input = _torch_int_ref(dtype, shape)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_output = tilize(tt_input, use_multicore=use_multicore)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.dtype == dtype
    assert list(tt_output.shape) == list(shape)

    output = ttnn.to_torch(tt_output).to(torch.int32)
    assert torch.equal(
        output, torch_input
    ), f"int identity mismatch dtype={dtype} shape={shape} multicore={use_multicore}"


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_tilize_int_explicit_dtype(device, dtype):
    """Explicit same-dtype kwarg on an integer input (no-cast passthrough)."""
    shape = (1, 1, 64, 128)
    torch_input = _torch_int_ref(dtype, shape)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_output = tilize(tt_input, dtype=dtype)

    assert tt_output.dtype == dtype
    output = ttnn.to_torch(tt_output).to(torch.int32)
    assert torch.equal(output, torch_input)


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_tilize_int_explicit_memory_config(device, dtype):
    """Integer input with an explicit output memory_config (L1 interleaved)."""
    shape = (1, 1, 64, 128)
    torch_input = _torch_int_ref(dtype, shape)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_output = tilize(tt_input, memory_config=ttnn.L1_MEMORY_CONFIG)

    assert tt_output.dtype == dtype
    output = ttnn.to_torch(tt_output).to(torch.int32)
    assert torch.equal(output, torch_input)
