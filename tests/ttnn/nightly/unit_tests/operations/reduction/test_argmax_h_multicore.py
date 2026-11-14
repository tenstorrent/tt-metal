# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
ROW_MAJOR ttnn.argmax over dim == rank-2 (issue #63942).

That reduction used to go through a single-core TILE program. When W % 32 == 0 it is now reshaped to
(..., H, W/32, 32) so the multi-core NC path handles it. These tests check numerical equivalence with
torch.argmax for the reshaped path and for the fallback (W % 32 != 0) that keeps the old path.
"""

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

TORCH_DTYPES = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def _distinct_values(shape, dtype):
    """Values that are unique along every reduction (no ties), exactly representable in bf16."""
    numel = 1
    for s in shape:
        numel *= s
    # bf16 has 8 bits of mantissa, so integers are only exact up to 256; float32 is exact far beyond any H used here.
    if dtype == torch.bfloat16:
        assert shape[-2] <= 256, "keep H <= 256 so values stay exact in bfloat16"
    flat = torch.stack([torch.randperm(shape[-2]) for _ in range(numel // shape[-2])])
    # flat: (numel / H, H) -> place H at dim -2
    lead = list(shape[:-2]) + [shape[-1]]
    x = flat.reshape(*lead, shape[-2]).movedim(-1, -2)
    return (x.to(torch.float32) - shape[-2] / 2).to(dtype)


def _run(device, torch_input, tt_dtype, dim, keepdim, **kwargs):
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = ttnn.argmax(tt_input, dim=dim, keepdim=keepdim, **kwargs)
    return tt_output, ttnn.to_torch(tt_output).to(torch.int64)


@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dim", [-2, "rank-2"])
@pytest.mark.parametrize(
    "shape",
    [
        (64, 32),  # rank 2
        (32, 64),
        (100, 96),  # H not a multiple of 32
        (4, 64, 128),  # rank 3
        (3, 37, 64),  # odd H
        (2, 3, 48, 96),  # rank 4
        (1, 1, 250, 512),  # large-ish H, W=512 as in the issue
        (48, 128, 512),  # issue shape with H reduced from 2048 (keeps values exact in bf16)
    ],
)
def test_argmax_rm_dim_rank_minus_2_multicore(device, shape, dim, keepdim, tt_dtype):
    torch.manual_seed(0)
    if dim == "rank-2":
        dim = len(shape) - 2
    torch_dtype = TORCH_DTYPES[tt_dtype]
    torch_input = _distinct_values(shape, torch_dtype)

    expected = torch.argmax(torch_input.to(torch.float32), dim=dim, keepdim=keepdim)
    tt_output, actual = _run(device, torch_input, tt_dtype, dim, keepdim)

    assert tt_output.dtype == ttnn.uint32
    assert list(tt_output.shape) == list(expected.shape)
    assert torch.equal(actual, expected.to(torch.int64))


@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape", [(4, 64, 33), (2, 40, 100), (50, 17)])
def test_argmax_rm_dim_rank_minus_2_non_multiple_of_32_width_fallback(device, shape, tt_dtype):
    """W % 32 != 0 keeps the single-core TILE path; results must still be correct."""
    torch.manual_seed(1)
    torch_input = _distinct_values(shape, TORCH_DTYPES[tt_dtype])
    expected = torch.argmax(torch_input.to(torch.float32), dim=-2, keepdim=False)
    _, actual = _run(device, torch_input, tt_dtype, -2, False)
    assert torch.equal(actual, expected.to(torch.int64))


@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16, ttnn.float32])
def test_argmax_rm_dim_rank_minus_2_ties_first_occurrence(device, tt_dtype):
    """Constant and heavily-tied inputs: torch returns the first maximal index."""
    torch_dtype = TORCH_DTYPES[tt_dtype]
    shape = (2, 64, 64)

    constant = torch.ones(shape, dtype=torch_dtype)
    _, actual = _run(device, constant, tt_dtype, -2, False)
    assert torch.equal(actual, torch.zeros(2, 64, dtype=torch.int64))

    torch.manual_seed(2)
    tied = torch.randint(0, 3, shape).to(torch_dtype)
    expected = torch.argmax(tied.to(torch.float32), dim=-2, keepdim=False)
    _, actual = _run(device, tied, tt_dtype, -2, False)
    assert torch.equal(actual, expected.to(torch.int64))


@pytest.mark.parametrize("tt_dtype", [ttnn.bfloat16, ttnn.float32])
def test_argmax_rm_dim_rank_minus_2_negative_values(device, tt_dtype):
    torch.manual_seed(3)
    shape = (3, 96, 64)
    # Shift the distinct values so that every element is negative (still distinct, still exact in bf16).
    torch_input = (_distinct_values(shape, torch.float32) - 100).to(TORCH_DTYPES[tt_dtype])
    expected = torch.argmax(torch_input.to(torch.float32), dim=-2, keepdim=True)
    _, actual = _run(device, torch_input, tt_dtype, -2, True)
    assert torch.equal(actual, expected.to(torch.int64))


def test_argmax_rm_dim_rank_minus_2_sub_core_grids(device):
    torch.manual_seed(4)
    shape = (8, 64, 128)
    torch_input = _distinct_values(shape, torch.float32)
    grids = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    expected = torch.argmax(torch_input, dim=-2, keepdim=False)
    _, actual = _run(device, torch_input, ttnn.float32, -2, False, sub_core_grids=grids)
    assert torch.equal(actual, expected.to(torch.int64))


def test_argmax_rm_dim_rank_minus_2_preallocated_output(device):
    torch.manual_seed(5)
    shape = (4, 64, 96)
    torch_input = _distinct_values(shape, torch.float32)
    expected = torch.argmax(torch_input, dim=-2, keepdim=False)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    prealloc = ttnn.from_torch(
        torch.zeros(expected.shape, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.argmax(tt_input, dim=-2, keepdim=False, output_tensor=prealloc)
    assert torch.equal(ttnn.to_torch(out).to(torch.int64), expected.to(torch.int64))
    assert torch.equal(ttnn.to_torch(prealloc).to(torch.int64), expected.to(torch.int64))


def test_argmax_rm_dim_rank_minus_2_issue_63942_shape(device):
    """Scaled-down version of the shape reported in #63942: (B, N, C) = (2, 2048, 512) float32, dim=1 (rank-2).

    The report uses B=48; B is reduced for speed, while H=2048 and W=512 (W/32 = 16, i.e. padded tiles) are kept.
    float32 is required: bfloat16 cannot hold a unique value per row once H > 256.
    """
    torch.manual_seed(0)
    shape = (2, 2048, 512)
    torch_input = _distinct_values(shape, torch.float32)
    expected = torch.argmax(torch_input, dim=1, keepdim=False)
    tt_output, actual = _run(device, torch_input, ttnn.float32, 1, False)
    assert list(tt_output.shape) == list(expected.shape)
    assert torch.equal(actual, expected.to(torch.int64))
