# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Diagnostic tests for multi-axis bfloat16 reduction precision.

Background (PR #43367, Issue #43994):
PR #43367 fixed ttnn.sum precision for multi-axis bfloat16 reductions by
keeping intermediate results as FP32 between reduction stages instead of
packing them back to BF16 at each stage boundary.  That fix only applies
to ReduceType::Sum (via the ``chain_active`` flag in generic_reductions.cpp).

For Mean, Max, and Min, multi-axis reductions still iterate dim-by-dim
through ``reduce_nd_loop`` (in reduce_impl), converting back to BF16 at
each step.  This truncation can compound across stages and degrade
precision, especially when the shape is large in the reduced dimensions.

These tests probe that multi-stage precision.  They are DIAGNOSTIC:
  - If a test passes, the op's current precision is acceptable for the
    given shape and dim combination.
  - If a test fails, the op likely needs an FP32-chain fix analogous to
    the one applied to ttnn.sum in PR #43367.

Note on Prod: ttnn.prod is handled by a separate implementation in
prod.cpp, NOT by the generic_reductions two-stage path.  A basic
multi-dim prod test is included for completeness but it exercises a
different code path.

Note on Max/Min: the ``call_fast_nc`` function in generic_reductions.cpp
currently returns false for all types except Sum, so Max/Min multi-axis
reductions do NOT use the fast_reduce_nc two-stage split.  Instead they
go through reduce_nd_loop which reduces one dim at a time with
transpose-reduce-transpose for non-H/W dims.  Intermediate BF16 packing
still occurs between these sequential reductions.
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bf16_reference(torch_input_bf16, op, dim, keepdim=True):
    """Compute a float32 reference from a bf16 input tensor.

    We cast the bf16 input to fp32 and reduce there.  This isolates the
    device-side precision loss from the reduction itself.
    """
    fp32 = torch_input_bf16.float()
    if op == "mean":
        return torch.mean(fp32, dim=dim, keepdim=keepdim)
    elif op == "max":
        return torch.amax(fp32, dim=dim, keepdim=keepdim)
    elif op == "min":
        return torch.amin(fp32, dim=dim, keepdim=keepdim)
    elif op == "prod":
        # torch.prod only supports a single dim; for multi-dim we iterate.
        result = fp32
        for d in sorted(dim, reverse=True):
            result = torch.prod(result, dim=d, keepdim=True)
        if not keepdim:
            # Remove the reduced dims (they are all size-1 due to keepdim=True above).
            for d in sorted(dim):
                result = result.squeeze(d)
        return result
    else:
        raise ValueError(f"Unknown op: {op}")


# ---------------------------------------------------------------------------
# MEAN — multi-axis precision
# ---------------------------------------------------------------------------

# Dim combos that cross the N/C vs H/W boundary (i.e. reduce at least one
# of dims 0,1 AND at least one of the last two dims).  In a 4D tensor
# [N, C, H, W] with rank-relative indexing: H = -2, W = -1.
_MEAN_4D_SHAPE = (4, 8, 32, 64)
_MEAN_4D_DIMS = [
    [0, 3],  # N + W
    [0, 1, 3],  # N + C + W
    [1, 2, 3],  # C + H + W
    [0, 1, 2, 3],  # full reduction (all dims)
]

_MEAN_5D_SHAPE = (2, 4, 8, 32, 64)
_MEAN_5D_DIMS = [
    [0, 4],  # batch-0 + W
    [0, 1, 4],  # batch-0 + batch-1 + W
    [2, 3, 4],  # C + H + W
]


@pytest.mark.parametrize(
    "shape, dims",
    [pytest.param(_MEAN_4D_SHAPE, d, id=f"4D_dim{'_'.join(map(str, d))}") for d in _MEAN_4D_DIMS]
    + [pytest.param(_MEAN_5D_SHAPE, d, id=f"5D_dim{'_'.join(map(str, d))}") for d in _MEAN_5D_DIMS],
)
def test_mean_two_stage_precision(device, shape, dims):
    """Mean across mixed N/C + H/W dims — probes multi-stage bf16 precision."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16).clamp(-1.0, 1.0)
    torch_ref = _bf16_reference(torch_input, "mean", dim=dims, keepdim=True)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.mean(tt_input, dim=dims, keepdim=True)
    tt_output = ttnn.to_torch(tt_output)

    assert_numeric_metrics(
        torch_ref,
        tt_output,
        pcc_threshold=0.999,
        rtol=0.02,
        atol=0.002,
        frobenius_threshold=0.01,
    )


# ---------------------------------------------------------------------------
# MAX — multi-axis precision
# ---------------------------------------------------------------------------

_MAX_4D_SHAPE = (4, 8, 32, 64)
_MAX_4D_DIMS_SINGLE = [
    # Single dims that exercise each axis individually.  Two-stage (fast_nc +
    # hw reduce) is NOT triggered for single dims, but these establish a
    # baseline for comparison.
    0,
    1,
    2,
    3,
]

# Multi-dim max: ttnn.max supports a tuple of dims (it goes through
# reduce_nd_loop when dims span both non-HW and HW axes).
_MAX_4D_DIMS_MULTI = [
    [0, 3],  # N + W — crosses the N/C vs H/W boundary
    [1, 3],  # C + W
    [0, 2, 3],  # N + H + W
]

# Tile padding for max: use a very negative value so it can never be the max.
_MAX_PAD_VALUE = -1e6


@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param(_MAX_4D_SHAPE, d, id=f"4D_dim{'_'.join(map(str, [d] if isinstance(d, int) else d))}")
        for d in _MAX_4D_DIMS_SINGLE
    ]
    + [pytest.param(_MAX_4D_SHAPE, d, id=f"4D_dim{'_'.join(map(str, d))}") for d in _MAX_4D_DIMS_MULTI],
)
def test_max_two_stage_precision(device, shape, dim):
    """Max across mixed dims — probes multi-stage bf16 precision.

    Max should be exact (order-independent), so tight tolerances are used.
    Any failure indicates a bug in intermediate packing or padding.
    """
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16).mul(100)
    if isinstance(dim, int):
        torch_ref = torch.max(torch_input.float(), dim=dim, keepdim=True).values
    else:
        torch_ref = torch.amax(torch_input.float(), dim=dim, keepdim=True)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    # Fill tile padding with a very negative sentinel so it can never be the max.
    tt_input = ttnn.fill_implicit_tile_padding(tt_input, _MAX_PAD_VALUE)
    tt_output = ttnn.max(tt_input, dim=dim, keepdim=True)
    tt_output = ttnn.to_torch(tt_output)

    # Max should be exact for bf16 inputs — the maximum element is a value
    # that already exists in the tensor, so no rounding should occur.
    assert_numeric_metrics(
        torch_ref,
        tt_output,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-06,
    )


# ---------------------------------------------------------------------------
# MIN — multi-axis precision
# ---------------------------------------------------------------------------

_MIN_4D_SHAPE = (4, 8, 32, 64)
_MIN_4D_DIMS_SINGLE = [0, 1, 2, 3]
_MIN_4D_DIMS_MULTI = [
    [0, 3],  # N + W
    [1, 3],  # C + W
    [0, 2, 3],  # N + H + W
]

# Use a padding value that cannot be the minimum of any real data.
_MIN_PAD_VALUE = 1e6


@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param(_MIN_4D_SHAPE, d, id=f"4D_dim{'_'.join(map(str, [d] if isinstance(d, int) else d))}")
        for d in _MIN_4D_DIMS_SINGLE
    ]
    + [pytest.param(_MIN_4D_SHAPE, d, id=f"4D_dim{'_'.join(map(str, d))}") for d in _MIN_4D_DIMS_MULTI],
)
def test_min_two_stage_precision(device, shape, dim):
    """Min across mixed dims — probes multi-stage bf16 precision.

    Like max, min should be exact.  Tight tolerances are used.
    """
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16).mul(100)
    if isinstance(dim, int):
        torch_ref = torch.min(torch_input.float(), dim=dim, keepdim=True).values
    else:
        torch_ref = torch.amin(torch_input.float(), dim=dim, keepdim=True)

    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    # Fill tile padding with a large positive value so it can never be the min.
    tt_input = ttnn.fill_implicit_tile_padding(tt_input, _MIN_PAD_VALUE)

    tt_output = ttnn.min(tt_input, dim=dim, keepdim=True)
    tt_output = ttnn.to_torch(tt_output)

    assert_numeric_metrics(
        torch_ref,
        tt_output,
        pcc_threshold=0.9999,
        rtol=1e-06,
        atol=1e-06,
        frobenius_threshold=1e-06,
    )


# ---------------------------------------------------------------------------
# PROD — multi-axis precision (separate code path via prod.cpp)
# ---------------------------------------------------------------------------

# Smaller shapes to avoid overflow/underflow in products.
_PROD_SHAPE = (4, 4, 32, 32)
_PROD_DIMS = [
    # ttnn.prod only supports single-dim reduction, so we test individual dims.
    # Multi-dim prod would need iterative calls; the issue is whether
    # intermediate bf16 packing between sequential prod reductions loses
    # precision.
    0,
    1,
    2,
    3,
]


@pytest.mark.parametrize(
    "shape, dim",
    [pytest.param(_PROD_SHAPE, d, id=f"dim{d}") for d in _PROD_DIMS],
)
def test_prod_two_stage_precision(device, shape, dim):
    """Prod along individual dims — baseline precision check.

    ttnn.prod is implemented in prod.cpp (not generic_reductions) and only
    supports single-dim reduction.  This test establishes a precision
    baseline.  Values are drawn from [0.9, 1.1] to keep products from
    exploding or collapsing to zero.
    """
    torch.manual_seed(42)
    # Narrow range to avoid overflow/underflow in products.
    torch_input = torch.empty(shape, dtype=torch.bfloat16).uniform_(0.9, 1.1)
    torch_ref = torch.prod(torch_input.float(), dim=dim, keepdim=True)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output = ttnn.prod(tt_input, dim=dim, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output, dtype=torch.float32)

    # Prod is inherently imprecise in bf16 — use loose tolerances.
    assert_numeric_metrics(
        torch_ref,
        tt_output,
        pcc_threshold=0.99,
        rtol=0.05,
        atol=0.5,
        frobenius_threshold=0.1,
    )
