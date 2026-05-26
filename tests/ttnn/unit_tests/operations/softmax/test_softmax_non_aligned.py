# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Non-tile-aligned shape matrix for ttnn.operations.softmax.softmax —
Refinement 4.

This test exercises the partial-scaler routing added in this refinement:

- ``SUPPORTED["alignment"]`` now contains ``"w_non_aligned"`` and
  ``"h_non_aligned"`` alongside ``"tile_aligned"``.
- The program descriptor computes ``partial = reduce_dim_size % 32`` from
  the *logical* shape and (a) sizes each scaler CB to 2 tiles when
  ``partial > 0``, (b) routes ``calculate_and_prepare_partial_reduce_scalers``
  in the reader, (c) hands ``ReducePartialScaler::last_tile_at(1)`` to
  ``reduce<MAX>`` (Pass 1) and ``accumulate_reduce_block<SUM>`` (Pass 2).

The matrix covers four orthogonal corners:

1. Reduce axis non-aligned — the partial scaler runs.
2. Non-reduce axis non-aligned — no partial scaler, but ``Ht``/``Wt`` use
   ceil division so the storage tile count is right.
3. Both axes non-aligned — combines (1) and (2). The alignment tag is
   ``"w_non_aligned"`` (W check wins) but both ``dim=-1`` and ``dim=-2``
   exercise the partial-scaler path.
4. Garbage in implicit padding via ``ttnn.fill_implicit_tile_padding`` —
   verifies the partial scaler actually masks the padded positions instead
   of accidentally consuming zero-padding (the same trick
   ``toy_reduce_partial``'s tests use).

Both ``numeric_stable`` modes and both layouts (TILE / ROW_MAJOR) are
swept so the bf16-touching paths and the RM-wrapper (Refinement 3) hold
under the new shape envelope.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.softmax import softmax


# --------------------------------------------------------------------------
# Shape sets — each ``(shape, expected_alignment_tag)`` documents which
# axis is non-aligned and which alignment value the tagger emits.
#
# The ``partial`` selection in the program descriptor is by ``dim``, not by
# tag — see the docstring above. These shapes cover the four corners.
# --------------------------------------------------------------------------
W_NON_ALIGNED_SHAPES = [
    pytest.param((1, 1, 32, 50), id="r4_1x1x32x50_partialW=18"),
    pytest.param((1, 1, 64, 17), id="r4_1x1x64x17_partialW=17"),
    pytest.param((1, 1, 64, 33), id="r4_1x1x64x33_partialW=1"),
    pytest.param((1, 1, 32, 63), id="r4_1x1x32x63_partialW=31"),
    pytest.param((2, 1, 32, 100), id="r4_2x1x32x100_partialW=4"),
]

H_NON_ALIGNED_SHAPES = [
    pytest.param((1, 1, 17, 64), id="r4_1x1x17x64_partialH=17"),
    pytest.param((1, 1, 50, 128), id="r4_1x1x50x128_partialH=18"),
    pytest.param((1, 1, 33, 32), id="r4_1x1x33x32_partialH=1"),
    pytest.param((1, 1, 63, 64), id="r4_1x1x63x64_partialH=31"),
    pytest.param((2, 1, 100, 64), id="r4_2x1x100x64_partialH=4"),
]

BOTH_NON_ALIGNED_SHAPES = [
    pytest.param((1, 1, 17, 50), id="r4_1x1x17x50"),
    pytest.param((2, 1, 100, 47), id="r4_2x1x100x47"),
    pytest.param((1, 1, 33, 33), id="r4_1x1x33x33"),
]

# Rank-3 and rank-2 variants — verify Refinement 3's entry-point
# canonicalisation still composes with Refinement 4's partial scaler.
LOW_RANK_NON_ALIGNED_SHAPES = [
    pytest.param((4, 128, 47), id="r3_4x128x47_partialW=15"),
    pytest.param((1, 17, 128), id="r3_1x17x128_partialH=17"),
    pytest.param((128, 100), id="r2_128x100_partialW=4"),
    pytest.param((17, 64), id="r2_17x64_partialH=17"),
]


LAYOUTS = [
    pytest.param(ttnn.TILE_LAYOUT, id="tile"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="row_major"),
]


# --------------------------------------------------------------------------
# 1. Reduce-axis non-aligned (W non-aligned + dim=-1).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", W_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_w_non_aligned_reduce_w(device, shape, layout, numeric_stable):
    """W % 32 != 0 with dim=-1: partial scaler masks padded columns."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=-1)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = softmax(ttnn_input, dim=-1, numeric_stable=numeric_stable)
    assert tuple(ttnn_output.shape) == shape
    assert ttnn_output.layout == layout

    assert_with_pcc(torch_expected, ttnn.to_torch(ttnn_output), 0.999)


# --------------------------------------------------------------------------
# 2. Non-reduce-axis non-aligned (W non-aligned + dim=-2): aligned reduce,
#    non-aligned non-reduce axis. No partial scaler; verifies ceil-Ht/Wt.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", W_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_w_non_aligned_reduce_h(device, shape, layout, numeric_stable):
    """W % 32 != 0 with dim=-2: reduce H (aligned), non-aligned axis is
    the non-reduce axis — no partial scaler runs but Ht/Wt must use ceil
    division so the storage tile count is right."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=-2)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = softmax(ttnn_input, dim=-2, numeric_stable=numeric_stable)
    assert tuple(ttnn_output.shape) == shape
    assert ttnn_output.layout == layout

    assert_with_pcc(torch_expected, ttnn.to_torch(ttnn_output), 0.999)


# --------------------------------------------------------------------------
# 3. H non-aligned, reduce H — partial scaler on the H (REDUCE_COL) path.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", H_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_h_non_aligned_reduce_h(device, shape, layout, numeric_stable):
    """H % 32 != 0 with dim=-2: partial scaler masks padded rows."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=-2)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = softmax(ttnn_input, dim=-2, numeric_stable=numeric_stable)
    assert tuple(ttnn_output.shape) == shape
    assert ttnn_output.layout == layout

    assert_with_pcc(torch_expected, ttnn.to_torch(ttnn_output), 0.999)


# --------------------------------------------------------------------------
# 4. H non-aligned, reduce W (aligned) — non-aligned non-reduce axis.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", H_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_h_non_aligned_reduce_w(device, shape, layout, numeric_stable):
    """H % 32 != 0 with dim=-1: reduce W (aligned). Partial scaler is *not*
    used; the non-reduce H axis just yields padded-row output that
    read-back discards."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=-1)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = softmax(ttnn_input, dim=-1, numeric_stable=numeric_stable)
    assert tuple(ttnn_output.shape) == shape
    assert ttnn_output.layout == layout

    assert_with_pcc(torch_expected, ttnn.to_torch(ttnn_output), 0.999)


# --------------------------------------------------------------------------
# 5. Both axes non-aligned — combines partial scaler on the reduce axis
#    with ceil-Ht/Wt sizing on the non-reduce axis.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", BOTH_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_both_non_aligned(device, shape, dim, layout, numeric_stable):
    """Both H and W non-aligned. Tagger emits ``w_non_aligned`` (W check
    wins) but both ``dim=-1`` and ``dim=-2`` exercise the partial-scaler
    path because the reduce axis is also non-aligned."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = softmax(ttnn_input, dim=dim, numeric_stable=numeric_stable)
    assert tuple(ttnn_output.shape) == shape
    assert ttnn_output.layout == layout

    assert_with_pcc(torch_expected, ttnn.to_torch(ttnn_output), 0.999)


# --------------------------------------------------------------------------
# 6. Rank-2 / rank-3 + non-aligned — verifies Refinement 3's entry-point
# canonicalisation composes correctly with the partial scaler path.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", LOW_RANK_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_softmax_non_aligned_low_rank(device, shape, dim, layout):
    """Rank-2 / rank-3 shapes with non-aligned axes. The entry point
    unsqueezes to rank-4 first, then the partial-scaler path runs
    against the canonical 4D layout."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = softmax(ttnn_input, dim=dim)
    assert tuple(ttnn_output.shape) == shape
    assert ttnn_output.layout == layout

    assert_with_pcc(torch_expected, ttnn.to_torch(ttnn_output), 0.999)


# --------------------------------------------------------------------------
# 7. Partial-scaler stress: fill implicit tile padding with non-zero
#    "garbage" before running softmax. If the partial scaler is broken
#    (or the padding is being read as a real value), the result will
#    diverge from torch's reference. The same trick the toy_reduce_partial
#    test uses to verify the mask, lifted onto softmax.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param((1, 1, 32, 50), -1, id="W_partial_garbage_dim_-1"),
        pytest.param((1, 1, 32, 33), -1, id="W_partial=1_garbage_dim_-1"),
        pytest.param((1, 1, 17, 64), -2, id="H_partial_garbage_dim_-2"),
        pytest.param((1, 1, 33, 32), -2, id="H_partial=1_garbage_dim_-2"),
        pytest.param((1, 1, 17, 50), -1, id="both_partial_garbage_dim_-1"),
        pytest.param((1, 1, 17, 50), -2, id="both_partial_garbage_dim_-2"),
    ],
)
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_partial_scaler_masks_padding(device, shape, dim, numeric_stable):
    """Fill the storage padding with a large positive value (99.0) BEFORE
    running softmax. If the partial scaler is correctly masking the
    padded positions, the result is unchanged from random-padded torch.
    If the mask is broken, padded 99.0 would dominate the max and
    skew the softmax distribution massively."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Inject non-zero garbage into the implicit padding. If the partial
    # scaler is dropped (or the kernel reads padded positions as real
    # data) the result would shift dramatically — softmax(99) overflows
    # the rest of the distribution.
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, 99.0)

    ttnn_output = softmax(ttnn_input, dim=dim, numeric_stable=numeric_stable)
    torch_output = ttnn.to_torch(ttnn_output)
    assert tuple(ttnn_output.shape) == shape

    assert_with_pcc(torch_expected, torch_output, 0.999)
    # Tight max-abs gate: if padding leaked in, we'd see >> 1e-2 here.
    max_abs = (torch_expected - torch_output).abs().max().item()
    assert max_abs < 5e-3, f"max abs diff {max_abs} suggests partial-scaler mask leak (shape={shape}, dim={dim})"


# --------------------------------------------------------------------------
# 8. Sum-to-one sanity check — softmax output along the reduce dim must
# sum to ~1 in the valid (non-padded) region. The padded region is allowed
# to contain garbage; we slice it off before checking.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param((1, 1, 32, 50), -1, id="W=50_sum_check"),
        pytest.param((1, 1, 17, 64), -2, id="H=17_sum_check"),
        pytest.param((1, 1, 33, 32), -2, id="H=33_sum_check"),
        pytest.param((1, 1, 17, 50), -1, id="both_dim_-1_sum_check"),
        pytest.param((1, 1, 17, 50), -2, id="both_dim_-2_sum_check"),
    ],
)
def test_softmax_non_aligned_sums_to_one(device, shape, dim):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = softmax(ttnn_input, dim=dim, numeric_stable=True)
    torch_output = ttnn.to_torch(ttnn_output)

    # Sum along the reduce dim over the LOGICAL (non-padded) region. The
    # ttnn-side shape already reflects the logical dims, so to_torch
    # returns a tensor of the right shape.
    sums = torch_output.sum(dim=dim)
    assert torch.allclose(
        sums, torch.ones_like(sums), atol=2e-3, rtol=2e-3
    ), f"softmax non-aligned does not sum to 1 (shape={shape}, dim={dim}): max dev {(sums - 1).abs().max().item():.3e}"


# --------------------------------------------------------------------------
# 9. bf16 + non-aligned spot check — verify the Refinement 2 precision
# matrix composes cleanly with Refinement 4. The verifier note flagged
# bf16 + non_tile_aligned_dim as the canonical EXCLUSIONS candidate; this
# test asserts it passes (i.e., no EXCLUSIONS entry is needed).
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "math_fidelity,fp32_dest_acc_en,precision_name",
    [
        (ttnn.MathFidelity.HiFi2, True, "bf16_hifi2_fp32acc"),
        (ttnn.MathFidelity.HiFi2, False, "bf16_hifi2_bf16acc"),
        (ttnn.MathFidelity.HiFi4, True, "bf16_hifi4_fp32acc"),
        (ttnn.MathFidelity.HiFi4, False, "bf16_hifi4_bf16acc"),
    ],
)
@pytest.mark.parametrize(
    "shape, dim",
    [
        pytest.param((1, 1, 32, 50), -1, id="W_partial_dim_-1"),
        pytest.param((1, 1, 17, 64), -2, id="H_partial_dim_-2"),
    ],
)
def test_softmax_bf16_non_aligned(device, math_fidelity, fp32_dest_acc_en, precision_name, shape, dim):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_expected = torch.softmax(torch_input.to(torch.float32), dim=dim).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    ttnn_output = softmax(ttnn_input, dim=dim, compute_kernel_config=config)
    assert tuple(ttnn_output.shape) == shape
    assert ttnn_output.dtype == ttnn.bfloat16

    torch_output = ttnn.to_torch(ttnn_output)
    # Loose band: lowest bf16 tier in TOLERANCES is PCC >= 0.98.
    assert_with_pcc(torch_expected.to(torch.float32), torch_output.to(torch.float32), 0.98)
