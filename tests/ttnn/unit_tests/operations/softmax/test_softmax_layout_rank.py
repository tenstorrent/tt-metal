# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layout + rank-canonicalisation matrix for ttnn.operations.softmax.softmax —
Refinement 3.

This test exercises the two new entry-point capabilities added in this
refinement:

1. ``SUPPORTED["layout"]`` now contains both ``ttnn.TILE_LAYOUT`` and
   ``ttnn.ROW_MAJOR_LAYOUT``. The entry-point converts RM → TILE on the
   way in (via ``ttnn.to_layout``) and TILE → RM on the way out,
   preserving the user's layout end-to-end. The kernel itself still
   operates in TILE.

2. ``SUPPORTED["rank"]`` now contains 2, 3, and 4. Rank-2 ``(H, W)`` and
   rank-3 ``(B, H, W)`` inputs are canonicalised to rank-4 with leading
   ``1`` dims via ``ttnn.unsqueeze_to_4D``; the output is reshaped back
   to the original rank on exit. ``dim`` is passed as a negative offset
   (``-1`` or ``-2``) so the leading-1 unsqueeze does not shift its
   meaning.

The matrix mirrors the structure from ``/memory-layouts``: a small set
of tile-aligned shapes per rank crossed with the four supported (layout,
output-layout) pairs (we always preserve layout, so the relevant pairs
are TILE→TILE and RM→RM) and both reduce dims. Precision is fp32 + HiFi4
+ fp32_dest_acc to keep this test focused on layout/rank — precision
sweeps live in ``test_softmax_precision_matrix.py``.

Block formats (``bfloat8_b``, ``bfloat4_b``) are not in the softmax
SUPPORTED["precision"] axis at all, so the ``/memory-layouts`` §5 skip
guard is not needed here.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.softmax import softmax


# --------------------------------------------------------------------------
# Shape sets — one per rank. All shapes are tile-aligned in H and W
# (alignment is Refinement 4's axis; we stay inside SUPPORTED here).
# --------------------------------------------------------------------------
RANK2_SHAPES = [
    pytest.param((32, 32), id="r2_32x32"),
    pytest.param((32, 64), id="r2_32x64"),
    pytest.param((64, 128), id="r2_64x128"),
    pytest.param((128, 512), id="r2_128x512"),
]

RANK3_SHAPES = [
    pytest.param((1, 32, 128), id="r3_1x32x128"),
    pytest.param((4, 128, 512), id="r3_4x128x512"),
    pytest.param((2, 64, 64), id="r3_2x64x64"),
]

RANK4_SHAPES = [
    pytest.param((1, 1, 32, 32), id="r4_1x1x32x32"),
    pytest.param((2, 4, 32, 256), id="r4_2x4x32x256"),
    pytest.param((1, 2, 128, 128), id="r4_1x2x128x128"),
]


# --------------------------------------------------------------------------
# Layouts. We exercise both supported layouts; output layout always
# matches input layout (the entry-point contract).
# --------------------------------------------------------------------------
LAYOUTS = [
    pytest.param(ttnn.TILE_LAYOUT, id="tile"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="row_major"),
]


# --------------------------------------------------------------------------
# Rank-2 matrix (H, W). dim=-1 reduces W; dim=-2 reduces H.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", RANK2_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_rank2(device, shape, layout, dim, numeric_stable):
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

    # Shape and layout must match the user-visible (rank-2) input.
    assert tuple(ttnn_output.shape) == shape
    assert ttnn_output.layout == layout
    assert ttnn_output.dtype == ttnn.float32

    torch_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_expected, torch_output, 0.999)


# --------------------------------------------------------------------------
# Rank-3 matrix (B, H, W).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", RANK3_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_rank3(device, shape, layout, dim, numeric_stable):
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
    assert ttnn_output.dtype == ttnn.float32

    torch_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_expected, torch_output, 0.999)


# --------------------------------------------------------------------------
# Rank-4 matrix — covers ROW_MAJOR on the existing rank, since rank-4
# was Phase 0's SUPPORTED rank and the only thing added here is RM.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", RANK4_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_rank4_layout(device, shape, layout, dim, numeric_stable):
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
    assert ttnn_output.dtype == ttnn.float32

    torch_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_expected, torch_output, 0.999)


# --------------------------------------------------------------------------
# Negative gates — verify that ranks outside SUPPORTED still get rejected
# and that the rank-canonicalisation didn't accidentally widen the surface.
# --------------------------------------------------------------------------
VALIDATION_ERRORS = (NotImplementedError, ValueError, RuntimeError)


def test_softmax_rejects_rank5(device):
    """Rank-5 is still outside SUPPORTED["rank"] — the canonicaliser only
    handles rank ∈ {2, 3, 4}. A rank-5 tensor must be rejected by
    validate() before any reshape happens."""
    torch_input = torch.randn((1, 1, 1, 32, 32), dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(VALIDATION_ERRORS):
        softmax(ttnn_input, dim=-1)


# --------------------------------------------------------------------------
# Cross-precision spot-check — bf16 + ROW_MAJOR + a rank-3 shape exercises
# Refinement 2 × Refinement 3 in one cell. This is the "no structural
# EXCLUSIONS surface" assertion from the verifier notes.
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
@pytest.mark.parametrize("layout", LAYOUTS)
def test_softmax_bf16_layout_rank3_spotcheck(device, math_fidelity, fp32_dest_acc_en, precision_name, layout):
    """bf16 input + ROW_MAJOR layout + rank-3 shape: one cell per bf16 precision
    mode. PCC bands are loose to match bf16; the goal is to assert no
    structural EXCLUSIONS leak in at the Refinement 2 × Refinement 3
    intersection."""
    torch.manual_seed(0)
    shape = (2, 64, 128)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_expected = torch.softmax(torch_input.to(torch.float32), dim=-1).to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    ttnn_output = softmax(ttnn_input, dim=-1, compute_kernel_config=config)
    assert tuple(ttnn_output.shape) == shape
    assert ttnn_output.layout == layout
    assert ttnn_output.dtype == ttnn.bfloat16

    torch_output = ttnn.to_torch(ttnn_output)
    # bf16 + bf16acc tier (lowest precision in SUPPORTED) — use the helpers
    # band's lower bound (PCC ≥ 0.98) so this test passes for all four bf16 modes.
    assert_with_pcc(torch_expected.to(torch.float32), torch_output.to(torch.float32), 0.98)
