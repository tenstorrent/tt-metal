# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Layout matrix for ttnn.operations.layer_norm_rm.layer_norm — Refinement 2.

Refinement 2 adds TILE_LAYOUT to SUPPORTED["layout"] (input tensor) and
drops the two EXCLUSIONS entries `{"affine": "gamma_*", "affine_layout":
TILE_LAYOUT}`. After this refinement the op accepts TILE-layout input
and TILE-layout gamma/beta end-to-end.

Implementation pattern (mirror of softmax-R3): the kernel beneath
layer_norm() is RM-input / RM-output. The entry point wraps any TILE
input tensor with `ttnn.to_layout(x, ROW_MAJOR_LAYOUT)` on the way in,
runs the kernel, and converts the output back to TILE on the way out
when the user supplied TILE. The same wrap applies to gamma/beta.

This test verifies:
1. All four (input_layout × affine_layout) combinations produce the same
   numerical result against the PyTorch reference, with PCC ≥ 0.999.
2. The output's layout mirrors the input's layout (round-trip
   TILE → layer_norm → TILE).
3. Mixed affine layouts (gamma RM, beta TILE; or vice versa) work.
4. The two cells removed from EXCLUSIONS are now accepted (positive
   acceptance contract — they previously raised NotImplementedError).
5. bf16 input × TILE-layout × all affine modes spot-check against the
   bf16 tolerance band (matches the Refinement 1 surface).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.layer_norm_rm import EXCLUSIONS, SUPPORTED, layer_norm


# --------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------
def pytorch_reference(
    input_tensor: torch.Tensor,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    x = input_tensor.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    y = (x - mean) / torch.sqrt(var + epsilon)
    if gamma is not None:
        y = y * gamma.reshape(-1).to(torch.float32)
    if beta is not None:
        y = y + beta.reshape(-1).to(torch.float32)
    return y.to(input_tensor.dtype)


# --------------------------------------------------------------------------
# Shape × affine matrix.
# --------------------------------------------------------------------------
SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 64, 128), id="2x4_tiles"),
    pytest.param((2, 4, 32, 256), id="multi_batch"),
]

AFFINE_MODES = [
    pytest.param("no_affine", id="affine=none"),
    pytest.param("gamma_only", id="affine=gamma_only"),
    pytest.param("gamma_beta", id="affine=gamma_beta"),
]

LAYOUTS = [
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="layout=RM"),
    pytest.param(ttnn.TILE_LAYOUT, id="layout=TILE"),
]


def _make_inputs(shape, affine_mode, input_layout, affine_layout, dtype, device):
    """Build the torch + ttnn tensors for a single cell."""
    torch.manual_seed(42)
    if dtype == ttnn.float32:
        torch_dtype = torch.float32
    elif dtype == ttnn.bfloat16:
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"unsupported dtype {dtype}")

    torch_input = torch.randn(shape, dtype=torch_dtype)
    W = shape[-1]
    torch_gamma = None
    torch_beta = None
    ttnn_gamma = None
    ttnn_beta = None

    if affine_mode in ("gamma_only", "gamma_beta"):
        torch_gamma = (torch.randn((1, 1, 1, W), dtype=torch.float32) * 0.5 + 1.0).to(torch_dtype)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma,
            dtype=dtype,
            layout=affine_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if affine_mode == "gamma_beta":
        torch_beta = (torch.randn((1, 1, 1, W), dtype=torch.float32) * 0.1).to(torch_dtype)
        ttnn_beta = ttnn.from_torch(
            torch_beta,
            dtype=dtype,
            layout=affine_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=input_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta


# --------------------------------------------------------------------------
# 1. Layout cartesian — input layout × affine layout × affine mode × shape.
#    fp32 inputs, PCC ≥ 0.999 (the Phase-0 tier band).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("input_layout", LAYOUTS)
@pytest.mark.parametrize("affine_layout", LAYOUTS)
@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
@pytest.mark.parametrize("shape", SHAPES)
def test_layer_norm_rm_layout_matrix(device, shape, affine_mode, affine_layout, input_layout):
    """Every (input_layout, affine_layout, affine_mode, shape) combo passes PCC."""
    torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta = _make_inputs(
        shape, affine_mode, input_layout, affine_layout, ttnn.float32, device
    )
    torch_expected = pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=1e-5)

    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta)

    # Output layout mirrors input layout (round-trip contract).
    assert list(ttnn_output.shape) == list(shape), f"shape mismatch: got {ttnn_output.shape}, expected {shape}"
    assert (
        ttnn_output.layout == input_layout
    ), f"layout mismatch: got {ttnn_output.layout}, expected {input_layout} (input layout must be preserved)"
    assert ttnn_output.dtype == ttnn.float32

    torch_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_expected, torch_output, 0.999)


# --------------------------------------------------------------------------
# 2. TILE round-trip — explicit verification that TILE → layer_norm → TILE
#    preserves layout. Listed explicitly in op_requirements.md's "Done when".
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_layer_norm_rm_tile_round_trip(device, shape):
    """TILE input → TILE output (single-cell sanity for the 'Done when' criterion)."""
    torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta = _make_inputs(
        shape, "gamma_beta", ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, ttnn.float32, device
    )
    torch_expected = pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=1e-5)

    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta)

    assert ttnn_output.layout == ttnn.TILE_LAYOUT, f"TILE input must yield TILE output, got {ttnn_output.layout}"

    torch_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_expected, torch_output, 0.999)


# --------------------------------------------------------------------------
# 3. Mixed affine layouts — gamma RM × beta TILE (and vice versa).
#    Tests that the wrap handles gamma and beta independently.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gamma_layout,beta_layout",
    [
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, id="gamma=RM,beta=TILE"),
        pytest.param(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, id="gamma=TILE,beta=RM"),
    ],
)
def test_layer_norm_rm_mixed_affine_layout(device, gamma_layout, beta_layout):
    """Gamma and beta with different layouts both get wrapped independently."""
    shape = (1, 1, 64, 128)
    W = shape[-1]
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_gamma = torch.randn((1, 1, 1, W), dtype=torch.float32) * 0.5 + 1.0
    torch_beta = torch.randn((1, 1, 1, W), dtype=torch.float32) * 0.1
    torch_expected = pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=1e-5)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.float32,
        layout=gamma_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_beta = ttnn.from_torch(
        torch_beta,
        dtype=ttnn.float32,
        layout=beta_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta)

    # The output mirrors input layout (RM here, since input was RM).
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT
    torch_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_expected, torch_output, 0.999)


# --------------------------------------------------------------------------
# 4. Positive acceptance — the two cells removed from EXCLUSIONS at R2.
#    Before R2 these raised NotImplementedError; now they must succeed.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "affine_mode",
    [
        pytest.param("gamma_only", id="affine=gamma_only"),
        pytest.param("gamma_beta", id="affine=gamma_beta"),
    ],
)
def test_layer_norm_rm_accepts_tile_affine(device, affine_mode):
    """The (affine=gamma_*, affine_layout=TILE) cells removed from EXCLUSIONS at R2."""
    shape = (1, 1, 32, 64)
    torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta = _make_inputs(
        shape, affine_mode, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT, ttnn.float32, device
    )
    torch_expected = pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=1e-5)

    # Should not raise — these cells were previously in EXCLUSIONS.
    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta)

    torch_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_expected, torch_output, 0.999)


# --------------------------------------------------------------------------
# 5. Drift signal — confirm SUPPORTED and EXCLUSIONS reflect R2.
#    Catches op-file regressions where someone re-adds the lifted cells
#    or drops the bf8b structural-gap entry.
# --------------------------------------------------------------------------
def test_layer_norm_rm_supported_reflects_r2():
    """SUPPORTED['layout'] contains both layouts; bf8b in EXCLUSIONS; lifted pairs gone."""
    assert (
        ttnn.TILE_LAYOUT in SUPPORTED["layout"]
    ), f"SUPPORTED['layout']={SUPPORTED['layout']} missing TILE_LAYOUT after R2"
    assert (
        ttnn.ROW_MAJOR_LAYOUT in SUPPORTED["layout"]
    ), f"SUPPORTED['layout']={SUPPORTED['layout']} missing ROW_MAJOR_LAYOUT (non-regression)"
    # The two Phase-0 (affine=gamma_*, affine_layout=TILE) pairs must NOT appear.
    forbidden = [
        {"affine": "gamma_only", "affine_layout": ttnn.TILE_LAYOUT},
        {"affine": "gamma_beta", "affine_layout": ttnn.TILE_LAYOUT},
    ]
    for entry in forbidden:
        assert entry not in EXCLUSIONS, f"EXCLUSIONS still contains {entry} after R2 — should have been removed"
    # bf8b joined EXCLUSIONS: R2 made it reachable (TILE-only) but the
    # entry-point's TILE→RM wrap can't preserve bf8b (bf8b in RM is INVALID).
    assert {"precision": "bf8b_hifi4_bf16acc"} in EXCLUSIONS, (
        "EXCLUSIONS missing the bf8b structural-gap entry — bf8b cannot round-trip through "
        "the TILE→RM entry-point wrap because bf8b in RM is INVALID."
    )


# --------------------------------------------------------------------------
# 6. bf16 × TILE spot check — composes R1 (bf16) and R2 (TILE) cleanly.
#    Tolerance follows helpers.TOLERANCES bf16_hifi4_fp32acc band (PCC ≥ 0.995).
# --------------------------------------------------------------------------
@pytest.mark.parametrize("input_layout", LAYOUTS)
@pytest.mark.parametrize("affine_layout", LAYOUTS)
@pytest.mark.parametrize("affine_mode", ["no_affine", "gamma_beta"])
def test_layer_norm_rm_bf16_tile(device, input_layout, affine_layout, affine_mode):
    """bf16 input + bf16 affine across all layout combos."""
    shape = (1, 1, 64, 128)
    torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta = _make_inputs(
        shape, affine_mode, input_layout, affine_layout, ttnn.bfloat16, device
    )
    torch_expected = pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=1e-5)

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta, compute_kernel_config=cfg)

    assert ttnn_output.layout == input_layout
    assert ttnn_output.dtype == ttnn.bfloat16

    torch_output = ttnn.to_torch(ttnn_output)
    # bf16_hifi4_fp32acc band: PCC ≥ 0.995 per helpers.TOLERANCES.
    assert_with_pcc(torch_expected.float(), torch_output.float(), 0.995)
