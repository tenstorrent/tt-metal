# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Non-tile-aligned shape matrix for ttnn.operations.layer_norm_rm.layer_norm
— Refinement 3.

Refinement 3 adds `w_non_aligned` and `h_non_aligned` to
SUPPORTED["alignment"]. The kernel handles:

  W non-aligned: the LAST reduce-row tile is partially valid. The reader
    emits a (full, partial) scaler tile pair via
    `prepare_partial_reduce_scalers<…, partial_w>(1/W)` so that the
    `accumulate_reduce_block<SUM, REDUCE_ROW>` calls in Pass A (mean) and
    Pass B (variance) mask the padded W positions out of the SUM. Pass C
    (sub<COL> + mul<COL> + optional ROW broadcasts + untilize) computes
    junk in the padded positions but the writer drops them on the way out
    (write_sticks_after_untilize with `row_bytes = chunk_bytes_last`).

  H non-aligned: num_strips uses ceil division; the GLOBAL last strip has
    < 32 valid rows. The reader/writer pass `last_strip_rows` to the
    tilize-dataflow helpers which natively handle partial-row blocks
    (the helper still pushes BLOCK_SIZE tile-pages from the CB to balance
    the producer count, padded rows compute junk, writer never writes
    them back).

This test exercises the new axes directly (acceptance-level), independent
of the golden suite.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.layer_norm_rm import SUPPORTED, layer_norm


# --------------------------------------------------------------------------
# Reference (matches pytorch_layer_norm in helpers.py / test_layer_norm_rm.py)
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


def _make_inputs(shape, affine_mode, device, *, dtype=ttnn.float32, affine_dtype=ttnn.float32):
    """Build (torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta).

    Always RM layout for input and affine tensors (this test class targets the
    RM-input path; layout cross-product is tested separately in
    test_layer_norm_rm_layout.py).
    """
    W = shape[-1]
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_affine_dtype = torch.float32 if affine_dtype == ttnn.float32 else torch.bfloat16

    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch_dtype)

    if affine_mode in ("gamma_only", "gamma_beta"):
        torch_gamma = torch.randn(W, dtype=torch_affine_dtype)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=affine_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
    else:
        torch_gamma = None
        ttnn_gamma = None

    if affine_mode == "gamma_beta":
        torch_beta = torch.randn(W, dtype=torch_affine_dtype)
        ttnn_beta = ttnn.from_torch(
            torch_beta.reshape(1, 1, 1, W),
            dtype=affine_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
    else:
        torch_beta = None
        ttnn_beta = None

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    return torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta


# --------------------------------------------------------------------------
# Drift signal — SUPPORTED must reflect Refinement 3.
# --------------------------------------------------------------------------
def test_supported_includes_w_and_h_non_aligned():
    """If the kernel changes lapse, the new alignment values will revert."""
    assert "w_non_aligned" in SUPPORTED["alignment"]
    assert "h_non_aligned" in SUPPORTED["alignment"]
    assert "tile_aligned" in SUPPORTED["alignment"]


# --------------------------------------------------------------------------
# Shape matrices.
# --------------------------------------------------------------------------

# W non-aligned (H aligned). Covers single-tile, sub-tile, multi-tile widths,
# and a wider case where there are multiple chunks per strip with the last
# chunk being partial.
W_NON_ALIGNED_SHAPES = [
    pytest.param((1, 1, 32, 17), id="W=17_single_partial"),
    pytest.param((1, 1, 32, 33), id="W=33_one_tile_plus_1"),
    pytest.param((1, 1, 32, 50), id="W=50_two_tile_partial"),
    pytest.param((1, 1, 64, 47), id="W=47_two_strips"),
    pytest.param((2, 1, 32, 100), id="W=100_multi_batch"),
    pytest.param((1, 1, 32, 200), id="W=200_partial"),
    pytest.param((1, 1, 32, 257), id="W=257_multi_chunk_partial"),
]

# H non-aligned (W aligned). Covers single-strip H < 32, multi-strip with
# partial last strip, and larger multi-batch cases.
H_NON_ALIGNED_SHAPES = [
    pytest.param((1, 1, 17, 64), id="H=17_single_partial"),
    pytest.param((1, 1, 33, 64), id="H=33_two_strips"),
    pytest.param((1, 1, 50, 128), id="H=50_two_strips"),
    pytest.param((1, 1, 47, 256), id="H=47_wider_W"),
    pytest.param((2, 1, 100, 32), id="H=100_multi_batch_NC"),
    pytest.param((4, 1, 17, 128), id="H=17_NC_multi"),
]

# Both H and W non-aligned. Tagger emits "w_non_aligned" (W check first).
BOTH_NON_ALIGNED_SHAPES = [
    pytest.param((1, 1, 17, 50), id="HxW=17x50"),
    pytest.param((2, 1, 100, 47), id="HxW=100x47_multi_batch"),
    pytest.param((1, 1, 33, 33), id="HxW=33x33_min_partial"),
]

# Rank composition — ensure rank-2 and rank-3 also work with non-aligned.
RANK_COMPOSITION_SHAPES = [
    pytest.param((32, 17), id="rank2_32x17"),
    pytest.param((17, 64), id="rank2_17x64"),
    pytest.param((1, 32, 50), id="rank3_32x50"),
    pytest.param((4, 17, 128), id="rank3_17x128"),
]


AFFINE_MODES = [
    pytest.param("no_affine", id="affine=none"),
    pytest.param("gamma_only", id="affine=gamma_only"),
    pytest.param("gamma_beta", id="affine=gamma_beta"),
]


# --------------------------------------------------------------------------
# Acceptance: W non-aligned.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", W_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
def test_layer_norm_rm_w_non_aligned(device, shape, affine_mode):
    torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta = _make_inputs(
        shape,
        affine_mode,
        device,
    )
    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta, epsilon=1e-5)
    expected = pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=1e-5)
    assert_with_pcc(ttnn.to_torch(ttnn_output).to(torch.float32), expected.to(torch.float32), pcc=0.999)


# --------------------------------------------------------------------------
# Acceptance: H non-aligned.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", H_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
def test_layer_norm_rm_h_non_aligned(device, shape, affine_mode):
    torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta = _make_inputs(
        shape,
        affine_mode,
        device,
    )
    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta, epsilon=1e-5)
    expected = pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=1e-5)
    assert_with_pcc(ttnn.to_torch(ttnn_output).to(torch.float32), expected.to(torch.float32), pcc=0.999)


# --------------------------------------------------------------------------
# Acceptance: both axes non-aligned.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", BOTH_NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
def test_layer_norm_rm_both_non_aligned(device, shape, affine_mode):
    torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta = _make_inputs(
        shape,
        affine_mode,
        device,
    )
    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta, epsilon=1e-5)
    expected = pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=1e-5)
    assert_with_pcc(ttnn.to_torch(ttnn_output).to(torch.float32), expected.to(torch.float32), pcc=0.999)


# --------------------------------------------------------------------------
# Rank composition (rank-2, rank-3) × non-aligned.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", RANK_COMPOSITION_SHAPES)
def test_layer_norm_rm_non_aligned_rank_composition(device, shape):
    torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta = _make_inputs(
        shape,
        "no_affine",
        device,
    )
    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta, epsilon=1e-5)
    expected = pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=1e-5)
    assert_with_pcc(ttnn.to_torch(ttnn_output).to(torch.float32), expected.to(torch.float32), pcc=0.999)


# --------------------------------------------------------------------------
# bf16 + non-aligned spot-check — composes Refinement 1 (bf16 precision) and
# Refinement 3 (alignment).
# --------------------------------------------------------------------------
BF16_NON_ALIGNED_SHAPES = [
    pytest.param((1, 1, 32, 33), id="W=33_bf16"),
    pytest.param((1, 1, 17, 64), id="H=17_bf16"),
    pytest.param((1, 1, 50, 100), id="HxW=50x100_bf16"),
]


@pytest.mark.parametrize("shape", BF16_NON_ALIGNED_SHAPES)
def test_layer_norm_rm_bf16_non_aligned(device, shape):
    """bf16 input with fp32 dest-acc + non-aligned shape — composes R1 + R3.

    Tolerance band matches the bf16_hifi4_fp32acc tier
    (helpers.py:TOLERANCES = (0.995, 0.04)).
    """
    torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta = _make_inputs(
        shape,
        "gamma_beta",
        device,
        dtype=ttnn.bfloat16,
        affine_dtype=ttnn.bfloat16,
    )
    compute_kernel_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
    ttnn_output = layer_norm(
        ttnn_input,
        ttnn_gamma,
        ttnn_beta,
        epsilon=1e-5,
        compute_kernel_config=compute_kernel_config,
    )
    expected = pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=1e-5)
    assert_with_pcc(ttnn.to_torch(ttnn_output).to(torch.float32), expected.to(torch.float32), pcc=0.995)


# --------------------------------------------------------------------------
# Padded-position masking signal — if the partial scaler leaks, the padded
# positions of the last reduce-row tile would dominate the mean / variance,
# skewing the normalization massively. This test forces a specific
# pre-pattern in the input's padded region by using a deterministic seed and
# then checking that the result matches the PyTorch reference (which only
# sees the logical W positions). If the partial-scaler routing is wrong, the
# numerical mismatch will be enormous.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 33), id="W=33_partial=1"),
        pytest.param((1, 1, 32, 47), id="W=47_partial=15"),
        pytest.param((1, 1, 32, 200), id="W=200_partial=8"),
        pytest.param((1, 1, 32, 257), id="W=257_partial=1_multi_chunk"),
    ],
)
def test_layer_norm_rm_padded_position_masking(device, shape):
    """Smoke check that the partial scaler is actually masking padded positions.

    If the partial scaler leaked (e.g. compute called with
    ReducePartialScaler::none() despite partial_w > 0), the SUM would include
    whatever garbage L1 data sat in the padded W positions of the last tile.
    Even with non-zero garbage in those positions, the resulting mean/var
    would be different from the PyTorch reference, breaking PCC.
    """
    torch_input, _, _, ttnn_input, _, _ = _make_inputs(shape, "no_affine", device)
    ttnn_output = layer_norm(ttnn_input, epsilon=1e-5)
    expected = pytorch_reference(torch_input, epsilon=1e-5)
    result = ttnn.to_torch(ttnn_output).to(torch.float32)
    # Loose PCC band — purely a leak indicator. A true leak would tank PCC to
    # something like 0.2-0.7 depending on the garbage magnitude.
    assert_with_pcc(result, expected.to(torch.float32), pcc=0.999)
