# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for atan_mean — fused atan-then-row-mean along the last dim.

Math under test:
    atan_mean(x) == torch.atan(x).mean(dim=-1)

The operation is a single fused TTNN kernel that streams input tiles, applies
SFPU atan, and reduces the row inside the same program — no intermediate
atan(x) tensor is materialised to DRAM. The Python entry point dispatches
exactly one `ttnn.generic_op` call.

This test is the immutable spec. The implementer must NOT modify this file.
If a parametrized case is impossible under the Phase-0 constraints
(fp32, TILE_LAYOUT, rank == 4, H % 32 == 0, W % 32 == 0), the implementer
should fix the kernel rather than relax the test.

Coverage:
- The "tall" shape regime (small N, C; large H; small W) — exercises many
  output rows with short per-row reductions.
- The "high-channel" shape regime (large N or C; small H, W) — exercises many
  outer batches with short reductions distributed across the outer axis.
- Single-tile minimal case.
- Both call styles (positional and keyword).
- Numerical correctness against `torch.atan(input).mean(dim=-1)` over random
  N(0, 1) inputs — atan is well-conditioned everywhere, so no domain masking
  is needed.
- Negative-path validation: invalid dtype, layout, rank, non-tile-aligned
  H/W must be rejected by the Python validator before any kernel launches.
"""

import pytest
import torch
import ttnn

from ttnn.operations.atan_mean import atan_mean

from tests.ttnn.utils_for_testing import assert_with_pcc


# Phase-0 tolerances per the task spec:
#   PCC >= 0.9995
#   max abs err <= 1e-2
#   rel RMS err <= 1e-3
# atan is Lipschitz with constant <= 1 over R, and the row mean is exact apart
# from the bf16 reduce-scaler quantisation (1/W is bit-exact in bf16 for the
# powers-of-two W values exercised here). These tolerances reflect that — the
# only real error source is the SFPU atan_tile approximation plus matmul-path
# reduce in fp32 destination accumulation.
PCC_THRESHOLD = 0.9995
MAX_ABS_TOL = 1e-2
RMS_REL_TOL = 1e-3


def _torch_reference(x: torch.Tensor) -> torch.Tensor:
    """torch.atan(x).mean(dim=-1) computed in fp32, matching Phase-0 dtype."""
    return torch.atan(x).mean(dim=-1)


def _compute_max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().max().item()


def _compute_rms_rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    err = (actual - expected).float()
    rms = torch.sqrt((err * err).mean()).item()
    # Relative to the expected RMS magnitude (avoids divide-by-zero when
    # the reference is all-zero; for N(0, 1) inputs `atan(x).mean` is not
    # identically zero so this denominator is well-defined).
    expected_rms = torch.sqrt((expected.float() ** 2).mean()).item()
    if expected_rms < 1e-12:
        return rms  # fall back to absolute RMS if reference magnitude is ~0
    return rms / expected_rms


# -----------------------------------------------------------------------------
# Positive cases — bulk correctness across both shape regimes
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        # --- Minimal case (single tile per (n, c) plane) ---
        pytest.param((1, 1, 32, 32), id="single_tile_32x32"),
        # --- "Tall" regime: small NC, large H, small W ---
        pytest.param((1, 1, 2048, 64), id="tall_2048x64"),
        pytest.param((1, 1, 1024, 64), id="tall_1024x64"),
        pytest.param((1, 1, 2048, 32), id="tall_2048x32"),
        pytest.param((1, 1, 1024, 32), id="tall_1024x32"),
        # --- "High-channel" regime: large NC, small HxW ---
        pytest.param((1, 256, 64, 64), id="high_channel_C256_64x64"),
        pytest.param((256, 1, 64, 64), id="high_channel_N256_64x64"),
        pytest.param((1, 128, 128, 128), id="high_channel_C128_128x128"),
        pytest.param((128, 1, 128, 128), id="high_channel_N128_128x128"),
    ],
)
def test_atan_mean_correctness(device, shape):
    """
    atan_mean(x) matches torch.atan(x).mean(dim=-1) within the Phase-0
    tolerances across both shape regimes.
    """
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = atan_mean(ttnn_input)

    # --- Shape / dtype / layout sanity ---
    assert tuple(ttnn_output.shape) == tuple(torch_expected.shape), (
        f"Output shape {tuple(ttnn_output.shape)} != expected {tuple(torch_expected.shape)} " f"for input shape {shape}"
    )
    assert ttnn_output.dtype == ttnn.float32, f"Output dtype {ttnn_output.dtype} != float32"

    actual = ttnn.to_torch(ttnn_output).float()
    expected = torch_expected.float()

    # --- Numerical match: PCC, max-abs, rel-RMS ---
    max_abs = _compute_max_abs(actual, expected)
    rms_rel = _compute_rms_rel(actual, expected)

    assert max_abs <= MAX_ABS_TOL, (
        f"Max abs error {max_abs:.6f} > {MAX_ABS_TOL} for shape={shape}.\n"
        f"  actual.flatten()[:6]   = {actual.flatten()[:6].tolist()}\n"
        f"  expected.flatten()[:6] = {expected.flatten()[:6].tolist()}"
    )
    assert rms_rel <= RMS_REL_TOL, (
        f"Relative RMS error {rms_rel:.6f} > {RMS_REL_TOL} for shape={shape}.\n"
        f"  actual.flatten()[:6]   = {actual.flatten()[:6].tolist()}\n"
        f"  expected.flatten()[:6] = {expected.flatten()[:6].tolist()}"
    )

    # PCC last — gives the most readable diagnostic when correlation is broken.
    assert_with_pcc(expected, actual, pcc=PCC_THRESHOLD)


# -----------------------------------------------------------------------------
# Call-style coverage — both positional and keyword forms must work
# -----------------------------------------------------------------------------


def test_atan_mean_positional_call(device):
    """atan_mean(t) — positional argument call style works."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = atan_mean(ttnn_input)  # positional
    actual = ttnn.to_torch(ttnn_output).float()
    expected = torch_expected.float()

    assert tuple(actual.shape) == tuple(expected.shape)
    assert_with_pcc(expected, actual, pcc=PCC_THRESHOLD)


def test_atan_mean_keyword_call(device):
    """atan_mean(input_tensor=t) — keyword argument call style works."""
    torch.manual_seed(42)
    shape = (1, 1, 32, 32)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = atan_mean(input_tensor=ttnn_input)  # keyword
    actual = ttnn.to_torch(ttnn_output).float()
    expected = torch_expected.float()

    assert tuple(actual.shape) == tuple(expected.shape)
    assert_with_pcc(expected, actual, pcc=PCC_THRESHOLD)


# -----------------------------------------------------------------------------
# Negative cases (Python-side validation) — must raise before any kernel runs
# -----------------------------------------------------------------------------


def _make_tensor(device, shape, *, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    """Helper to build on-device tensors with controlled dtype / layout."""
    torch.manual_seed(42)
    torch_dtype = {
        ttnn.float32: torch.float32,
        ttnn.bfloat16: torch.bfloat16,
    }.get(dtype, torch.float32)
    return ttnn.from_torch(
        torch.randn(shape, dtype=torch_dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_atan_mean_accepts_bf16_dtype(device):
    """
    Refinement 2: bfloat16 is an accepted input dtype. The validator must
    pass it through and the op must produce a numerically-reasonable result.
    """
    bf16_input = _make_tensor(device, (1, 1, 32, 32), dtype=ttnn.bfloat16)
    out = atan_mean(bf16_input)
    assert out.dtype == ttnn.bfloat16
    assert tuple(out.shape) == (1, 1, 32)


def test_atan_mean_rejects_row_major_layout(device):
    """TILE_LAYOUT only. ROW_MAJOR_LAYOUT must be rejected."""
    rm_input = _make_tensor(device, (1, 1, 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
    with pytest.raises((ValueError, RuntimeError)):
        atan_mean(rm_input)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="rank_2"),
        pytest.param((1, 32, 32), id="rank_3"),
        pytest.param((1, 1, 1, 32, 32), id="rank_5"),
    ],
)
def test_atan_mean_rejects_wrong_rank(device, shape):
    """Phase 0 fixes rank to 4. Other ranks must be rejected."""
    try:
        bad_input = ttnn.from_torch(
            torch.randn(shape, dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    except (ValueError, RuntimeError):
        # If ttnn.from_torch itself rejects the shape, that's also a valid
        # precondition enforcement point.
        return
    with pytest.raises((ValueError, RuntimeError)):
        atan_mean(bad_input)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 30, 32), id="H_not_tile_aligned"),
        pytest.param((1, 1, 32, 30), id="W_not_tile_aligned"),
        pytest.param((1, 1, 64, 48), id="W_not_tile_aligned_48"),
        pytest.param((1, 1, 48, 64), id="H_not_tile_aligned_48"),
    ],
)
def test_atan_mean_rejects_non_tile_aligned(device, shape):
    """
    Phase 0 requires H and W divisible by 32. Non-aligned inputs must be
    rejected by the Python validator — the validator must inspect the
    underlying logical shape (ttnn.from_torch pads to a tile internally).
    """
    bad_input = _make_tensor(device, shape)
    with pytest.raises((ValueError, RuntimeError)):
        atan_mean(bad_input)
