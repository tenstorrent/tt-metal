# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 2 — Input dtype coverage for atan_mean.

Exercises the relaxed dtype contract:
  - float32   : baseline (Phase-0 dtype, unchanged behavior)
  - bfloat16  : 7-bit mantissa; expect PCC >= 0.999, looser max-abs
  - bfloat8_b : block-format 8 bits with shared exponent; expect PCC >= 0.99,
                looser max-abs still

The matmul-mode REDUCE_ROW path unpacks the input on SrcA into the destination
accumulator (fp32 with ``fp32_dest_acc_en=True``), so the per-row accumulation
itself is performed in fp32 regardless of input dtype. Numerical degradation
arises only from quantising the input tile values themselves — not the reduce
path — which matches the verifier's note in op_requirements.md.

Output dtype follows the input dtype (``allocate_tensor_on_device`` uses
``input_tensor.dtype``), so we also assert that contract here.
"""

import pytest
import torch
import ttnn

from ttnn.operations.atan_mean import atan_mean

from tests.ttnn.utils_for_testing import assert_with_pcc


# Per-dtype precision contract. The matmul path keeps accumulation in fp32,
# so the degradation is bounded by atan's Lipschitz=1 property times the
# input-dtype quantisation error.
_DTYPE_TOLERANCES = {
    # dtype:        (pcc_threshold, max_abs_tol)
    ttnn.float32: (0.99995, 1e-2),
    ttnn.bfloat16: (0.999, 3e-2),
    ttnn.bfloat8_b: (0.99, 1e-1),
}


_DTYPE_IDS = {
    ttnn.float32: "float32",
    ttnn.bfloat16: "bfloat16",
    ttnn.bfloat8_b: "bfloat8_b",
}


def _to_device(torch_input: torch.Tensor, device, *, dtype: ttnn.DataType) -> ttnn.Tensor:
    """Build an on-device TILE_LAYOUT tensor of the requested dtype.

    Notes:
      - For bfloat8_b, the host-side torch tensor remains float32; the dtype
        conversion happens inside ``ttnn.from_torch`` during tile-format
        packing.
      - For bfloat16, we cast the host tensor so the quantisation that
        downstream tests observe matches what the operation actually consumes.
    """
    if dtype == ttnn.bfloat16:
        torch_for_device = torch_input.to(torch.bfloat16)
    else:
        torch_for_device = torch_input

    return ttnn.from_torch(
        torch_for_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _expected_reference(torch_input: torch.Tensor, dtype: ttnn.DataType) -> torch.Tensor:
    """Compute the reference in fp32, mirroring the input quantisation level."""
    if dtype == ttnn.bfloat16:
        torch_input = torch_input.to(torch.bfloat16).to(torch.float32)
    # bfloat8_b cannot be represented host-side; round-trip via TTNN packing
    # is not worth simulating — the input retains fp32 precision in the
    # reference, which yields a (mildly) pessimistic error bound. The looser
    # PCC threshold for bfloat8_b accounts for this.
    return torch.atan(torch_input).mean(dim=-1).float()


# Shape set: covers the single-tile minimal, a "tall" regime, a "high-channel"
# regime, and a mid-Wt shape. Drawn from the acceptance + extended shape sets,
# kept small enough that the 3-dtype parametrisation does not blow up runtime.
_SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile_32x32"),
    pytest.param((1, 1, 64, 64), id="small_64x64"),
    pytest.param((1, 1, 128, 64), id="medium_128x64"),
    pytest.param((1, 1, 32, 96), id="Wt3_mid"),
    pytest.param((1, 8, 64, 64), id="batched_1x8x64x64"),
]


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b],
    ids=lambda d: _DTYPE_IDS[d],
)
def test_atan_mean_dtype_correctness(device, shape, dtype):
    """atan_mean handles all supported input dtypes within per-dtype PCC bounds."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = _expected_reference(torch_input, dtype)

    ttnn_input = _to_device(torch_input, device, dtype=dtype)
    ttnn_output = atan_mean(ttnn_input)

    # Output dtype contract: matches input dtype.
    assert ttnn_output.dtype == dtype, f"Output dtype {ttnn_output.dtype} != input dtype {dtype} for shape={shape}."
    assert tuple(ttnn_output.shape) == tuple(expected.shape)

    actual = ttnn.to_torch(ttnn_output).float()

    pcc_threshold, max_abs_tol = _DTYPE_TOLERANCES[dtype]
    max_abs = (actual - expected).abs().max().item()
    assert max_abs <= max_abs_tol, (
        f"[{_DTYPE_IDS[dtype]} shape={shape}] max_abs={max_abs:.6f} > {max_abs_tol}.\n"
        f"  actual.flatten()[:6]   = {actual.flatten()[:6].tolist()}\n"
        f"  expected.flatten()[:6] = {expected.flatten()[:6].tolist()}"
    )
    assert_with_pcc(expected, actual, pcc=pcc_threshold)


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=lambda d: _DTYPE_IDS[d],
)
def test_atan_mean_dtype_zero_input(device, dtype):
    """atan(0) == 0 for every supported dtype; row mean is identically zero."""
    shape = (1, 1, 64, 64)
    torch_input = torch.zeros(shape, dtype=torch.float32)
    ttnn_input = _to_device(torch_input, device, dtype=dtype)

    actual = ttnn.to_torch(atan_mean(ttnn_input)).float()
    assert actual.abs().max().item() < 1e-2, (
        f"Zero-input row-mean is not zero for dtype={_DTYPE_IDS[dtype]}: " f"max-abs={actual.abs().max().item()}"
    )


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
    ids=lambda d: _DTYPE_IDS[d],
)
def test_atan_mean_dtype_saturation(device, dtype):
    """Large positive inputs saturate to atan(∞) = π/2 for every supported dtype."""
    shape = (1, 1, 32, 64)
    # Use 1e4 (representable losslessly in bf16/bf8) so the input encoding
    # itself is not the bottleneck — we're stressing the SFPU saturation path.
    torch_input = torch.full(shape, 1e4, dtype=torch.float32)
    ttnn_input = _to_device(torch_input, device, dtype=dtype)

    actual = ttnn.to_torch(atan_mean(ttnn_input)).float()
    half_pi = 3.141592653589793 / 2.0

    # bf16/bf8 quantise π/2 ≈ 1.5708 with limited mantissa; a 5e-2 budget
    # is comfortably above the worst-case rounding for either format.
    tol = 5e-2 if dtype == ttnn.bfloat8_b else 1e-2
    assert (actual - half_pi).abs().max().item() <= tol, (
        f"[{_DTYPE_IDS[dtype]}] saturation row-mean max-error " f"{(actual - half_pi).abs().max().item()} > {tol}"
    )


def test_atan_mean_rejects_unsupported_dtype(device):
    """uint32 (or any non-{fp32, bf16, bf8}) inputs must still be rejected."""
    # uint32 is not representable as torch.atan input but ttnn supports it as
    # a tensor dtype; we just need any dtype outside the supported set.
    torch.manual_seed(0)
    try:
        bad_input = ttnn.from_torch(
            torch.randint(0, 100, (1, 1, 32, 32), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    except (ValueError, RuntimeError, TypeError):
        # Some ttnn versions reject uint32 + TILE_LAYOUT at construction time.
        # Either rejection point (construction or atan_mean) is acceptable for
        # this negative test.
        return

    with pytest.raises((ValueError, RuntimeError)):
        atan_mean(bad_input)
