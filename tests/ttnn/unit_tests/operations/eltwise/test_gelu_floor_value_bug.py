# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
GELU Precision Bug Reproducer - Multiple Regions

This test demonstrates THREE precision bugs in ttnn.gelu() accurate mode:

1. DEEP NEGATIVE TAIL (x < -5.5): Max ULP = 32,767
   - Hardware returns 0.0 for x < -5.5
   - Exact GELU has tiny negative values (e.g., GELU(-13.5) ≈ -1e-40)
   - Cause: Threshold at -5.5 is too aggressive

2. NEAR-ZERO (|x| < ~1e-4): Max ULP = 14,276
   - Hardware returns floor value 2.98e-05 (Chebyshev c0)
   - Expected: GELU(x) ≈ 0.5*x for tiny x
   - Cause: c0 coefficient dominates for tiny inputs

3. TRANSITION REGION (-5.5 to ~-4.0): Max ULP = 1,475
   - Polynomial poorly fitted near the -5.5 boundary
   - Errors range from 100-1500 ULP

Source: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h

Run: pytest tests/ttnn/unit_tests/operations/eltwise/test_gelu_floor_value_bug.py -v -s
"""

import struct
import math
import pytest
import torch
import ttnn
import numpy as np
from loguru import logger


def float_to_bf16_bits(f: float) -> int:
    """Convert float to BFloat16 bit representation."""
    f32_bits = struct.unpack(">I", struct.pack(">f", f))[0]
    return f32_bits >> 16


def bf16_bits_to_float(bits: int) -> float:
    """Convert BFloat16 bits to float."""
    f32_bits = bits << 16
    return struct.unpack(">f", struct.pack(">I", f32_bits))[0]


def ulp_distance_bf16(a: float, b: float) -> int:
    """Calculate ULP distance between two values in BFloat16."""
    a_bits = float_to_bf16_bits(a)
    b_bits = float_to_bf16_bits(b)

    if (a_bits >> 15) != (b_bits >> 15):
        if a_bits >> 15:
            a_bits = 0x8000 - (a_bits & 0x7FFF) if a_bits != 0x8000 else 0
        if b_bits >> 15:
            b_bits = 0x8000 - (b_bits & 0x7FFF) if b_bits != 0x8000 else 0
        return a_bits + b_bits

    return abs(int(a_bits) - int(b_bits))


def gelu_exact(x: float) -> float:
    """Exact GELU using erfc to avoid catastrophic cancellation for negative x."""
    if x >= 0:
        return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
    else:
        return 0.5 * x * math.erfc(-x / math.sqrt(2.0))


# Constants
CHEBYSHEV_C0 = 2.98325768482e-05
FLOOR_VALUE_BF16 = bf16_bits_to_float(float_to_bf16_bits(CHEBYSHEV_C0))


# =============================================================================
# Region 1: Deep Negative Tail (x < -5.5) - WORST BUG
# =============================================================================


class TestGeluDeepNegativeTailBug:
    """
    Tests for the deep negative tail bug where hardware returns 0.0 for x < -5.5.

    This is the WORST bug with Max ULP = 32,767 (maximum possible for BF16).
    """

    @pytest.mark.parametrize(
        "input_value,expected_ulp_min",
        [
            (-13.5, 32000),  # Max ULP at saturation boundary
            (-12.0, 29000),
            (-10.0, 24000),
            (-8.0, 22000),
            (-6.0, 20000),
            (-5.5625, 19000),  # Just below threshold
        ],
    )
    def test_deep_negative_returns_zero(self, device, input_value, expected_ulp_min):
        """
        Verifies that deep negative inputs produce catastrophic ULP errors.
        Hardware returns 0.0 but exact GELU has tiny negative values.
        """
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        actual = ttnn.to_torch(tt_result).item()

        expected = gelu_exact(input_value)
        ulp_error = ulp_distance_bf16(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.2e}, actual={actual:.2e}, ULP={ulp_error:,}")

        # Verify the bug exists
        assert actual == 0.0, f"Expected 0.0 for x={input_value}, got {actual}"
        assert ulp_error >= expected_ulp_min, f"Expected ULP >= {expected_ulp_min}, got {ulp_error}"


# =============================================================================
# Region 2: Near-Zero Floor Value Bug
# =============================================================================


class TestGeluNearZeroFloorValueBug:
    """
    Tests for the near-zero floor value bug where hardware returns 2.98e-05
    for all tiny inputs instead of 0.5*x.

    Max ULP = 14,276 for input 1e-38.
    """

    @pytest.mark.parametrize(
        "input_value",
        [1e-38, 1e-35, 1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-8],
    )
    def test_tiny_inputs_return_floor_value(self, device, input_value):
        """
        Verifies that tiny positive inputs return the floor value instead of 0.5*x.
        """
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        actual = ttnn.to_torch(tt_result).item()

        expected = 0.5 * input_value
        ulp_error = ulp_distance_bf16(actual, expected)

        logger.info(f"x={input_value:.0e}: expected={expected:.2e}, actual={actual:.2e}, ULP={ulp_error:,}")

        # Verify the bug: actual should be floor value, not expected
        assert (
            abs(actual - FLOOR_VALUE_BF16) < 1e-7
        ), f"Expected floor value {FLOOR_VALUE_BF16:.6e} for x={input_value:.0e}, got {actual:.6e}"
        assert ulp_error > 1000, f"Expected ULP > 1000, got {ulp_error}"


# =============================================================================
# Region 3: Transition Region (-5.5 to ~-4.0)
# =============================================================================


class TestGeluTransitionRegionBug:
    """
    Tests for the transition region bug near the -5.5 threshold.

    The polynomial is poorly fitted in this region, causing ULP errors of 100-1500.
    """

    @pytest.mark.parametrize(
        "input_value",
        [-5.5, -5.4375, -5.375, -5.25, -5.0, -4.75, -4.5, -4.25, -4.0],
    )
    def test_transition_region_errors(self, device, input_value):
        """
        Verifies elevated ULP errors in the transition region.
        """
        torch_input = torch.tensor([[input_value]], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        actual = ttnn.to_torch(tt_result).item()

        expected = gelu_exact(input_value)
        ulp_error = ulp_distance_bf16(actual, expected)

        logger.info(f"x={input_value}: expected={expected:.2e}, actual={actual:.2e}, ULP={ulp_error:,}")

        # Log for analysis - actual assertion depends on expected behavior
        if ulp_error > 100:
            logger.warning(f"High ULP error ({ulp_error}) at x={input_value}")


# =============================================================================
# Comprehensive Sweep and Summary
# =============================================================================


def test_gelu_ulp_summary(device):
    """
    Generates a comprehensive summary of GELU ULP errors across all regions.
    """
    logger.info("")
    logger.info("=" * 100)
    logger.info("GELU PRECISION BUG SUMMARY - THREE PROBLEMATIC REGIONS")
    logger.info("=" * 100)

    # Region 1: Deep Negative Tail
    logger.info("")
    logger.info("REGION 1: DEEP NEGATIVE TAIL (x < -5.5)")
    logger.info("-" * 80)
    logger.info("Cause: Hardware returns 0.0 for x < -5.5, but exact GELU has tiny negative values")
    logger.info("")
    logger.info(f"{'Value':>10} | {'Expected':>14} | {'Actual':>14} | {'ULP Error':>12}")
    logger.info("-" * 60)

    deep_neg_values = [-13.5, -12.0, -10.0, -8.0, -6.0, -5.5625]
    max_ulp_region1 = 0

    for val in deep_neg_values:
        torch_input = torch.tensor([[val]], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        actual = ttnn.to_torch(tt_result).item()
        expected = gelu_exact(val)
        ulp = ulp_distance_bf16(actual, expected)
        max_ulp_region1 = max(max_ulp_region1, ulp)
        logger.info(f"{val:10.4f} | {expected:14.2e} | {actual:14.2e} | {ulp:12,}")

    logger.info(f"\nMax ULP in Region 1: {max_ulp_region1:,}")

    # Region 2: Near-Zero
    logger.info("")
    logger.info("REGION 2: NEAR-ZERO (|x| < ~1e-4)")
    logger.info("-" * 80)
    logger.info(f"Cause: Chebyshev c0 = {CHEBYSHEV_C0:.11e} dominates for tiny inputs")
    logger.info("")
    logger.info(f"{'Value':>12} | {'Expected':>14} | {'Actual':>14} | {'ULP Error':>12}")
    logger.info("-" * 60)

    near_zero_values = [1e-38, 1e-30, 1e-20, 1e-10, 1e-8]
    max_ulp_region2 = 0

    for val in near_zero_values:
        torch_input = torch.tensor([[val]], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        actual = ttnn.to_torch(tt_result).item()
        expected = 0.5 * val
        ulp = ulp_distance_bf16(actual, expected)
        max_ulp_region2 = max(max_ulp_region2, ulp)
        logger.info(f"{val:12.2e} | {expected:14.2e} | {actual:14.2e} | {ulp:12,}")

    logger.info(f"\nMax ULP in Region 2: {max_ulp_region2:,}")

    # Region 3: Transition
    logger.info("")
    logger.info("REGION 3: TRANSITION (-5.5 to ~-4.0)")
    logger.info("-" * 80)
    logger.info("Cause: Polynomial poorly fitted near the -5.5 threshold boundary")
    logger.info("")
    logger.info(f"{'Value':>10} | {'Expected':>14} | {'Actual':>14} | {'ULP Error':>12}")
    logger.info("-" * 60)

    transition_values = [-5.5, -5.375, -5.25, -5.0, -4.5, -4.0]
    max_ulp_region3 = 0

    for val in transition_values:
        torch_input = torch.tensor([[val]], dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
        actual = ttnn.to_torch(tt_result).item()
        expected = gelu_exact(val)
        ulp = ulp_distance_bf16(actual, expected)
        max_ulp_region3 = max(max_ulp_region3, ulp)
        logger.info(f"{val:10.4f} | {expected:14.2e} | {actual:14.2e} | {ulp:12,}")

    logger.info(f"\nMax ULP in Region 3: {max_ulp_region3:,}")

    # Overall Summary
    logger.info("")
    logger.info("=" * 100)
    logger.info("OVERALL SUMMARY")
    logger.info("=" * 100)
    logger.info(f"Region 1 (Deep Negative Tail): Max ULP = {max_ulp_region1:,}")
    logger.info(f"Region 2 (Near-Zero):          Max ULP = {max_ulp_region2:,}")
    logger.info(f"Region 3 (Transition):         Max ULP = {max_ulp_region3:,}")
    logger.info("")
    logger.info("NOTE: Region 1 has the WORST error (32,767 = max possible for BF16)")
    logger.info("")
    logger.info("Source: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h")
    logger.info("Research: https://github.com/ivoitovych/bf16_gelu_research (shows Max ULP=1 is achievable)")
    logger.info("=" * 100)

    # Always pass - this is a documentation test
    assert True
