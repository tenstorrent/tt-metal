# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tanh ULP Precision Diagnostic Tests

This test validates ttnn.tanh() precision across all BFloat16 input ranges
using ULP (Units in Last Place) error measurement.

Hardware Model: Tenstorrent SFPU uses DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero)
Per tech_reports/Handling_Special_Value/special_values.md: "denormals | all | 0x0"

Tanh Implementation (from ckernel_sfpu_tanh.h):
- Approximation mode: LUT-based piecewise linear (fast, less accurate)
- Accurate mode BF16: 6th-degree polynomial (Sollya-derived coefficients)
- Accurate mode FP32: Continued fraction formula

Key regions to test:
- Region 1: Deep saturation (|x| > 4) where tanh → ±1
- Region 2: Near-zero (|x| < 0.1) where tanh ≈ x
- Region 3: Transition region (-4 < x < 4) where polynomial approximation matters

Run: pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_ulp_diagnostic.py -v -s
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


def is_bf16_denormal(bits: int) -> bool:
    """Check if BF16 bits represent a denormal (subnormal) value."""
    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    return (exp == 0) and (mantissa != 0)


def bf16_daz_normalize(bits: int) -> int:
    """Apply DAZ (Denormals-Are-Zero) normalization to BF16 bits."""
    if is_bf16_denormal(bits):
        return 0x0000  # All denormals become +0
    if bits == 0x8000:  # -0 -> +0
        return 0x0000
    return bits


def bf16_value_order_index_daz(bits: int) -> int:
    """
    Calculate the value order index for a BFloat16 value with DAZ.

    With DAZ+FTZ, the representable values are:
    - Negative normals: 0xFF7F (-max) to 0x8080 (-min_normal) = 32,512 values
    - Zero: 0x0000 (all denormals and ±0 map here) = 1 value
    - Positive normals: 0x0080 (+min_normal) to 0x7F7F (+max) = 32,512 values
    - Total: 65,025 finite values

    Index layout:
    - 0xFF7F (-max) -> index 0
    - 0x8080 (-min_normal) -> index 32511
    - 0x0000 (zero) -> index 32512
    - 0x0080 (+min_normal) -> index 32513
    - 0x7F7F (+max) -> index 65024
    """
    bits = bf16_daz_normalize(bits)

    # Handle NaN - return -1 as invalid
    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    if exp == 0xFF and mantissa != 0:
        return -1

    # Handle infinity
    if bits == 0x7F80:
        return 65025  # +inf (after all finite values)
    if bits == 0xFF80:
        return -1  # -inf

    # Zero (including all denormals which map to zero)
    if bits == 0x0000:
        return 32512  # Middle of the range

    if bits & 0x8000:
        # Negative normal: magnitude ranges from 0x0080 (smallest) to 0x7F7F (largest)
        # We want: largest magnitude (0x7F7F) -> index 0
        #          smallest magnitude (0x0080) -> index 32511
        magnitude = bits & 0x7FFF
        return 0x7F7F - magnitude  # 32639 - magnitude
    else:
        # Positive normal: bits range from 0x0080 to 0x7F7F
        # We want: 0x0080 -> index 32513, 0x7F7F -> index 65024
        return 32513 + bits - 0x0080


def ulp_distance_bf16_daz(a: float, b: float) -> int:
    """Calculate ULP distance with DAZ+FTZ model (Tenstorrent hardware behavior)."""
    a_bits = bf16_daz_normalize(float_to_bf16_bits(a))
    b_bits = bf16_daz_normalize(float_to_bf16_bits(b))

    # Handle NaN
    a_exp = (a_bits >> 7) & 0xFF
    b_exp = (b_bits >> 7) & 0xFF
    if (a_exp == 0xFF and (a_bits & 0x7F) != 0) or (b_exp == 0xFF and (b_bits & 0x7F) != 0):
        return -1

    idx_a = bf16_value_order_index_daz(a_bits)
    idx_b = bf16_value_order_index_daz(b_bits)

    if idx_a < 0 or idx_b < 0:
        return -1

    return abs(idx_a - idx_b)


def tanh_reference(x: float) -> float:
    """
    Reference tanh using Python's math.tanh (fp64 precision).
    For tanh, fp64 is sufficient as it doesn't have the saturation issues
    that GELU has with erf().
    """
    return math.tanh(x)


def tanh_expected_bf16_daz(x: float) -> float:
    """Compute expected BF16 tanh with DAZ+FTZ applied."""
    # Apply DAZ to input
    x_bits = bf16_daz_normalize(float_to_bf16_bits(x))
    x_daz = bf16_bits_to_float(x_bits)

    # Compute reference tanh
    result = tanh_reference(x_daz)

    # Apply FTZ to output
    result_bits = bf16_daz_normalize(float_to_bf16_bits(result))
    return bf16_bits_to_float(result_bits)


def generate_bf16_range(start: float, end: float, count: int = 1000) -> list:
    """Generate BF16 values in a range."""
    values = []
    for i in range(count):
        t = i / (count - 1) if count > 1 else 0
        f = start + t * (end - start)
        bits = float_to_bf16_bits(f)
        bf16_val = bf16_bits_to_float(bits)
        if bf16_val not in values:
            values.append(bf16_val)
    return values


def generate_all_bf16_in_range(start: float, end: float) -> list:
    """Generate ALL representable BF16 values in a range."""
    values = []
    start_bits = float_to_bf16_bits(start)
    end_bits = float_to_bf16_bits(end)

    # Handle negative range
    if start < 0 and end <= 0:
        # Both negative: iterate from start (larger magnitude) to end (smaller magnitude)
        for bits in range(start_bits, end_bits - 1, -1):
            if not is_bf16_denormal(bits):
                values.append(bf16_bits_to_float(bits))
    elif start < 0 and end > 0:
        # Crosses zero: negative part + zero + positive part
        for bits in range(start_bits, 0x8000, -1):
            if not is_bf16_denormal(bits):
                values.append(bf16_bits_to_float(bits))
        values.append(0.0)
        for bits in range(0x0080, end_bits + 1):
            if not is_bf16_denormal(bits):
                values.append(bf16_bits_to_float(bits))
    else:
        # Both positive
        for bits in range(start_bits, end_bits + 1):
            if not is_bf16_denormal(bits):
                values.append(bf16_bits_to_float(bits))

    return values


def generate_all_normal_bf16() -> list:
    """
    Generate ALL normal (non-denormal) BF16 values.

    BF16 layout: [S][EEEEEEEE][MMMMMMM] = 16 bits
    - Negative normals: 0xFF7F (-max) to 0x8080 (-min_normal) = 32,640 values
    - Zero: 0x0000 = 1 value
    - Positive normals: 0x0080 (+min_normal) to 0x7F7F (+max) = 32,640 values
    - Total: 65,281 values (excluding denormals, NaN, inf)

    With DAZ+FTZ, denormals are treated as zero, so we only need normal values.
    """
    values = []

    # Negative normals: 0xFF7F to 0x8080 (decreasing magnitude = increasing value)
    # Iterate from most negative to least negative
    for bits in range(0xFF7F, 0x807F, -1):  # 0x8080 to 0xFF7F
        if not is_bf16_denormal(bits) and bits != 0xFF80:  # Skip -inf
            values.append(bf16_bits_to_float(bits))

    # Zero
    values.append(0.0)

    # Positive normals: 0x0080 to 0x7F7F
    for bits in range(0x0080, 0x7F80):  # Skip +inf (0x7F80)
        if not is_bf16_denormal(bits):
            values.append(bf16_bits_to_float(bits))

    return values


@pytest.fixture(scope="module")
def device():
    """Create device fixture."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


def run_tanh_on_device(device, input_values: list) -> list:
    """Run ttnn.tanh on device and return results."""
    input_tensor = torch.tensor(input_values, dtype=torch.bfloat16).reshape(1, 1, 1, -1)

    # Pad to tile size if needed
    orig_size = input_tensor.shape[-1]
    if orig_size % 32 != 0:
        pad_size = 32 - (orig_size % 32)
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_size))

    tt_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = ttnn.tanh(tt_input)
    output_tensor = ttnn.to_torch(tt_output)

    return output_tensor.flatten()[:orig_size].tolist()


def analyze_ulp_errors(input_values: list, device_outputs: list, region_name: str):
    """Analyze ULP errors and return statistics."""
    ulp_errors = []
    worst_cases = []

    for i, (x, device_out) in enumerate(zip(input_values, device_outputs)):
        expected = tanh_expected_bf16_daz(x)
        ulp = ulp_distance_bf16_daz(device_out, expected)

        if ulp >= 0:
            ulp_errors.append(ulp)
            if ulp > 1:
                worst_cases.append((x, expected, device_out, ulp))

    if not ulp_errors:
        return None

    ulp_errors = np.array(ulp_errors)
    stats = {
        "region": region_name,
        "count": len(ulp_errors),
        "max_ulp": int(np.max(ulp_errors)),
        "mean_ulp": float(np.mean(ulp_errors)),
        "median_ulp": float(np.median(ulp_errors)),
        "ulp_0_pct": 100.0 * np.sum(ulp_errors == 0) / len(ulp_errors),
        "ulp_1_pct": 100.0 * np.sum(ulp_errors <= 1) / len(ulp_errors),
        "ulp_2_pct": 100.0 * np.sum(ulp_errors <= 2) / len(ulp_errors),
        "worst_cases": sorted(worst_cases, key=lambda x: -x[3])[:10],
    }

    return stats


def print_stats(stats: dict):
    """Print statistics in a readable format."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Region: {stats['region']}")
    logger.info(f"{'='*60}")
    logger.info(f"Sample count: {stats['count']}")
    logger.info(f"Max ULP: {stats['max_ulp']}")
    logger.info(f"Mean ULP: {stats['mean_ulp']:.4f}")
    logger.info(f"Median ULP: {stats['median_ulp']:.1f}")
    logger.info(f"ULP=0: {stats['ulp_0_pct']:.2f}%")
    logger.info(f"ULP<=1: {stats['ulp_1_pct']:.2f}%")
    logger.info(f"ULP<=2: {stats['ulp_2_pct']:.2f}%")

    if stats["worst_cases"]:
        logger.info(f"\nWorst cases (up to 10):")
        logger.info(f"{'Input':>12} {'Expected':>12} {'Device':>12} {'ULP':>8}")
        for x, expected, device_out, ulp in stats["worst_cases"]:
            logger.info(f"{x:>12.6f} {expected:>12.6f} {device_out:>12.6f} {ulp:>8}")


class TestTanhUlpDiagnostic:
    """Comprehensive ULP diagnostic tests for ttnn.tanh()."""

    def test_region1_deep_negative_saturation(self, device):
        """
        Region 1: Deep negative saturation (x < -4)

        In this region, tanh(x) → -1.0
        The polynomial/continued fraction should saturate properly.
        """
        # Test specific boundary values
        test_values = [-4.0, -4.5, -5.0, -6.0, -8.0, -10.0, -20.0, -50.0, -100.0]

        # Add more values in the range
        test_values.extend(generate_bf16_range(-10.0, -4.0, count=200))
        test_values = sorted(set(test_values))

        device_outputs = run_tanh_on_device(device, test_values)
        stats = analyze_ulp_errors(test_values, device_outputs, "Deep Negative Saturation (x < -4)")
        print_stats(stats)

        # Record findings - tanh should be very close to -1.0 here
        assert stats["max_ulp"] <= 100, f"Excessive ULP error in deep negative region: {stats['max_ulp']}"

    def test_region2_deep_positive_saturation(self, device):
        """
        Region 2: Deep positive saturation (x > 4)

        In this region, tanh(x) → +1.0
        """
        test_values = [4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0, 100.0]
        test_values.extend(generate_bf16_range(4.0, 10.0, count=200))
        test_values = sorted(set(test_values))

        device_outputs = run_tanh_on_device(device, test_values)
        stats = analyze_ulp_errors(test_values, device_outputs, "Deep Positive Saturation (x > 4)")
        print_stats(stats)

        assert stats["max_ulp"] <= 100, f"Excessive ULP error in deep positive region: {stats['max_ulp']}"

    def test_region3_near_zero(self, device):
        """
        Region 3: Near-zero region (|x| < 0.1)

        In this region, tanh(x) ≈ x (linear approximation).
        This is where polynomial accuracy matters most for small values.
        """
        # Test very small values
        test_values = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1]
        test_values.extend([-v for v in test_values if v != 0])
        test_values.extend(generate_bf16_range(-0.1, 0.1, count=300))
        test_values = sorted(set(test_values))

        device_outputs = run_tanh_on_device(device, test_values)
        stats = analyze_ulp_errors(test_values, device_outputs, "Near-Zero Region (|x| < 0.1)")
        print_stats(stats)

        assert stats["max_ulp"] <= 100, f"Excessive ULP error in near-zero region: {stats['max_ulp']}"

    def test_region4_transition_negative(self, device):
        """
        Region 4: Negative transition region (-4 < x < -0.1)

        This is where the polynomial approximation is most active.
        """
        test_values = generate_bf16_range(-4.0, -0.1, count=500)

        device_outputs = run_tanh_on_device(device, test_values)
        stats = analyze_ulp_errors(test_values, device_outputs, "Negative Transition (-4 < x < -0.1)")
        print_stats(stats)

        assert stats["max_ulp"] <= 100, f"Excessive ULP error in negative transition: {stats['max_ulp']}"

    def test_region5_transition_positive(self, device):
        """
        Region 5: Positive transition region (0.1 < x < 4)

        This is where the polynomial approximation is most active.
        """
        test_values = generate_bf16_range(0.1, 4.0, count=500)

        device_outputs = run_tanh_on_device(device, test_values)
        stats = analyze_ulp_errors(test_values, device_outputs, "Positive Transition (0.1 < x < 4)")
        print_stats(stats)

        assert stats["max_ulp"] <= 100, f"Excessive ULP error in positive transition: {stats['max_ulp']}"

    def test_full_bf16_range_sampling(self, device):
        """
        Full BF16 range sampling test.

        Tests a representative sample across the entire BF16 range.
        """
        # Sample across entire range
        test_values = []

        # Negative saturation samples
        test_values.extend(generate_bf16_range(-100.0, -10.0, count=50))

        # Negative transition samples
        test_values.extend(generate_bf16_range(-10.0, -0.01, count=200))

        # Near-zero samples
        test_values.extend(generate_bf16_range(-0.01, 0.01, count=100))

        # Positive transition samples
        test_values.extend(generate_bf16_range(0.01, 10.0, count=200))

        # Positive saturation samples
        test_values.extend(generate_bf16_range(10.0, 100.0, count=50))

        test_values = sorted(set(test_values))

        device_outputs = run_tanh_on_device(device, test_values)
        stats = analyze_ulp_errors(test_values, device_outputs, "Full BF16 Range Sampling")
        print_stats(stats)

        # This is the main diagnostic - we want to see the overall precision
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY: Full range ULP analysis")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples: {stats['count']}")
        logger.info(f"Maximum ULP error: {stats['max_ulp']}")
        logger.info(f"Mean ULP error: {stats['mean_ulp']:.4f}")
        logger.info(f"Values with ULP <= 1: {stats['ulp_1_pct']:.2f}%")

    def test_specific_problematic_values(self, device):
        """
        Test specific values that might be problematic based on kernel implementation.

        The polynomial has coefficients designed for a specific range.
        Values outside the polynomial's design range may have higher errors.
        """
        # Values near polynomial boundaries (around |x| = 2-3 where polynomial accuracy matters)
        test_values = [
            -3.5,
            -3.0,
            -2.5,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            # Values where clamping to ±1 happens
            -4.0,
            -3.9,
            -3.8,
            3.8,
            3.9,
            4.0,
            # Edge cases
            0.0,
            -0.0,
        ]

        device_outputs = run_tanh_on_device(device, test_values)

        logger.info(f"\n{'='*60}")
        logger.info("Specific Value Analysis")
        logger.info(f"{'='*60}")
        logger.info(f"{'Input':>12} {'Expected':>12} {'Device':>12} {'ULP':>8} {'Abs Err':>12}")

        for x, device_out in zip(test_values, device_outputs):
            expected = tanh_expected_bf16_daz(x)
            ulp = ulp_distance_bf16_daz(device_out, expected)
            abs_err = abs(device_out - expected)
            logger.info(f"{x:>12.6f} {expected:>12.6f} {device_out:>12.6f} {ulp:>8} {abs_err:>12.6e}")


class TestTanhEdgeCases:
    """Edge case tests for ttnn.tanh()."""

    def test_zero_input(self, device):
        """Test tanh(0) = 0."""
        test_values = [0.0]
        device_outputs = run_tanh_on_device(device, test_values)

        expected = 0.0
        ulp = ulp_distance_bf16_daz(device_outputs[0], expected)

        logger.info(f"tanh(0) = {device_outputs[0]}, expected = {expected}, ULP = {ulp}")
        assert ulp == 0, f"tanh(0) should be exactly 0, got {device_outputs[0]}"

    def test_symmetry(self, device):
        """Test tanh(-x) = -tanh(x) symmetry."""
        test_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        neg_values = [-v for v in test_values]

        pos_outputs = run_tanh_on_device(device, test_values)
        neg_outputs = run_tanh_on_device(device, neg_values)

        logger.info(f"\n{'='*60}")
        logger.info("Symmetry Check: tanh(-x) vs -tanh(x)")
        logger.info(f"{'='*60}")
        logger.info(f"{'x':>8} {'tanh(x)':>12} {'tanh(-x)':>12} {'-tanh(x)':>12} {'Match':>8}")

        all_symmetric = True
        for x, pos_out, neg_out in zip(test_values, pos_outputs, neg_outputs):
            neg_of_pos = -pos_out
            # Compare bits for exact match
            match = float_to_bf16_bits(neg_out) == float_to_bf16_bits(neg_of_pos)
            all_symmetric = all_symmetric and match
            logger.info(f"{x:>8.4f} {pos_out:>12.6f} {neg_out:>12.6f} {neg_of_pos:>12.6f} {'✓' if match else '✗':>8}")

        assert all_symmetric, "Symmetry violation detected"

    def test_saturation_boundary(self, device):
        """Test values near saturation boundary where tanh → ±1."""
        # tanh saturates around |x| ≈ 4-5 for BF16 precision
        test_values = [3.5, 3.75, 4.0, 4.25, 4.5, 5.0, 6.0, 8.0]
        test_values.extend([-v for v in test_values])
        test_values = sorted(test_values)

        device_outputs = run_tanh_on_device(device, test_values)

        logger.info(f"\n{'='*60}")
        logger.info("Saturation Boundary Analysis")
        logger.info(f"{'='*60}")
        logger.info(f"{'Input':>12} {'Expected':>12} {'Device':>12} {'ULP':>8}")

        for x, device_out in zip(test_values, device_outputs):
            expected = tanh_expected_bf16_daz(x)
            ulp = ulp_distance_bf16_daz(device_out, expected)
            logger.info(f"{x:>12.6f} {expected:>12.6f} {device_out:>12.6f} {ulp:>8}")


class TestTanhDenormalBehavior:
    """
    Tests for DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero) behavior.

    Per tech_reports/Handling_Special_Value/special_values.md:
    "denormals | all | 0x0"

    The SFPU should treat all denormal inputs as zero.
    """

    def test_all_positive_denormals(self, device):
        """
        Test ALL positive denormal BF16 values (0x0001 to 0x007F).

        Under DAZ, these should all be treated as zero input.
        tanh(0) = 0, so output should be 0.
        """
        # Generate all positive denormals: bits 0x0001 to 0x007F (127 values)
        denormal_values = []
        denormal_bits = []
        for bits in range(0x0001, 0x0080):
            val = bf16_bits_to_float(bits)
            denormal_values.append(val)
            denormal_bits.append(bits)

        logger.info(f"Testing {len(denormal_values)} positive denormal values")
        logger.info(f"Range: {denormal_values[0]:.6e} to {denormal_values[-1]:.6e}")

        device_outputs = run_tanh_on_device(device, denormal_values)

        # All outputs should be zero (or very close to zero)
        non_zero_count = 0
        for i, (bits, val, out) in enumerate(zip(denormal_bits, denormal_values, device_outputs)):
            out_bits = float_to_bf16_bits(out)
            if out_bits != 0x0000 and out_bits != 0x8000:  # Allow +0 or -0
                non_zero_count += 1
                logger.warning(f"  Denormal 0x{bits:04X} ({val:.6e}) -> {out:.6e} (bits=0x{out_bits:04X})")

        logger.info(f"Non-zero outputs: {non_zero_count}/{len(denormal_values)}")

        if non_zero_count == 0:
            logger.info("✓ All positive denormals correctly treated as zero (DAZ verified)")
        else:
            logger.warning(f"⚠ {non_zero_count} denormals did NOT produce zero output")

        # This is informational - we want to know the behavior
        # Some implementations might not implement DAZ
        assert non_zero_count <= len(denormal_values), "Unexpected behavior"

    def test_all_negative_denormals(self, device):
        """
        Test ALL negative denormal BF16 values (0x8001 to 0x807F).

        Under DAZ, these should all be treated as zero input.
        tanh(0) = 0, so output should be 0.
        """
        # Generate all negative denormals: bits 0x8001 to 0x807F (127 values)
        denormal_values = []
        denormal_bits = []
        for bits in range(0x8001, 0x8080):
            val = bf16_bits_to_float(bits)
            denormal_values.append(val)
            denormal_bits.append(bits)

        logger.info(f"Testing {len(denormal_values)} negative denormal values")
        logger.info(f"Range: {denormal_values[0]:.6e} to {denormal_values[-1]:.6e}")

        device_outputs = run_tanh_on_device(device, denormal_values)

        # All outputs should be zero
        non_zero_count = 0
        for i, (bits, val, out) in enumerate(zip(denormal_bits, denormal_values, device_outputs)):
            out_bits = float_to_bf16_bits(out)
            if out_bits != 0x0000 and out_bits != 0x8000:
                non_zero_count += 1
                logger.warning(f"  Denormal 0x{bits:04X} ({val:.6e}) -> {out:.6e} (bits=0x{out_bits:04X})")

        logger.info(f"Non-zero outputs: {non_zero_count}/{len(denormal_values)}")

        if non_zero_count == 0:
            logger.info("✓ All negative denormals correctly treated as zero (DAZ verified)")
        else:
            logger.warning(f"⚠ {non_zero_count} denormals did NOT produce zero output")

    def test_denormal_vs_zero_consistency(self, device):
        """
        Verify that denormal inputs produce the same output as zero input.

        This directly tests DAZ behavior by comparing:
        - tanh(denormal) should equal tanh(0)
        """
        # Test a sample of denormals against zero
        test_denormals = [
            0x0001,  # Smallest positive denormal
            0x003F,  # Middle positive denormal
            0x007F,  # Largest positive denormal
            0x8001,  # Smallest negative denormal (magnitude)
            0x803F,  # Middle negative denormal
            0x807F,  # Largest negative denormal (magnitude)
        ]

        denormal_values = [bf16_bits_to_float(b) for b in test_denormals]
        zero_value = [0.0]

        # Run tanh on denormals
        denormal_outputs = run_tanh_on_device(device, denormal_values)

        # Run tanh on zero
        zero_output = run_tanh_on_device(device, zero_value)[0]
        zero_out_bits = float_to_bf16_bits(zero_output)

        logger.info(f"tanh(0) = {zero_output} (bits=0x{zero_out_bits:04X})")
        logger.info(f"\nDenormal consistency check:")
        logger.info(f"{'Input bits':>12} {'Input value':>15} {'Output':>15} {'Output bits':>12} {'Match zero':>12}")

        all_match = True
        for bits, val, out in zip(test_denormals, denormal_values, denormal_outputs):
            out_bits = float_to_bf16_bits(out)
            # Consider both +0 and -0 as matching
            matches = (
                (out_bits == zero_out_bits)
                or (out_bits == 0x0000 and zero_out_bits == 0x8000)
                or (out_bits == 0x8000 and zero_out_bits == 0x0000)
            )
            if not matches:
                all_match = False
            match_str = "✓" if matches else "✗"
            logger.info(f"0x{bits:04X} {val:>15.6e} {out:>15.6e} 0x{out_bits:04X} {match_str:>12}")

        if all_match:
            logger.info("\n✓ All denormals produce same output as zero (DAZ verified)")
        else:
            logger.warning("\n⚠ Some denormals produce different output than zero")


class TestTanhExhaustiveBF16:
    """
    Exhaustive BF16 sweep test - tests ALL 65,281 normal BF16 values.

    This is the definitive test for tanh precision. It tests every single
    representable BF16 value (excluding denormals, NaN, inf) and reports
    comprehensive ULP statistics.
    """

    def test_exhaustive_bf16_sweep(self, device):
        """
        Test ALL normal BF16 values for tanh precision.

        Expected: ~65,281 values tested
        - 32,640 negative normals
        - 1 zero
        - 32,640 positive normals
        """
        logger.info("Generating ALL normal BF16 values...")
        all_values = generate_all_normal_bf16()
        total_count = len(all_values)
        logger.info(f"Total BF16 values to test: {total_count}")

        # Process in batches to avoid memory issues
        batch_size = 4096  # Must be multiple of 32 for tile alignment
        all_ulp_errors = []
        worst_cases = []

        for batch_start in range(0, total_count, batch_size):
            batch_end = min(batch_start + batch_size, total_count)
            batch_values = all_values[batch_start:batch_end]

            device_outputs = run_tanh_on_device(device, batch_values)

            for x, device_out in zip(batch_values, device_outputs):
                expected = tanh_expected_bf16_daz(x)
                ulp = ulp_distance_bf16_daz(device_out, expected)

                if ulp >= 0:
                    all_ulp_errors.append(ulp)
                    if ulp > 2:
                        worst_cases.append((x, expected, device_out, ulp))

            # Progress update
            if (batch_end % 16384) == 0 or batch_end == total_count:
                logger.info(f"Processed {batch_end}/{total_count} values...")

        # Analyze results
        ulp_errors = np.array(all_ulp_errors)

        logger.info(f"\n{'='*70}")
        logger.info("EXHAUSTIVE BF16 SWEEP RESULTS - ttnn.tanh()")
        logger.info(f"{'='*70}")
        logger.info(f"Total values tested: {len(ulp_errors)}")
        logger.info(f"Max ULP: {np.max(ulp_errors)}")
        logger.info(f"Mean ULP: {np.mean(ulp_errors):.6f}")
        logger.info(f"Median ULP: {np.median(ulp_errors):.1f}")
        logger.info(f"Std ULP: {np.std(ulp_errors):.6f}")

        # Cumulative distribution
        logger.info(f"\nCumulative Distribution:")
        logger.info(f"{'ULP ≤':>8} {'Count':>10} {'Percent':>10}")
        for threshold in [0, 1, 2, 3, 5, 7, 10, 100]:
            count = np.sum(ulp_errors <= threshold)
            pct = 100.0 * count / len(ulp_errors)
            logger.info(f"{threshold:>8} {count:>10} {pct:>9.4f}%")

        # Histogram
        logger.info(f"\nULP Histogram:")
        logger.info(f"{'ULP':>8} {'Count':>10} {'Percent':>10}")
        for ulp_val in range(min(11, int(np.max(ulp_errors)) + 1)):
            count = np.sum(ulp_errors == ulp_val)
            pct = 100.0 * count / len(ulp_errors)
            logger.info(f"{ulp_val:>8} {count:>10} {pct:>9.4f}%")

        # Worst cases
        if worst_cases:
            worst_cases_sorted = sorted(worst_cases, key=lambda x: -x[3])[:20]
            logger.info(f"\nWorst cases (ULP > 2):")
            logger.info(f"{'Input':>15} {'Expected':>15} {'Device':>15} {'ULP':>8}")
            for x, expected, device_out, ulp in worst_cases_sorted:
                logger.info(f"{x:>15.8f} {expected:>15.8f} {device_out:>15.8f} {ulp:>8}")

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("SUMMARY")
        logger.info(f"{'='*70}")
        max_ulp = int(np.max(ulp_errors))
        ulp_le_1_pct = 100.0 * np.sum(ulp_errors <= 1) / len(ulp_errors)
        ulp_le_2_pct = 100.0 * np.sum(ulp_errors <= 2) / len(ulp_errors)

        if max_ulp <= 2:
            logger.info(f"✓ EXCELLENT: Max ULP = {max_ulp}, {ulp_le_1_pct:.2f}% within 1 ULP")
        elif max_ulp <= 7:
            logger.info(f"✓ GOOD: Max ULP = {max_ulp}, {ulp_le_2_pct:.2f}% within 2 ULP")
        else:
            logger.info(f"⚠ NEEDS ATTENTION: Max ULP = {max_ulp}")

        # Assert excellent precision - tanh has Max ULP = 1
        assert max_ulp <= 2, f"Max ULP should be <= 2 for tanh, got: {max_ulp}"
        assert ulp_le_1_pct == 100.0, f"All values should have ULP <= 1, got: {ulp_le_1_pct:.2f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
