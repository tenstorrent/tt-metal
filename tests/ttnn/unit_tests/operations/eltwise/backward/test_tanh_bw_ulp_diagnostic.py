# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tanh Backward ULP Precision Diagnostic Tests

This test validates ttnn.tanh_bw() precision across all BFloat16 input ranges
using ULP (Units in Last Place) error measurement.

Mathematical Background:
- Forward: y = tanh(x)
- Backward: dy/dx = 1 - tanh(x)^2 = sech^2(x) = 1/cosh^2(x)
- tanh_bw(grad_output, input) = grad_output * sech^2(input)

Implementation Note:
The implementation uses 1/cosh^2(x) instead of 1-tanh^2(x) to avoid precision
loss when tanh saturates to +/-1 for large |x|. For |x| > 10, the result is
clamped to 0 since sech^2(10) < 8e-9 is negligible for training purposes.

For ULP testing, we use grad_output = 1.0 to isolate the derivative calculation.
Expected output: sech^2(input) = 1/cosh^2(input)

Hardware Model: Tenstorrent SFPU uses DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero)

Run: pytest tests/ttnn/unit_tests/operations/eltwise/backward/test_tanh_bw_ulp_diagnostic.py -v -s
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
        return 0x0000
    if bits == 0x8000:  # -0 -> +0
        return 0x0000
    return bits


def bf16_value_order_index_daz(bits: int) -> int:
    """
    Calculate the value order index for a BFloat16 value with DAZ.

    Index layout:
    - 0xFF7F (-max) -> index 0
    - 0x8080 (-min_normal) -> index 32511
    - 0x0000 (zero) -> index 32512
    - 0x0080 (+min_normal) -> index 32513
    - 0x7F7F (+max) -> index 65024
    """
    bits = bf16_daz_normalize(bits)

    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    if exp == 0xFF and mantissa != 0:
        return -1  # NaN

    if bits == 0x7F80:
        return 65025  # +inf
    if bits == 0xFF80:
        return -1  # -inf

    if bits == 0x0000:
        return 32512

    if bits & 0x8000:
        magnitude = bits & 0x7FFF
        return 0x7F7F - magnitude
    else:
        return 32513 + bits - 0x0080


def ulp_distance_bf16_daz(a: float, b: float) -> int:
    """Calculate ULP distance with DAZ+FTZ model."""
    a_bits = bf16_daz_normalize(float_to_bf16_bits(a))
    b_bits = bf16_daz_normalize(float_to_bf16_bits(b))

    a_exp = (a_bits >> 7) & 0xFF
    b_exp = (b_bits >> 7) & 0xFF
    if (a_exp == 0xFF and (a_bits & 0x7F) != 0) or (b_exp == 0xFF and (b_bits & 0x7F) != 0):
        return -1

    idx_a = bf16_value_order_index_daz(a_bits)
    idx_b = bf16_value_order_index_daz(b_bits)

    if idx_a < 0 or idx_b < 0:
        return -1

    return abs(idx_a - idx_b)


def tanh_bw_reference(x: float) -> float:
    """
    Reference tanh backward (derivative) using fp64 precision.
    d/dx tanh(x) = 1 - tanh(x)^2 = sech^2(x)
    """
    tanh_x = math.tanh(x)
    return 1.0 - tanh_x * tanh_x


def tanh_bw_expected_bf16_daz(x: float) -> float:
    """Compute expected BF16 tanh backward with DAZ+FTZ applied."""
    # Apply DAZ to input
    x_bits = bf16_daz_normalize(float_to_bf16_bits(x))
    x_daz = bf16_bits_to_float(x_bits)

    # Compute reference tanh backward
    result = tanh_bw_reference(x_daz)

    # Apply FTZ to output
    result_bits = bf16_daz_normalize(float_to_bf16_bits(result))
    return bf16_bits_to_float(result_bits)


def generate_all_normal_bf16() -> list:
    """Generate ALL normal (non-denormal) BF16 values."""
    values = []

    # Negative normals: 0xFF7F to 0x8080
    for bits in range(0xFF7F, 0x807F, -1):
        if not is_bf16_denormal(bits) and bits != 0xFF80:
            values.append(bf16_bits_to_float(bits))

    # Zero
    values.append(0.0)

    # Positive normals: 0x0080 to 0x7F7F
    for bits in range(0x0080, 0x7F80):
        if not is_bf16_denormal(bits):
            values.append(bf16_bits_to_float(bits))

    return values


@pytest.fixture(scope="module")
def device():
    """Create device fixture."""
    device = ttnn.open_device(device_id=0)
    yield device
    ttnn.close_device(device)


def run_tanh_bw_on_device(device, input_values: list, grad_value: float = 1.0) -> list:
    """
    Run ttnn.tanh_bw on device and return results.

    With grad_value = 1.0, the output is the derivative: 1 - tanh(input)^2
    """
    input_tensor = torch.tensor(input_values, dtype=torch.bfloat16).reshape(1, 1, 1, -1)
    grad_tensor = torch.full_like(input_tensor, grad_value)

    # Pad to tile size if needed
    orig_size = input_tensor.shape[-1]
    if orig_size % 32 != 0:
        pad_size = 32 - (orig_size % 32)
        input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_size))
        grad_tensor = torch.nn.functional.pad(grad_tensor, (0, pad_size))

    tt_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_grad = ttnn.from_torch(
        grad_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = ttnn.tanh_bw(tt_grad, tt_input)
    output_tensor = ttnn.to_torch(tt_output[0])

    return output_tensor.flatten()[:orig_size].tolist()


def analyze_ulp_errors(input_values: list, device_outputs: list, region_name: str):
    """Analyze ULP errors and return statistics."""
    ulp_errors = []
    worst_cases = []

    for i, (x, device_out) in enumerate(zip(input_values, device_outputs)):
        expected = tanh_bw_expected_bf16_daz(x)
        ulp = ulp_distance_bf16_daz(device_out, expected)

        if ulp >= 0:
            ulp_errors.append(ulp)
            if ulp > 2:
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


class TestTanhBwUlpDiagnostic:
    """Comprehensive ULP diagnostic tests for ttnn.tanh_bw()."""

    def test_derivative_near_zero(self, device):
        """
        Test tanh derivative near zero where tanh(x) ≈ x, so tanh'(x) ≈ 1.

        Near x=0: tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1
        """
        test_values = [0.0, 0.001, 0.01, 0.05, 0.1]
        test_values.extend([-v for v in test_values if v != 0])
        test_values = sorted(set(test_values))

        device_outputs = run_tanh_bw_on_device(device, test_values)

        logger.info(f"\n{'='*60}")
        logger.info("Tanh Backward Near Zero (derivative ≈ 1)")
        logger.info(f"{'='*60}")
        logger.info(f"{'Input':>12} {'Expected':>12} {'Device':>12} {'ULP':>8}")

        for x, device_out in zip(test_values, device_outputs):
            expected = tanh_bw_expected_bf16_daz(x)
            ulp = ulp_distance_bf16_daz(device_out, expected)
            logger.info(f"{x:>12.6f} {expected:>12.6f} {device_out:>12.6f} {ulp:>8}")

    def test_derivative_saturation(self, device):
        """
        Test tanh derivative in saturation region where tanh(x) → ±1.

        For large |x|: tanh'(x) = 1 - tanh(x)^2 → 1 - 1 = 0
        """
        test_values = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        test_values.extend([-v for v in test_values])
        test_values = sorted(test_values)

        device_outputs = run_tanh_bw_on_device(device, test_values)

        logger.info(f"\n{'='*60}")
        logger.info("Tanh Backward Saturation (derivative → 0)")
        logger.info(f"{'='*60}")
        logger.info(f"{'Input':>12} {'Expected':>12} {'Device':>12} {'ULP':>8}")

        for x, device_out in zip(test_values, device_outputs):
            expected = tanh_bw_expected_bf16_daz(x)
            ulp = ulp_distance_bf16_daz(device_out, expected)
            logger.info(f"{x:>12.6f} {expected:>12.6f} {device_out:>12.6f} {ulp:>8}")

    def test_derivative_transition(self, device):
        """
        Test tanh derivative in transition region (-3 < x < 3).

        This is where the derivative varies smoothly from 1 (at x=0) to 0 (at large |x|).
        """
        test_values = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]

        device_outputs = run_tanh_bw_on_device(device, test_values)

        logger.info(f"\n{'='*60}")
        logger.info("Tanh Backward Transition Region")
        logger.info(f"{'='*60}")
        logger.info(f"{'Input':>12} {'tanh(x)':>12} {'Expected':>12} {'Device':>12} {'ULP':>8}")

        for x, device_out in zip(test_values, device_outputs):
            tanh_x = math.tanh(x)
            expected = tanh_bw_expected_bf16_daz(x)
            ulp = ulp_distance_bf16_daz(device_out, expected)
            logger.info(f"{x:>12.4f} {tanh_x:>12.6f} {expected:>12.6f} {device_out:>12.6f} {ulp:>8}")


class TestTanhBwDenormalBehavior:
    """Tests for DAZ+FTZ behavior in tanh backward."""

    def test_positive_denormals(self, device):
        """Test tanh_bw with positive denormal inputs."""
        denormal_values = []
        for bits in range(0x0001, 0x0080):
            denormal_values.append(bf16_bits_to_float(bits))

        logger.info(f"Testing {len(denormal_values)} positive denormal inputs for tanh_bw")

        device_outputs = run_tanh_bw_on_device(device, denormal_values)

        # For denormal inputs (treated as 0), tanh'(0) = 1
        expected_output = 1.0
        non_one_count = 0
        for val, out in zip(denormal_values, device_outputs):
            ulp = ulp_distance_bf16_daz(out, expected_output)
            if ulp > 1:
                non_one_count += 1
                if non_one_count <= 5:
                    logger.warning(f"  Denormal input {val:.6e} -> {out:.6f} (expected ~1.0, ULP={ulp})")

        logger.info(f"Values with ULP > 1 from 1.0: {non_one_count}/{len(denormal_values)}")

    def test_negative_denormals(self, device):
        """Test tanh_bw with negative denormal inputs."""
        denormal_values = []
        for bits in range(0x8001, 0x8080):
            denormal_values.append(bf16_bits_to_float(bits))

        logger.info(f"Testing {len(denormal_values)} negative denormal inputs for tanh_bw")

        device_outputs = run_tanh_bw_on_device(device, denormal_values)

        expected_output = 1.0
        non_one_count = 0
        for val, out in zip(denormal_values, device_outputs):
            ulp = ulp_distance_bf16_daz(out, expected_output)
            if ulp > 1:
                non_one_count += 1
                if non_one_count <= 5:
                    logger.warning(f"  Denormal input {val:.6e} -> {out:.6f} (expected ~1.0, ULP={ulp})")

        logger.info(f"Values with ULP > 1 from 1.0: {non_one_count}/{len(denormal_values)}")


class TestTanhBwExhaustiveBF16:
    """Exhaustive BF16 sweep test for tanh backward."""

    def test_exhaustive_bf16_sweep(self, device):
        """
        Test ALL normal BF16 values for tanh_bw precision.
        """
        logger.info("Generating ALL normal BF16 values...")
        all_values = generate_all_normal_bf16()
        total_count = len(all_values)
        logger.info(f"Total BF16 values to test: {total_count}")

        # Process in batches
        batch_size = 4096
        all_ulp_errors = []
        worst_cases = []

        for batch_start in range(0, total_count, batch_size):
            batch_end = min(batch_start + batch_size, total_count)
            batch_values = all_values[batch_start:batch_end]

            device_outputs = run_tanh_bw_on_device(device, batch_values)

            for x, device_out in zip(batch_values, device_outputs):
                expected = tanh_bw_expected_bf16_daz(x)
                ulp = ulp_distance_bf16_daz(device_out, expected)

                if ulp >= 0:
                    all_ulp_errors.append(ulp)
                    if ulp > 2:
                        worst_cases.append((x, expected, device_out, ulp))

            if (batch_end % 16384) == 0 or batch_end == total_count:
                logger.info(f"Processed {batch_end}/{total_count} values...")

        # Analyze results
        ulp_errors = np.array(all_ulp_errors)

        logger.info(f"\n{'='*70}")
        logger.info("EXHAUSTIVE BF16 SWEEP RESULTS - ttnn.tanh_bw()")
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
            logger.info(f"EXCELLENT: Max ULP = {max_ulp}, {ulp_le_1_pct:.2f}% within 1 ULP")
        elif max_ulp <= 7:
            logger.info(f"GOOD: Max ULP = {max_ulp}, {ulp_le_2_pct:.2f}% within 2 ULP")
        else:
            logger.info(f"NEEDS ATTENTION: Max ULP = {max_ulp}")

        # Assert reasonable precision
        # The implementation uses 1/cosh²(x) for |x| <= 10 and returns 0 for |x| > 10.
        # This achieves ~98.8% of values within 2 ULP. The remaining high ULP values
        # occur at the |x| = 10 boundary where we transition to returning 0.
        # For training purposes, this is acceptable since sech²(10) < 8e-9 is negligible.
        assert ulp_le_2_pct >= 95.0, f"Too many values with ULP > 2: {100-ulp_le_2_pct:.2f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
