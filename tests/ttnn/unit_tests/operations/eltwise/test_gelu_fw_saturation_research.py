#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Forward GELU BF16 Saturation Region Research

Sweeps all ~65K BF16 values to find exact saturation thresholds for GELU(x)
under the DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero) hardware model.

Determines:
1. Negative zero-saturation threshold: largest |x| where GELU(x) rounds to 0 in BF16
2. Positive identity-saturation threshold: smallest x where GELU(x) rounds to x in BF16
3. Optimal polynomial region boundaries

Run: python -m pytest tests/ttnn/unit_tests/operations/eltwise/test_gelu_fw_saturation_research.py -v -s
"""

import struct
import math
import pytest

try:
    import mpmath

    mpmath.mp.dps = 80  # 256-bit precision
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


# =============================================================================
# BF16 utilities
# =============================================================================


def float_to_bf16_bits_truncate(f: float) -> int:
    """Convert float to BF16 bit representation (truncation, for enumeration only)."""
    f32_bytes = struct.pack(">f", f)
    return struct.unpack(">H", f32_bytes[:2])[0]


def float_to_bf16_bits(f: float) -> int:
    """Convert float to BF16 bit representation (round-to-nearest-even).

    Matches hardware pack behavior per tech_reports/data_formats/data_formats.md
    and bfloat16::from_float() in tt_metal/impl/data_format/bfloat16.cpp.
    """
    if math.isnan(f):
        return 0x7FC0
    f32_bytes = struct.pack("<f", f)  # little-endian for uint32 arithmetic
    u32 = struct.unpack("<I", f32_bytes)[0]
    rounding_bias = ((u32 >> 16) & 1) + 0x7FFF
    return ((u32 + rounding_bias) >> 16) & 0xFFFF


def bf16_bits_to_float(bits: int) -> float:
    """Convert BF16 bits back to float."""
    f32_bytes = struct.pack(">H", bits) + b"\x00\x00"
    return struct.unpack(">f", f32_bytes)[0]


def is_bf16_denormal(bits: int) -> bool:
    """Check if BF16 value is denormal (exponent = 0, mantissa != 0)."""
    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    return exp == 0 and mantissa != 0


def apply_daz(f: float) -> float:
    """Apply Denormals-Are-Zero: if f is denormal in BF16, return 0."""
    bits = float_to_bf16_bits(f)
    if is_bf16_denormal(bits):
        return 0.0
    return f


def apply_ftz(f: float) -> float:
    """Apply Flush-To-Zero: if result is denormal in BF16, flush to 0."""
    bits = float_to_bf16_bits(f)
    if is_bf16_denormal(bits):
        return 0.0
    return f


def round_to_bf16(f: float) -> float:
    """Round float to nearest BF16 value (RNE, matching hardware)."""
    bits = float_to_bf16_bits(f)
    return bf16_bits_to_float(bits)


def round_to_bf16_daz_ftz(f: float) -> float:
    """Round to BF16 with DAZ+FTZ model."""
    result = round_to_bf16(f)
    if is_bf16_denormal(float_to_bf16_bits(result)):
        return 0.0
    return result


def ulp_distance_bf16_daz(a: float, b: float) -> int:
    """Compute ULP distance between two values in BF16 with DAZ model."""
    a_bits = float_to_bf16_bits(a)
    b_bits = float_to_bf16_bits(b)

    # DAZ: treat denormals as zero
    if is_bf16_denormal(a_bits):
        a_bits = 0x0000 if (a_bits & 0x8000) == 0 else 0x8000
    if is_bf16_denormal(b_bits):
        b_bits = 0x0000 if (b_bits & 0x8000) == 0 else 0x8000

    # Both zero (either sign)
    a_is_zero = (a_bits & 0x7FFF) == 0
    b_is_zero = (b_bits & 0x7FFF) == 0
    if a_is_zero and b_is_zero:
        return 0

    # Convert to signed linear index
    def to_linear(bits):
        if (bits & 0x7FFF) == 0:
            return 0
        if bits & 0x8000:  # negative
            return -(bits & 0x7FFF)
        return bits & 0x7FFF

    return abs(to_linear(a_bits) - to_linear(b_bits))


# =============================================================================
# Reference GELU computation
# =============================================================================


def gelu_reference_f64(x: float) -> float:
    """Compute GELU(x) using fp64 with erfc for numerical stability."""
    if x == 0.0:
        return 0.0
    if x > 0:
        # GELU(x) = x * 0.5 * (1 + erf(x/sqrt(2)))
        return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    else:
        # For negative x, use erfc for stability: 1 + erf(x/√2) = erfc(|x|/√2)
        return x * 0.5 * math.erfc(abs(x) / math.sqrt(2.0))


def gelu_reference_mpmath(x_float: float) -> float:
    """Compute GELU(x) using mpmath 256-bit precision."""
    if not HAS_MPMATH:
        return gelu_reference_f64(x_float)
    x = mpmath.mpf(x_float)
    sqrt2 = mpmath.sqrt(2)
    result = x * mpmath.mpf("0.5") * (1 + mpmath.erf(x / sqrt2))
    return float(result)


def gelu_expected_bf16_daz(x_float: float) -> float:
    """What we expect GELU(x) to be in BF16 with DAZ+FTZ model."""
    # DAZ: input denormals are zero
    x_daz = apply_daz(x_float)
    # Compute with high precision
    result = gelu_reference_mpmath(x_daz)
    # FTZ: flush denormal output to zero, then round to BF16
    return round_to_bf16_daz_ftz(result)


# =============================================================================
# Generate all valid BF16 values
# =============================================================================


def all_bf16_values():
    """Generate all valid (non-NaN, non-Inf, non-denormal) BF16 values."""
    values = []
    for bits in range(0x10000):
        exp = (bits >> 7) & 0xFF
        mantissa = bits & 0x7F
        # Skip NaN and Inf
        if exp == 0xFF:
            continue
        # Skip denormals (DAZ treats them as zero, +0 already included)
        if exp == 0 and mantissa != 0:
            continue
        f = bf16_bits_to_float(bits)
        values.append(f)
    return sorted(set(values))


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.skipif(not HAS_MPMATH, reason="mpmath not installed")
class TestGeluForwardSaturationResearch:
    """Research tests to find exact GELU(x) saturation thresholds in BF16."""

    def test_negative_zero_saturation_threshold(self):
        """Find the exact BF16 threshold where GELU(x) saturates to 0."""
        values = all_bf16_values()
        neg_values = sorted([v for v in values if v < 0], reverse=True)  # least negative first

        last_nonzero_x = None

        print("\n=== Negative Zero Saturation Research ===")
        print(f"{'x':>12s} {'GELU(x) exact':>20s} {'GELU(x) BF16':>15s} {'ULP':>6s}")
        print("-" * 60)

        for x in neg_values:
            gelu_exact = gelu_reference_mpmath(x)
            gelu_bf16 = gelu_expected_bf16_daz(x)

            if gelu_bf16 != 0.0 and last_nonzero_x is None:
                # Still in non-zero region (going more negative)
                pass

            if gelu_bf16 == 0.0 and last_nonzero_x is not None:
                print(f"... (transition) ...")
                print(f"{x:12.6f} {gelu_exact:20.12e} {gelu_bf16:15.8e}  <-- FIRST ZERO")
                break

            if gelu_bf16 != 0.0:
                last_nonzero_x = x

        # Now find exact boundary by scanning from most negative
        print(f"\n--- Scanning near boundary ---")
        boundary_values = [v for v in neg_values if -15.0 <= v <= -5.0]
        last_nonzero = None
        first_zero = None

        for x in sorted(boundary_values):  # most negative first
            gelu_bf16 = gelu_expected_bf16_daz(x)
            if gelu_bf16 == 0.0:
                if first_zero is None:
                    first_zero = x
            else:
                last_nonzero = x

        print(f"Last x with GELU(x) != 0 in BF16: {last_nonzero}")
        print(f"First x (going more negative) with GELU(x) == 0: {first_zero}")

        if last_nonzero and first_zero:
            gelu_last = gelu_reference_mpmath(last_nonzero)
            gelu_first = gelu_reference_mpmath(first_zero)
            print(f"  GELU({last_nonzero}) = {gelu_last:.15e} → BF16: {gelu_expected_bf16_daz(last_nonzero):.8e}")
            print(f"  GELU({first_zero}) = {gelu_first:.15e} → BF16: {gelu_expected_bf16_daz(first_zero):.8e}")

        # Print 20 values around the boundary
        print(f"\n--- Values around zero saturation boundary ---")
        boundary_zone = sorted([v for v in neg_values if -12.0 <= v <= -8.0])
        for x in boundary_zone:
            gelu_exact = gelu_reference_mpmath(x)
            gelu_bf16 = gelu_expected_bf16_daz(x)
            marker = (
                " <--"
                if gelu_bf16 == 0.0
                and gelu_expected_bf16_daz(
                    sorted([v for v in neg_values if v > x])[0] if any(v > x for v in neg_values) else x
                )
                != 0.0
                else ""
            )
            ulp = ulp_distance_bf16_daz(gelu_bf16, round_to_bf16_daz_ftz(gelu_exact))
            print(f"  x={x:10.4f}  GELU={gelu_exact:15.8e}  BF16={gelu_bf16:12.6e}  ULP={ulp}{marker}")

        assert last_nonzero is not None, "Should find non-zero GELU values"

    def test_positive_identity_saturation_threshold(self):
        """Find the exact BF16 threshold where GELU(x) saturates to x (identity)."""
        values = all_bf16_values()
        pos_values = sorted([v for v in values if v > 0])

        print("\n=== Positive Identity Saturation Research ===")
        print(f"{'x':>12s} {'GELU(x)':>20s} {'GELU(x) BF16':>15s} {'x BF16':>12s} {'Match?':>8s}")
        print("-" * 75)

        first_identity = None
        last_non_identity = None

        for x in pos_values:
            gelu_bf16 = gelu_expected_bf16_daz(x)
            x_bf16 = round_to_bf16_daz_ftz(x)

            is_identity = gelu_bf16 == x_bf16

            if not is_identity:
                last_non_identity = x
            elif is_identity and first_identity is None:
                first_identity = x

        print(f"Last x where GELU(x) != x in BF16: {last_non_identity}")
        print(f"First x where GELU(x) == x in BF16: {first_identity}")

        if last_non_identity and first_identity:
            gelu_last = gelu_reference_mpmath(last_non_identity)
            gelu_first = gelu_reference_mpmath(first_identity)
            print(f"  GELU({last_non_identity}) = {gelu_last:.15e}")
            print(f"  GELU({first_identity}) = {gelu_first:.15e}")

        # Print 30 values around the boundary
        print(f"\n--- Values around identity saturation boundary ---")
        boundary = sorted([v for v in pos_values if 2.0 <= v <= 6.0])
        for x in boundary:
            gelu_exact = gelu_reference_mpmath(x)
            gelu_bf16 = gelu_expected_bf16_daz(x)
            x_bf16 = round_to_bf16_daz_ftz(x)
            is_id = "YES" if gelu_bf16 == x_bf16 else "no"
            print(f"  x={x:8.4f}  GELU={gelu_exact:15.8e}  BF16={gelu_bf16:12.6e}  x_bf16={x_bf16:12.6e}  id={is_id}")

        assert first_identity is not None, "Should find identity region"

    def test_comprehensive_bf16_sweep(self):
        """Sweep ALL BF16 values and compute ULP errors with current reference.

        This identifies the worst-case regions and provides the data needed
        to design the piecewise polynomial approximation.
        """
        values = all_bf16_values()

        # Compute reference GELU for all values
        results = []
        for x in values:
            gelu_exact = gelu_reference_mpmath(x)
            gelu_bf16 = round_to_bf16_daz_ftz(gelu_exact)
            results.append((x, gelu_exact, gelu_bf16))

        # Analyze by regions
        regions = {
            "x <= -10": [],
            "(-10, -5]": [],
            "(-5, -3]": [],
            "(-3, 0]": [],
            "(0, 3]": [],
            "(3, 5]": [],
            "x > 5": [],
        }

        for x, gelu_exact, gelu_bf16 in results:
            if x <= -10:
                regions["x <= -10"].append((x, gelu_exact, gelu_bf16))
            elif x <= -5:
                regions["(-10, -5]"].append((x, gelu_exact, gelu_bf16))
            elif x <= -3:
                regions["(-5, -3]"].append((x, gelu_exact, gelu_bf16))
            elif x <= 0:
                regions["(-3, 0]"].append((x, gelu_exact, gelu_bf16))
            elif x <= 3:
                regions["(0, 3]"].append((x, gelu_exact, gelu_bf16))
            elif x <= 5:
                regions["(3, 5]"].append((x, gelu_exact, gelu_bf16))
            else:
                regions["x > 5"].append((x, gelu_exact, gelu_bf16))

        print("\n=== Forward GELU BF16 Reference Analysis ===")
        print(
            f"{'Region':>15s} {'Count':>7s} {'Min GELU':>15s} {'Max GELU':>15s} {'All zero?':>10s} {'All identity?':>14s}"
        )
        print("-" * 80)

        for region_name, region_data in regions.items():
            if not region_data:
                continue
            count = len(region_data)
            min_gelu = min(g for _, g, _ in region_data)
            max_gelu = max(g for _, g, _ in region_data)
            all_zero = all(gb == 0.0 for _, _, gb in region_data)
            all_identity = all(gb == round_to_bf16_daz_ftz(x) for x, _, gb in region_data)
            print(
                f"{region_name:>15s} {count:>7d} {min_gelu:>15.6e} {max_gelu:>15.6e} {'YES' if all_zero else 'no':>10s} {'YES' if all_identity else 'no':>14s}"
            )

        print(f"\nTotal BF16 values analyzed: {len(values)}")

    def test_polynomial_region_analysis(self):
        """Detailed analysis of the core polynomial region to determine optimal boundaries.

        Examines GELU(x) behavior in [-5, 5] with fine granularity to find
        where polynomial approximation should transition to other methods.
        """
        values = all_bf16_values()
        core_values = sorted([v for v in values if -6.0 <= v <= 6.0])

        print("\n=== Core Region Polynomial Analysis ===")
        print(f"{'x':>10s} {'GELU(x)':>15s} {'GELU/x':>12s} {'|GELU-x|':>12s} {'Notes':>20s}")
        print("-" * 75)

        # Sample every ~0.25 for readability
        prev_x = -999
        for x in core_values:
            if abs(x - prev_x) < 0.2 and abs(x) > 0.5:
                continue
            prev_x = x

            gelu = gelu_reference_mpmath(x)
            ratio = gelu / x if x != 0 else 0.5  # GELU(0)/0 = lim = 0.5
            diff = abs(gelu - x)

            notes = ""
            if abs(ratio - 1.0) < 1e-6:
                notes = "~identity"
            elif abs(gelu) < 1e-38:
                notes = "~zero"
            elif abs(ratio - 0.5) < 0.01:
                notes = "~x/2"

            print(f"{x:10.4f} {gelu:15.8e} {ratio:12.6f} {diff:12.6e} {notes:>20s}")

    def test_near_zero_detailed(self):
        """Detailed analysis of GELU(x) near x=0.

        GELU(0) = 0, and GELU(x) ≈ x/2 for small x.
        The Chebyshev polynomial has a floor value issue at 2.98e-05.
        """
        values = all_bf16_values()
        near_zero = sorted([v for v in values if -0.1 <= v <= 0.1])

        print("\n=== Near-Zero Analysis ===")
        print(f"{'x':>12s} {'GELU(x) exact':>18s} {'BF16 rounded':>15s} {'x/2':>12s} {'ratio GELU/x':>14s}")
        print("-" * 75)

        for x in near_zero:
            gelu = gelu_reference_mpmath(x)
            gelu_bf16 = round_to_bf16_daz_ftz(gelu)
            half_x = x / 2 if x != 0 else 0
            ratio = gelu / x if x != 0 else 0.5
            print(f"{x:12.8f} {gelu:18.10e} {gelu_bf16:15.8e} {half_x:12.8f} {ratio:14.8f}")
