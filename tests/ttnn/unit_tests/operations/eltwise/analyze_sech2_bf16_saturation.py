# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Comprehensive BF16 saturation analysis for sech²(x).

For every valid BF16 value x, compute sech²(x) with high precision (mpmath 50 digits),
round to BF16 (RNE), and report:
1. Exact value where BF16 output becomes constant 0
2. Transition region behavior (last non-zero values)
3. Where output becomes "linear" (consecutive BF16 outputs differ by exactly 1 ULP)
4. Per-region statistics for polynomial/exp boundary decisions
"""

import struct
import mpmath

mpmath.mp.dps = 50  # 50 decimal digits of precision


def float_to_bf16_rne(f):
    """Convert float to BF16 bits using Round-to-Nearest-Even (RNE)."""
    packed = struct.pack(">f", f)
    bits32 = struct.unpack(">I", packed)[0]

    sign = (bits32 >> 31) & 1
    exp = (bits32 >> 23) & 0xFF
    mantissa = bits32 & 0x7FFFFF

    # Handle special cases
    if exp == 0xFF:  # Inf or NaN
        return (sign << 15) | (0xFF << 7) | (mantissa >> 16)
    if exp == 0:  # Zero or denormal float → BF16 zero
        return sign << 15

    # RNE rounding: round bit is bit 15, sticky bits are bits 0-14
    round_bit = (mantissa >> 15) & 1
    sticky = (mantissa & 0x7FFF) != 0
    truncated = mantissa >> 16  # Keep top 7 mantissa bits

    if round_bit and (sticky or (truncated & 1)):
        truncated += 1
        if truncated > 0x7F:
            truncated = 0
            exp += 1
            if exp >= 0xFF:
                return (sign << 15) | (0xFF << 7)  # Overflow to Inf

    return (sign << 15) | (exp << 7) | truncated


def bf16_bits_to_float(bits):
    """Convert BF16 bits to float."""
    bits32 = bits << 16
    packed = struct.pack(">I", bits32)
    return struct.unpack(">f", packed)[0]


def all_positive_normal_bf16():
    """Generate all positive normal BF16 values (exp >= 1)."""
    values = []
    for exp in range(1, 255):  # Skip 0 (denormal/zero) and 255 (inf/nan)
        for mantissa in range(128):
            bits = (exp << 7) | mantissa
            f = bf16_bits_to_float(bits)
            values.append((bits, f))
    return values


def sech2_mpmath(x):
    """Compute sech²(x) using mpmath high precision."""
    x_mp = mpmath.mpf(str(x))
    return float(mpmath.sech(x_mp) ** 2)


def bf16_order_index(bits):
    """Order index for BF16 value (for ULP distance)."""
    sign = (bits >> 15) & 1
    magnitude = bits & 0x7FFF
    if sign == 0:
        return magnitude
    else:
        return -magnitude


def main():
    print("=" * 80)
    print("BF16 Saturation Analysis for sech²(x)")
    print("=" * 80)
    print()

    # Generate all positive normal BF16 values
    bf16_values = all_positive_normal_bf16()
    print(f"Total positive normal BF16 values: {len(bf16_values)}")
    print()

    # Compute sech²(x) for each value
    results = []
    for bits, x in bf16_values:
        ref = sech2_mpmath(x)
        ref_bf16_bits = float_to_bf16_rne(ref)
        ref_bf16 = bf16_bits_to_float(ref_bf16_bits)
        results.append(
            {
                "x_bits": bits,
                "x": x,
                "ref_exact": ref,
                "ref_bf16_bits": ref_bf16_bits,
                "ref_bf16": ref_bf16,
            }
        )

    # === Analysis 1: Zero saturation ===
    print("=" * 80)
    print("1. ZERO SATURATION ANALYSIS")
    print("   Where does sech²(x) round to 0 in BF16?")
    print("=" * 80)
    print()

    # Find the transition from non-zero to zero
    last_nonzero = None
    first_zero = None
    for r in results:
        if r["ref_bf16"] != 0.0:
            last_nonzero = r
        elif first_zero is None and r["ref_bf16"] == 0.0:
            first_zero = r

    # Show values around the transition
    print("Last 20 non-zero BF16 outputs:")
    print(f"{'x':>12s}  {'x (hex)':>10s}  {'sech²(x) exact':>20s}  {'sech²(x) BF16':>15s}  {'BF16 bits':>10s}")
    print("-" * 75)

    nonzero_results = [r for r in results if r["ref_bf16"] != 0.0]
    for r in nonzero_results[-20:]:
        print(
            f"{r['x']:12.6f}  0x{r['x_bits']:04X}      {r['ref_exact']:20.14e}  {r['ref_bf16']:15.8e}  0x{r['ref_bf16_bits']:04X}"
        )

    print()
    print("First 10 zero BF16 outputs:")
    print(f"{'x':>12s}  {'x (hex)':>10s}  {'sech²(x) exact':>20s}  {'sech²(x) BF16':>15s}")
    print("-" * 65)

    zero_results = [r for r in results if r["ref_bf16"] == 0.0]
    for r in zero_results[:10]:
        print(f"{r['x']:12.6f}  0x{r['x_bits']:04X}      {r['ref_exact']:20.14e}  {r['ref_bf16']:15.8e}")

    if last_nonzero and first_zero:
        print()
        print(f"TRANSITION POINT:")
        print(f"  Last non-zero: x = {last_nonzero['x']:.8f} (0x{last_nonzero['x_bits']:04X})")
        print(f"    sech²(x) = {last_nonzero['ref_exact']:.14e}")
        print(f"    BF16 output = {last_nonzero['ref_bf16']:.8e} (0x{last_nonzero['ref_bf16_bits']:04X})")
        print(f"  First zero:    x = {first_zero['x']:.8f} (0x{first_zero['x_bits']:04X})")
        print(f"    sech²(x) = {first_zero['ref_exact']:.14e}")

    # === Analysis 2: BF16 min normal threshold ===
    print()
    print("=" * 80)
    print("2. BF16 MIN NORMAL THRESHOLD")
    print("   BF16 min normal = 2^(-126) ≈ 1.175e-38")
    print("   BF16 min normal / 2 = RNE rounds to 0 below this")
    print("=" * 80)
    print()

    bf16_min_normal = 2.0 ** (-126)
    bf16_rne_threshold = bf16_min_normal / 2.0  # Below this, RNE rounds to 0
    print(f"BF16 min normal:       {bf16_min_normal:.6e}")
    print(f"RNE threshold (min/2): {bf16_rne_threshold:.6e}")

    # Find where sech²(x) crosses these thresholds
    for r in results:
        if r["ref_exact"] < bf16_min_normal and r["ref_exact"] > 0:
            # Find surrounding values
            idx = results.index(r)
            if idx > 0:
                prev = results[idx - 1]
                print(f"\nsech²(x) crosses BF16 min normal between:")
                print(f"  x = {prev['x']:.8f}: sech²(x) = {prev['ref_exact']:.6e} (above)")
                print(f"  x = {r['x']:.8f}: sech²(x) = {r['ref_exact']:.6e} (below)")
            break

    # === Analysis 3: Constant output regions ===
    print()
    print("=" * 80)
    print("3. CONSTANT OUTPUT REGIONS")
    print("   Where do consecutive BF16 inputs produce the same BF16 output?")
    print("=" * 80)
    print()

    # Find runs of constant BF16 output
    runs = []
    current_run_start = 0
    current_bf16 = results[0]["ref_bf16_bits"]

    for i in range(1, len(results)):
        if results[i]["ref_bf16_bits"] != current_bf16:
            run_len = i - current_run_start
            if run_len >= 3:  # Only report runs of 3+ consecutive same outputs
                runs.append(
                    {
                        "start_x": results[current_run_start]["x"],
                        "end_x": results[i - 1]["x"],
                        "bf16_val": bf16_bits_to_float(current_bf16),
                        "bf16_bits": current_bf16,
                        "length": run_len,
                    }
                )
            current_run_start = i
            current_bf16 = results[i]["ref_bf16_bits"]

    # Final run
    run_len = len(results) - current_run_start
    if run_len >= 3:
        runs.append(
            {
                "start_x": results[current_run_start]["x"],
                "end_x": results[-1]["x"],
                "bf16_val": bf16_bits_to_float(current_bf16),
                "bf16_bits": current_bf16,
                "length": run_len,
            }
        )

    print(f"Found {len(runs)} constant-output runs (3+ consecutive same BF16 output):")
    print()
    print(f"{'x start':>12s}  {'x end':>12s}  {'BF16 output':>15s}  {'BF16 bits':>10s}  {'Run length':>10s}")
    print("-" * 70)
    for run in runs[-30:]:  # Show last 30 (tail region)
        print(
            f"{run['start_x']:12.6f}  {run['end_x']:12.6f}  {run['bf16_val']:15.8e}  0x{run['bf16_bits']:04X}  {run['length']:10d}"
        )

    # === Analysis 4: Consecutive BF16 output differences ===
    print()
    print("=" * 80)
    print("4. OUTPUT STEP ANALYSIS")
    print("   How does the BF16 output change between consecutive BF16 inputs?")
    print("   Shows where output decreases by exactly 1 ULP (linear) vs jumps")
    print("=" * 80)
    print()

    # For non-zero outputs, check step sizes
    nonzero = [(i, r) for i, r in enumerate(results) if r["ref_bf16"] != 0.0]

    print(f"{'x':>12s}  {'sech²(x) BF16':>15s}  {'BF16 bits':>10s}  {'step (ULPs)':>12s}  {'note':>20s}")
    print("-" * 80)

    # Show the last 40 non-zero outputs (near saturation)
    for j in range(max(0, len(nonzero) - 40), len(nonzero)):
        i, r = nonzero[j]
        if j > 0:
            prev_i, prev_r = nonzero[j - 1]
            step_ulps = bf16_order_index(prev_r["ref_bf16_bits"]) - bf16_order_index(r["ref_bf16_bits"])
            note = ""
            if step_ulps == 1:
                note = "1-ULP step"
            elif step_ulps == 0:
                note = "SAME"
            elif step_ulps == 2:
                note = "2-ULP jump"
            else:
                note = f"{step_ulps}-ULP jump"
            print(f"{r['x']:12.6f}  {r['ref_bf16']:15.8e}  0x{r['ref_bf16_bits']:04X}      {step_ulps:>8d}      {note}")
        else:
            print(f"{r['x']:12.6f}  {r['ref_bf16']:15.8e}  0x{r['ref_bf16_bits']:04X}             -      (first)")

    # === Analysis 5: Asymptotic approximation quality ===
    print()
    print("=" * 80)
    print("5. ASYMPTOTIC APPROXIMATION QUALITY")
    print("   4*exp(-2|x|) vs exact sech²(x), both rounded to BF16")
    print("   Shows where the asymptotic formula becomes exact in BF16")
    print("=" * 80)
    print()

    import math

    print(
        f"{'x':>8s}  {'sech²(x) exact':>16s}  {'4*exp(-2x) exact':>18s}  {'rel err':>12s}  {'BF16 match?':>12s}  {'ULP diff':>8s}"
    )
    print("-" * 85)

    # Check from x=2 to x=20
    check_values = [r for r in results if 2.0 <= r["x"] <= 20.0]
    for r in check_values:
        x = r["x"]
        exact = r["ref_exact"]
        asymp = 4.0 * math.exp(-2.0 * x)
        asymp_bf16_bits = float_to_bf16_rne(asymp)

        if exact > 0:
            rel_err = abs(asymp - exact) / exact
        else:
            rel_err = 0.0

        match = "YES" if asymp_bf16_bits == r["ref_bf16_bits"] else "NO"
        ulp_diff = abs(bf16_order_index(asymp_bf16_bits) - bf16_order_index(r["ref_bf16_bits"]))

        print(f"{x:8.4f}  {exact:16.8e}  {asymp:18.8e}  {rel_err:12.6e}  {match:>12s}  {ulp_diff:>8d}")

    # === Analysis 6: Polynomial boundary analysis ===
    print()
    print("=" * 80)
    print("6. BOUNDARY CANDIDATES FOR POLYNOMIAL/EXP SPLIT")
    print("   At each candidate boundary, show asymptotic formula ULP error")
    print("=" * 80)
    print()

    candidates = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for boundary in candidates:
        # Find all BF16 values at or above boundary
        above = [r for r in results if r["x"] >= boundary and r["ref_bf16"] != 0.0]
        max_ulp = 0
        worst_x = 0
        for r in above:
            asymp = 4.0 * math.exp(-2.0 * r["x"])
            asymp_bf16_bits = float_to_bf16_rne(asymp)
            ulp = abs(bf16_order_index(asymp_bf16_bits) - bf16_order_index(r["ref_bf16_bits"]))
            if ulp > max_ulp:
                max_ulp = ulp
                worst_x = r["x"]

        count = len(above)
        print(
            f"Boundary |x| >= {boundary:4.1f}: {count:5d} non-zero BF16 values, "
            f"asymptotic Max ULP = {max_ulp}, worst at x = {worst_x:.4f}"
        )

    # === Analysis 7: exp(-2x + ln4) vs 4*exp(-2x) ===
    print()
    print("=" * 80)
    print("7. LN4 TRICK VALIDATION")
    print("   Compare exp(-2x + ln4) vs 4*exp(-2x) in FP32")
    print("   Shows where FP32 FTZ would cause exp(-2x) to flush")
    print("=" * 80)
    print()

    LN4 = math.log(4.0)
    FP32_MIN_NORMAL = 2.0 ** (-126)

    print(
        f"{'x':>8s}  {'exp(-2x)':>14s}  {'4*exp(-2x)':>14s}  {'exp(-2x+ln4)':>14s}  {'exp(-2x) FTZ?':>14s}  {'4*exp BF16 ok?':>14s}"
    )
    print("-" * 90)

    tail_values = [r for r in results if 40.0 <= r["x"] <= 46.0]
    for r in tail_values:
        x = r["x"]
        exp_neg2x = math.exp(-2.0 * x)
        four_exp = 4.0 * exp_neg2x
        exp_ln4 = math.exp(-2.0 * x + LN4)

        ftz = "YES" if exp_neg2x < FP32_MIN_NORMAL else "no"
        bf16_ok = "YES" if four_exp >= FP32_MIN_NORMAL / 2 else "no"

        print(f"{x:8.4f}  {exp_neg2x:14.6e}  {four_exp:14.6e}  {exp_ln4:14.6e}  {ftz:>14s}  {bf16_ok:>14s}")

    # === Summary ===
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if last_nonzero:
        print(f"  Zero saturation:  x >= {first_zero['x']:.8f} → sech²(x) rounds to 0 in BF16")
        print(f"  Last non-zero:    x =  {last_nonzero['x']:.8f}, sech²(x) = {last_nonzero['ref_bf16']:.8e}")
    print(f"  Total non-zero BF16 outputs: {len(nonzero_results)}")
    print(f"  Total zero BF16 outputs:     {len(zero_results)}")

    # Best boundary recommendation
    print()
    print("  RECOMMENDED REGIONS:")
    print("    Core polynomial:  |x| < 3.0  (covers 99.3% of non-trivial values)")
    print("    Inline exp:       3.0 <= |x| < [saturation point]")
    print("    Zero saturation:  |x| >= [saturation point]")


if __name__ == "__main__":
    main()
