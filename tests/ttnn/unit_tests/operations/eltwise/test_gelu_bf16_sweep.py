# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
GELU BF16 Full Sweep Test - Captures all BF16 values in [-14, 14] range

This test:
1. Generates ALL possible BF16 values in the [-14, 14] range
2. Compares ttnn.gelu(accurate) vs torch.nn.functional.gelu
3. Plots output comparison, ULP errors, and absolute tolerance

Run: python tests/ttnn/unit_tests/operations/eltwise/test_gelu_bf16_sweep.py
"""

import struct
import math
import torch
import ttnn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
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

    # Handle sign differences
    if (a_bits >> 15) != (b_bits >> 15):
        # Different signs - count distance through zero
        if a_bits >> 15:  # a is negative
            a_bits = 0x8000 - (a_bits & 0x7FFF) if a_bits != 0x8000 else 0
        if b_bits >> 15:  # b is negative
            b_bits = 0x8000 - (b_bits & 0x7FFF) if b_bits != 0x8000 else 0
        return a_bits + b_bits

    return abs(int(a_bits) - int(b_bits))


def generate_all_bf16_in_range(min_val: float, max_val: float) -> List[float]:
    """Generate all possible BF16 values in the given range."""
    values = []

    # BF16 has 16 bits: 1 sign + 8 exponent + 7 mantissa
    # Total possible values: 2^16 = 65536
    for bits in range(0x10000):
        f = bf16_bits_to_float(bits)
        # Skip NaN and Inf
        if math.isnan(f) or math.isinf(f):
            continue
        if min_val <= f <= max_val:
            values.append(f)

    # Sort by value for plotting
    values.sort()
    return values


def run_gelu_sweep(device, min_val: float = -14.0, max_val: float = 14.0):
    """Run GELU comparison for all BF16 values in range."""

    logger.info(f"Generating all BF16 values in [{min_val}, {max_val}]...")
    bf16_values = generate_all_bf16_in_range(min_val, max_val)
    logger.info(f"Generated {len(bf16_values)} BF16 values")

    # Convert to tensors
    input_tensor = torch.tensor(bf16_values, dtype=torch.bfloat16)

    # PyTorch reference (using float64 for maximum accuracy)
    input_f64 = torch.tensor(bf16_values, dtype=torch.float64)
    pytorch_gelu = torch.nn.functional.gelu(input_f64).numpy()

    # TTNN GELU
    logger.info("Running ttnn.gelu (accurate mode)...")
    tt_input = ttnn.from_torch(
        input_tensor.reshape(1, 1, 1, -1),  # Reshape to 4D for tile layout
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_result = ttnn.gelu(tt_input, fast_and_approximate_mode=False)
    # Convert bfloat16 to float32 before numpy (numpy doesn't support bfloat16)
    ttnn_gelu = ttnn.to_torch(tt_result).flatten().float().numpy()

    # Calculate errors
    logger.info("Calculating errors...")
    x_values = np.array(bf16_values)

    # Absolute tolerance
    atol = np.abs(ttnn_gelu - pytorch_gelu)

    # ULP distance
    ulp_errors = np.array(
        [ulp_distance_bf16(float(ttnn_gelu[i]), float(pytorch_gelu[i])) for i in range(len(bf16_values))]
    )

    # Statistics (handle inf/nan values)
    ulp_finite = ulp_errors[np.isfinite(ulp_errors)]
    atol_finite = atol[np.isfinite(atol)]

    max_ulp = np.max(ulp_finite) if len(ulp_finite) > 0 else float("inf")
    mean_ulp = np.mean(ulp_finite) if len(ulp_finite) > 0 else float("inf")
    max_atol = np.max(atol_finite) if len(atol_finite) > 0 else float("inf")
    mean_atol = np.mean(atol_finite) if len(atol_finite) > 0 else float("inf")

    # Count inf/nan values
    inf_count = np.sum(~np.isfinite(atol))

    # Find worst cases (use finite values only)
    ulp_for_argmax = np.where(np.isfinite(ulp_errors), ulp_errors, -1)
    atol_for_argmax = np.where(np.isfinite(atol), atol, -1)
    worst_ulp_idx = np.argmax(ulp_for_argmax)
    worst_atol_idx = np.argmax(atol_for_argmax)

    logger.info("=" * 80)
    logger.info("GELU BF16 SWEEP RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total BF16 values tested: {len(bf16_values)}")
    logger.info(f"Range: [{min_val}, {max_val}]")
    logger.info("")
    logger.info("ULP Statistics:")
    logger.info(f"  Max ULP:  {max_ulp:,}")
    logger.info(f"  Mean ULP: {mean_ulp:.2f}")
    logger.info(
        f"  Worst ULP at x={x_values[worst_ulp_idx]:.6f}: "
        f"expected={pytorch_gelu[worst_ulp_idx]:.6e}, "
        f"actual={ttnn_gelu[worst_ulp_idx]:.6e}"
    )
    logger.info("")
    logger.info("Absolute Tolerance Statistics:")
    logger.info(f"  Max Atol:  {max_atol:.6e}")
    logger.info(f"  Mean Atol: {mean_atol:.6e}")
    logger.info(
        f"  Worst Atol at x={x_values[worst_atol_idx]:.6f}: "
        f"expected={pytorch_gelu[worst_atol_idx]:.6e}, "
        f"actual={ttnn_gelu[worst_atol_idx]:.6e}"
    )
    logger.info("=" * 80)

    # Create plots
    create_plots(x_values, pytorch_gelu, ttnn_gelu, ulp_errors, atol, min_val, max_val)

    return {
        "x_values": x_values,
        "pytorch_gelu": pytorch_gelu,
        "ttnn_gelu": ttnn_gelu,
        "ulp_errors": ulp_errors,
        "atol": atol,
        "max_ulp": max_ulp,
        "mean_ulp": mean_ulp,
        "max_atol": max_atol,
        "mean_atol": mean_atol,
    }


def create_plots(x_values, pytorch_gelu, ttnn_gelu, ulp_errors, atol, min_val, max_val):
    """Create comprehensive plots for GELU comparison."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"GELU BF16 Accuracy Analysis: [{min_val}, {max_val}] Range\n" f"({len(x_values):,} BF16 values tested)",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1: GELU outputs comparison
    ax1 = axes[0, 0]
    ax1.plot(x_values, pytorch_gelu, "b-", label="PyTorch GELU (reference)", alpha=0.7, linewidth=0.5)
    ax1.plot(x_values, ttnn_gelu, "r--", label="TTNN GELU (accurate)", alpha=0.7, linewidth=0.5)
    ax1.set_xlabel("Input x")
    ax1.set_ylabel("GELU(x)")
    ax1.set_title("GELU Output Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min_val, max_val)

    # Plot 2: Output difference
    ax2 = axes[0, 1]
    diff = ttnn_gelu - pytorch_gelu
    ax2.scatter(x_values, diff, s=1, alpha=0.5, c="purple")
    ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Input x")
    ax2.set_ylabel("TTNN - PyTorch")
    ax2.set_title("Output Difference (TTNN - PyTorch)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min_val, max_val)

    # Plot 3: ULP errors
    ax3 = axes[1, 0]
    # Use log scale for ULP if there's a wide range
    colors = np.where(ulp_errors > 100, "red", np.where(ulp_errors > 10, "orange", "green"))
    ax3.scatter(x_values, ulp_errors, s=1, alpha=0.5, c=colors)
    ax3.axhline(y=1, color="green", linestyle="--", linewidth=1, label="ULP=1 (ideal)")
    ax3.axhline(y=10, color="orange", linestyle="--", linewidth=1, label="ULP=10")
    ax3.axhline(y=100, color="red", linestyle="--", linewidth=1, label="ULP=100")
    ax3.set_xlabel("Input x")
    ax3.set_ylabel("ULP Error")
    ulp_finite_plot = ulp_errors[np.isfinite(ulp_errors)]
    ax3.set_title(f"ULP Error (Max: {np.max(ulp_finite_plot):,}, Mean: {np.mean(ulp_finite_plot):.1f})")
    ax3.set_yscale("symlog", linthresh=1)  # Symmetric log scale
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(min_val, max_val)

    # Plot 4: Absolute tolerance
    ax4 = axes[1, 1]
    ax4.scatter(x_values, atol, s=1, alpha=0.5, c="blue")
    ax4.set_xlabel("Input x")
    ax4.set_ylabel("Absolute Tolerance |TTNN - PyTorch|")
    atol_finite_plot = atol[np.isfinite(atol)]
    ax4.set_title(f"Absolute Tolerance (Max: {np.max(atol_finite_plot):.2e}, Mean: {np.mean(atol_finite_plot):.2e})")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(min_val, max_val)

    plt.tight_layout()

    # Save the plot
    output_path = "/home/ubuntu/code/tt-metal/gelu_bf16_sweep_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved to: {output_path}")

    # Also create a zoomed plot for problematic regions
    create_zoomed_plots(x_values, pytorch_gelu, ttnn_gelu, ulp_errors, atol)

    plt.show()


def create_zoomed_plots(x_values, pytorch_gelu, ttnn_gelu, ulp_errors, atol):
    """Create zoomed plots for problematic regions."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("GELU Problematic Regions Analysis", fontsize=14, fontweight="bold")

    # Region 1: Deep negative tail (x < -5.5)
    mask1 = x_values < -5.5
    if np.any(mask1):
        ax = axes[0, 0]
        ax.scatter(x_values[mask1], ulp_errors[mask1], s=3, alpha=0.7, c="red")
        ax.set_xlabel("Input x")
        ax.set_ylabel("ULP Error")
        ax.set_title("Region 1: Deep Negative (x < -5.5)\nHardware returns 0")
        ax.grid(True, alpha=0.3)

    # Region 2: Near zero
    mask2 = np.abs(x_values) < 0.01
    if np.any(mask2):
        ax = axes[0, 1]
        ax.scatter(x_values[mask2], ulp_errors[mask2], s=3, alpha=0.7, c="orange")
        ax.set_xlabel("Input x")
        ax.set_ylabel("ULP Error")
        ax.set_title("Region 2: Near Zero (|x| < 0.01)\nFloor value bug")
        ax.grid(True, alpha=0.3)

    # Region 3: Transition region (-5.5 to -4)
    mask3 = (x_values >= -5.5) & (x_values <= -4.0)
    if np.any(mask3):
        ax = axes[0, 2]
        ax.scatter(x_values[mask3], ulp_errors[mask3], s=3, alpha=0.7, c="purple")
        ax.set_xlabel("Input x")
        ax.set_ylabel("ULP Error")
        ax.set_title("Region 3: Transition (-5.5 to -4.0)\nPolynomial boundary")
        ax.grid(True, alpha=0.3)

    # ULP histogram
    ax = axes[1, 0]
    ulp_clipped = np.clip(ulp_errors, 0, 1000)  # Clip for histogram
    ax.hist(ulp_clipped, bins=100, edgecolor="black", alpha=0.7)
    ax.set_xlabel("ULP Error")
    ax.set_ylabel("Count")
    ax.set_title("ULP Error Distribution (clipped to 1000)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Atol histogram
    ax = axes[1, 1]
    # Filter out inf/nan values for histogram
    atol_finite = atol[np.isfinite(atol)]
    if len(atol_finite) > 0:
        ax.hist(np.log10(atol_finite + 1e-45), bins=100, edgecolor="black", alpha=0.7, color="blue")
    ax.set_xlabel("log10(Atol)")
    ax.set_ylabel("Count")
    inf_count = np.sum(~np.isfinite(atol))
    ax.set_title(f"Absolute Tolerance Distribution (log scale)\n({inf_count} inf/nan values excluded)")
    ax.grid(True, alpha=0.3)

    # Summary table
    ax = axes[1, 2]
    ax.axis("off")

    # Helper function for region stats with inf handling
    def region_stats(mask, name):
        if not np.any(mask):
            return ""
        region_ulp = ulp_errors[mask]
        region_ulp_finite = region_ulp[np.isfinite(region_ulp)]
        inf_count = np.sum(~np.isfinite(region_ulp))
        text = f"{name}:\n"
        text += f"  Count: {np.sum(mask):,}\n"
        if len(region_ulp_finite) > 0:
            text += f"  Max ULP: {np.max(region_ulp_finite):,}\n"
            text += f"  Mean ULP: {np.mean(region_ulp_finite):.1f}\n"
        if inf_count > 0:
            text += f"  Inf/NaN: {inf_count}\n"
        text += "\n"
        return text

    # Calculate stats for each region
    stats_text = "REGION STATISTICS\n" + "=" * 40 + "\n\n"
    stats_text += region_stats(mask1, "Region 1 (x < -5.5)")
    stats_text += region_stats(mask2, "Region 2 (|x| < 0.01)")
    stats_text += region_stats(mask3, "Region 3 (-5.5 to -4.0)")

    # Good region
    mask_good = (x_values >= -4.0) & (x_values <= 4.0) & (np.abs(x_values) >= 0.01)
    stats_text += region_stats(mask_good, "Good Region (-4 to 4, |x|>0.01)")

    ax.text(
        0.1,
        0.9,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    output_path = "/home/ubuntu/code/tt-metal/gelu_bf16_regions_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Zoomed plot saved to: {output_path}")


def main():
    """Main entry point."""
    logger.info("Starting GELU BF16 Full Sweep Test")

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        results = run_gelu_sweep(device, min_val=-14.0, max_val=14.0)

        # Print summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Total BF16 values: {len(results['x_values']):,}")
        print(f"Max ULP Error:     {results['max_ulp']:,}")
        print(f"Mean ULP Error:    {results['mean_ulp']:.2f}")
        print(f"Max Atol:          {results['max_atol']:.6e}")
        print(f"Mean Atol:         {results['mean_atol']:.6e}")
        print("=" * 80)

    finally:
        ttnn.close_device(device)

    logger.info("Test completed!")


if __name__ == "__main__":
    main()
