#!/usr/bin/env python3
"""
Plot TTFT vs Input Sequence Length comparison:
- NVIDIA (extracted from reference graph)
- TT-Galaxy Baseline (Non-fused 7×8 4-links)
- TT-Galaxy Fused (6×8/6×9 3-links)
"""

import matplotlib.pyplot as plt
import numpy as np

# Data
seq_lens = [4, 8, 16, 32, 64, 128]  # in K
seq_lens_full = [x * 1024 for x in seq_lens]

# NVIDIA approximate values (extracted from reference graph - blue line)
# Graph shows ~200ms at 4k, ~400ms at 8k, scaling roughly linearly
nvidia_ttft = [200, 400, 800, 1600, 3200, 6000]  # ms (approximate)

# TT-Galaxy Baseline (Non-fused 7×8 4-links) - measured
baseline_ttft = [631.11, 1003.81, 1968.34, 4355.67, 10764.83, 29943.69]  # ms

# TT-Galaxy Fused (6×8/6×9 3-links) - measured
fused_ttft = [637.05, 977.19, 1899.16, 4206.46, 10378.19, 28917.57]  # ms

# ============================================================================
# PLOT 1: Full range (4k - 128k)
# ============================================================================
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12, 8))

# Plot lines
ax.plot(seq_lens_full, nvidia_ttft, "o-", color="#00BFFF", linewidth=2, markersize=8, label="NVIDIA")
ax.plot(
    seq_lens_full,
    baseline_ttft,
    "s-",
    color="#00FF7F",
    linewidth=2,
    markersize=8,
    label="TT-Galaxy Baseline (7×8, 4-links)",
)
ax.plot(
    seq_lens_full,
    fused_ttft,
    "^-",
    color="#FFA500",
    linewidth=2,
    markersize=8,
    label="TT-Galaxy Fused (6×8/6×9, 3-links)",
)

# Formatting
ax.set_xlabel("Input Sequence Length (ISL)", fontsize=14)
ax.set_ylabel("Time to First Token (ms)", fontsize=14)
ax.set_title("TTFT vs. Input Length (Concurrency=1, OSL=1A)\nLlama 70B on TT-Galaxy 6U", fontsize=16)

# Set x-axis ticks
ax.set_xticks(seq_lens_full)
ax.set_xticklabels(["4K", "8K", "16K", "32K", "64K", "128K"])

# Grid
ax.grid(True, alpha=0.3, linestyle="--")

# Legend
ax.legend(loc="upper left", fontsize=12)

# Add speedup annotations for fused vs baseline
for i, (seq, base, fused) in enumerate(zip(seq_lens_full, baseline_ttft, fused_ttft)):
    speedup = (base - fused) / base * 100
    if speedup > 0:
        ax.annotate(
            f"+{speedup:.1f}%",
            xy=(seq, fused),
            xytext=(seq + 2000, fused - 500),
            fontsize=9,
            color="#FFA500",
            arrowprops=dict(arrowstyle="->", color="#FFA500", lw=0.5),
        )

# Set y-axis to start from 0
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("/home/tvardhineni/tt-metal/ttft_comparison_plot.png", dpi=150, bbox_inches="tight")
plt.savefig("/home/tvardhineni/tt-metal/ttft_comparison_plot.pdf", bbox_inches="tight")
print("Saved: ttft_comparison_plot.png and ttft_comparison_plot.pdf")

# ============================================================================
# PLOT 2: Zoomed range (0 - 32k) with 4k gap on x-axis, 1000ms gap on y-axis
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Filter data for 0-32k range (indices 0-3: 4k, 8k, 16k, 32k)
seq_lens_32k = seq_lens_full[:4]  # [4096, 8192, 16384, 32768]
nvidia_32k = nvidia_ttft[:4]
baseline_32k = baseline_ttft[:4]
fused_32k = fused_ttft[:4]

# Plot lines
ax2.plot(seq_lens_32k, nvidia_32k, "o-", color="#00BFFF", linewidth=2, markersize=10, label="NVIDIA")
ax2.plot(
    seq_lens_32k,
    baseline_32k,
    "s-",
    color="#00FF7F",
    linewidth=2,
    markersize=10,
    label="TT-Galaxy Baseline (7×8, 4-links)",
)
ax2.plot(
    seq_lens_32k,
    fused_32k,
    "^-",
    color="#FFA500",
    linewidth=2,
    markersize=10,
    label="TT-Galaxy Fused (6×8/6×9, 3-links)",
)

# Formatting
ax2.set_xlabel("Input Sequence Length (ISL)", fontsize=14)
ax2.set_ylabel("Time to First Token (ms)", fontsize=14)
ax2.set_title("TTFT vs. Input Length (Concurrency=1, OSL=1A)\nLlama 70B on TT-Galaxy 6U (0-32K Range)", fontsize=16)

# Set x-axis: 0 to 32k with 4k gaps
ax2.set_xlim(0, 32768)
ax2.set_xticks(np.arange(0, 32768 + 1, 4096))
ax2.set_xticklabels(["0", "4K", "8K", "12K", "16K", "20K", "24K", "28K", "32K"])

# Set y-axis: 0 to 6000 with 1000ms gaps (like original plot)
ax2.set_ylim(0, 6000)
ax2.set_yticks(np.arange(0, 6001, 1000))

# Grid
ax2.grid(True, alpha=0.3, linestyle="--")

# Legend
ax2.legend(loc="upper left", fontsize=12)

# Add horizontal dashed lines at 2000 and 4000 (like original)
ax2.axhline(y=2000, color="white", linestyle="--", alpha=0.5, linewidth=1)
ax2.axhline(y=4000, color="white", linestyle="--", alpha=0.5, linewidth=1)
ax2.axhline(y=6000, color="white", linestyle="--", alpha=0.5, linewidth=1)

# Add speedup annotations
for i, (seq, base, fused) in enumerate(zip(seq_lens_32k, baseline_32k, fused_32k)):
    speedup = (base - fused) / base * 100
    if speedup > 0:
        ax2.annotate(
            f"+{speedup:.1f}%",
            xy=(seq, fused),
            xytext=(seq + 1500, fused + 200),
            fontsize=10,
            color="#FFA500",
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig("/home/tvardhineni/tt-metal/ttft_comparison_plot_32k.png", dpi=150, bbox_inches="tight")
plt.savefig("/home/tvardhineni/tt-metal/ttft_comparison_plot_32k.pdf", bbox_inches="tight")
print("Saved: ttft_comparison_plot_32k.png and ttft_comparison_plot_32k.pdf")

# Also print the data table
print("\n" + "=" * 80)
print("TTFT Comparison Table")
print("=" * 80)
print(f"{'Seq Len':<10} {'NVIDIA (ms)':<15} {'Baseline (ms)':<18} {'Fused (ms)':<15} {'Speedup':<10}")
print("-" * 80)
for i, seq in enumerate(seq_lens):
    speedup = (baseline_ttft[i] - fused_ttft[i]) / baseline_ttft[i] * 100
    print(f"{seq}K{'':<7} {nvidia_ttft[i]:<15.2f} {baseline_ttft[i]:<18.2f} {fused_ttft[i]:<15.2f} {speedup:+.2f}%")
print("=" * 80)
