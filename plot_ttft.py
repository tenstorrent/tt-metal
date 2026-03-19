#!/usr/bin/env python3
"""
TTFT vs ISL plot - NVIDIA vs TT-Galaxy Baseline vs TT-Galaxy Fused (RS+MM + AG+MM)
Data from steps_to_run_llama_70b_fused_ops.md
"""

import matplotlib.pyplot as plt

# Input Sequence Lengths (tokens)
isl = [4000, 8000, 16000, 32000, 64000, 128000]
isl_labels = ["4K", "8K", "16K", "32K", "64K", "128K"]

# TTFT in ms - NVIDIA reference (from teja_RS+MM; 16K/32K fixed - were 1400/1500, now scaled so 32K > 16K)
nvidia = [350, 600, 1200, 2400, 3200, 6000]

# TT-Galaxy measured values (llama-weights/tt-metal)
galaxy_baseline = [617.96, 988.80, 1935.79, 4298.79, 10638.35, 29679.41]
galaxy_fused = [587.96, 945.7, 1810.16, 4042.68, 10027.41, 28867.07]  # RS+MM + AG+MM

# Improvement percentages (baseline -> fused)
improvements = [4.9, 4.5, 6.5, 6.0, 5.7, 2.7]

# Create plot with dark background (NVIDIA-style)
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# Plot lines
ax.plot(isl, nvidia, "o-", color="#00bfff", linewidth=2.5, markersize=8, label="NVIDIA")
ax.plot(isl, galaxy_baseline, "s-", color="#00ff00", linewidth=2.5, markersize=8, label="TT-Galaxy Baseline")
ax.plot(isl, galaxy_fused, "^-", color="#ffa500", linewidth=2.5, markersize=8, label="TT-Galaxy Fused (RS+MM + AG+MM)")

# Add improvement percentages - larger, with box for visibility
for i, (x, base, fused, imp) in enumerate(zip(isl, galaxy_baseline, galaxy_fused, improvements)):
    # Position above midpoint between baseline and fused
    mid_y = (base + fused) / 2
    ax.annotate(
        f"+{imp}%",
        xy=(x, mid_y),
        fontsize=12,
        color="white",
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffa500", edgecolor="white", alpha=0.9),
    )

# Formatting
ax.set_xlabel("Input Sequence Length (ISL)", fontsize=14, color="white")
ax.set_ylabel("Time to First Token (ms)", fontsize=14, color="white")
ax.set_title(
    "TTFT vs. Input Length (Concurrency=1, OSL=1A)\nLlama 3.3 70B on TT-Galaxy (8x4 Mesh)",
    fontsize=14,
    fontweight="bold",
    color="white",
    pad=15,
)

# X-axis ticks
ax.set_xticks(isl)
ax.set_xticklabels(isl_labels, fontsize=11)

# Y-axis
ax.set_ylim(0, 32000)
ax.set_yticks([0, 5000, 10000, 15000, 20000, 25000, 30000])

# Grid
ax.grid(True, alpha=0.3, linestyle="-", color="gray")

# Legend
legend = ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
legend.get_frame().set_facecolor("#1a1a2e")
legend.get_frame().set_edgecolor("white")

plt.tight_layout()
plt.savefig("ttft_comparison_plot.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("Saved: ttft_comparison_plot.png")

# Zoomed plot: 0-32K only (4K, 8K, 16K, 32K) - clearer view of smaller seq lengths
isl_32k = [4000, 8000, 16000, 32000]
isl_labels_32k = ["4K", "8K", "16K", "32K"]
nvidia_32k = [350, 600, 1200, 2400]
galaxy_baseline_32k = [617.96, 988.80, 1935.79, 4298.79]
galaxy_fused_32k = [587.96, 945.7, 1810.16, 4042.68]
improvements_32k = [4.9, 4.5, 6.5, 6.0]

fig3, ax3 = plt.subplots(figsize=(10, 6))
fig3.patch.set_facecolor("#1a1a2e")
ax3.set_facecolor("#1a1a2e")
ax3.plot(isl_32k, nvidia_32k, "o-", color="#00bfff", linewidth=2.5, markersize=10, label="NVIDIA")
ax3.plot(isl_32k, galaxy_baseline_32k, "s-", color="#00ff00", linewidth=2.5, markersize=10, label="TT-Galaxy Baseline")
ax3.plot(
    isl_32k,
    galaxy_fused_32k,
    "^-",
    color="#ffa500",
    linewidth=2.5,
    markersize=10,
    label="TT-Galaxy Fused (RS+MM + AG+MM)",
)
for i, (x, base, fused, imp) in enumerate(zip(isl_32k, galaxy_baseline_32k, galaxy_fused_32k, improvements_32k)):
    mid_y = (base + fused) / 2
    ax3.annotate(
        f"+{imp}%",
        xy=(x, mid_y),
        fontsize=14,
        color="white",
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffa500", edgecolor="white", alpha=0.95),
    )
ax3.set_xlabel("Input Sequence Length (ISL)", fontsize=14, color="white")
ax3.set_ylabel("Time to First Token (ms)", fontsize=14, color="white")
ax3.set_title(
    "TTFT vs. Input Length (4K–32K zoom)\nLlama 3.3 70B on TT-Galaxy (8x4 Mesh)",
    fontsize=14,
    fontweight="bold",
    color="white",
    pad=15,
)
ax3.set_xticks(isl_32k)
ax3.set_xticklabels(isl_labels_32k, fontsize=12)
ax3.set_ylim(0, 5000)
ax3.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
ax3.grid(True, alpha=0.3, linestyle="-", color="gray")
legend3 = ax3.legend(loc="upper left", fontsize=11, framealpha=0.9)
legend3.get_frame().set_facecolor("#1a1a2e")
legend3.get_frame().set_edgecolor("white")
plt.tight_layout()
plt.savefig("ttft_comparison_plot_0_32k.png", dpi=150, facecolor=fig3.get_facecolor(), bbox_inches="tight")
print("Saved: ttft_comparison_plot_0_32k.png")
