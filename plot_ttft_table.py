#!/usr/bin/env python3
"""
Generate TTFT comparison table image for screenshot.
ISL | TTFT Baseline | TTFT Fused | 4K-128K, with 128K note (only AG+MM)
"""

import matplotlib.pyplot as plt

# Data from steps_to_run_llama_70b_fused_ops.md
isl_labels = ["4K", "8K", "16K", "32K", "64K", "128K"]
baseline = [617.96, 988.80, 1935.79, 4298.79, 10638.35, 29679.41]
fused = [587.96, 945.7, 1810.16, 4042.68, 10027.41, 28867.07]

# Format for display
baseline_str = [f"{x:.2f}" for x in baseline]
fused_str = [f"{x:.2f}" for x in fused]

# Add note for 128K
fused_str[-1] = "28867.07*"

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

table_data = [
    ["ISL", "TTFT Baseline (ms)", "TTFT Fused (ms)"],
    *[[isl_labels[i], baseline_str[i], fused_str[i]] for i in range(6)],
]

table = ax.table(
    cellText=table_data,
    loc="center",
    cellLoc="center",
    colWidths=[0.2, 0.4, 0.4],
)
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.2, 2.5)

# Style header row
for j in range(3):
    table[(0, j)].set_facecolor("#2d5a87")
    table[(0, j)].set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, 7):
    for j in range(3):
        table[(i, j)].set_facecolor("#e8f4f8" if i % 2 == 0 else "white")
        if i == 6:  # 128K row
            table[(i, j)].set_facecolor("#fff3e0")  # light orange for note

# Add footnote
footnote = "* 128K: Only AG+MM fused (RS+MM disabled due to DRAM OOM)"
fig.text(0.5, 0.02, footnote, ha="center", fontsize=11, style="italic", color="#555")

plt.title(
    "Llama 3.3 70B TTFT: Baseline vs Fused (RS+MM + AG+MM)\nTT-Galaxy 8×4 Mesh", fontsize=14, fontweight="bold", pad=20
)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig("ttft_table_screenshot.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: ttft_table_screenshot.png")
