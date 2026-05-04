#!/usr/bin/env python3
"""Plot DiT single-pass performance experiments from autoresearch-results.tsv."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(HERE, "autoresearch-results.tsv")
OUT_PATH = os.path.join(HERE, "experiments_plot.png")

rows = []
with open(TSV_PATH) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("iteration"):
            continue
        parts = line.split("\t")
        it = int(parts[0])
        try:
            metric = float(parts[2])
        except ValueError:
            continue
        status = parts[6]
        desc = parts[7]
        rows.append((it, metric, status, desc))

baseline = rows[0][1]

kept = [(r[0], r[1], r[3]) for r in rows if r[2] in ("baseline", "keep")]
discarded = [(r[0], r[1], r[3]) for r in rows if r[2] == "discard"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle("DiT Single Pass Optimization", fontsize=15, fontweight="bold", y=0.98)

# --- Top panel: latency over experiment iterations ---
all_its = [r[0] for r in rows]
all_ms = [r[1] for r in rows]
all_status = [r[2] for r in rows]

for it, ms, status in zip(all_its, all_ms, all_status):
    color = "#2ecc71" if status in ("baseline", "keep") else "#e74c3c"
    marker = "o" if status in ("baseline", "keep") else "x"
    size = 90 if status in ("baseline", "keep") else 60
    ax1.scatter(it, ms, color=color, marker=marker, s=size, zorder=5)

kept_its = [r[0] for r in kept]
kept_ms = [r[1] for r in kept]
ax1.plot(kept_its, kept_ms, color="#2ecc71", linewidth=2, alpha=0.7, label="Kept (best so far)")

ax1.axhline(
    y=baseline,
    color="#95a5a6",
    linestyle="--",
    linewidth=1,
    alpha=0.6,
    label=f"Baseline ({baseline:.1f} ms)",
)
ax1.axhline(
    y=kept_ms[-1],
    color="#3498db",
    linestyle=":",
    linewidth=1,
    alpha=0.6,
    label=f"Current best ({kept_ms[-1]:.1f} ms)",
)

for it, ms, desc in kept:
    short = desc.split("—")[0].strip() if "—" in desc else desc[:40]
    if ms < baseline - 10:
        ax1.annotate(
            short,
            (it, ms),
            textcoords="offset points",
            xytext=(8, -15),
            fontsize=7,
            color="#2c3e50",
            ha="left",
        )

ax1.set_xlabel("Experiment #")
ax1.set_ylabel("Avg Latency (ms)")
ax1.set_title(
    f"Latency: {baseline:.1f} ms -> {kept_ms[-1]:.1f} ms  "
    f"({baseline - kept_ms[-1]:.1f} ms / {(baseline - kept_ms[-1]) / baseline * 100:.1f}% improvement)",
    fontsize=11,
)
ax1.legend(loc="upper right", fontsize=9)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1.grid(axis="y", alpha=0.3)
ax1.set_xlim(-0.5, max(all_its) + 0.5)

# --- Bottom panel: cumulative delta waterfall ---
deltas = []
labels = []
for idx_k in range(1, len(kept)):
    it, ms, desc = kept[idx_k]
    prev_ms = kept[idx_k - 1][1]
    d = prev_ms - ms
    short = desc.split("—")[0].strip() if "—" in desc else desc[:35]
    deltas.append(d)
    labels.append(f"#{it}: {short}")

colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]
ax2.barh(range(len(deltas)), deltas, color=colors, edgecolor="white", linewidth=0.5)
ax2.set_yticks(range(len(deltas)))
ax2.set_yticklabels(labels, fontsize=8)
ax2.set_xlabel("Latency Reduction (ms)")
ax2.set_title("Per-Experiment Improvement (kept only)", fontsize=11)
ax2.invert_yaxis()
ax2.grid(axis="x", alpha=0.3)

for i, d in enumerate(deltas):
    ax2.text(d + 0.5, i, f"{d:.1f} ms", va="center", fontsize=8, color="#2c3e50")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved plot to {OUT_PATH}")
