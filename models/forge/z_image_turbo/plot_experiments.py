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
kept_its = [r[0] for r in kept]
kept_ms = [r[1] for r in kept]
best = kept_ms[-1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle(
    f"DiT Single Pass Optimization — {baseline:.0f} ms → {best:.0f} ms"
    f"  (−{baseline - best:.0f} ms / {(baseline - best) / baseline * 100:.1f}%)",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

# ── Top panel: latency trajectory ──────────────────────────────────────────────
for r in rows:
    it, ms, status = r[0], r[1], r[2]
    is_keep = status in ("baseline", "keep")
    ax1.scatter(
        it,
        ms,
        color="#2ecc71" if is_keep else "#e74c3c",
        marker="o" if is_keep else "x",
        s=80 if is_keep else 50,
        zorder=5,
        linewidths=1.5,
    )

ax1.plot(kept_its, kept_ms, color="#2ecc71", linewidth=2.5, alpha=0.7, zorder=4)
ax1.axhline(y=baseline, color="#bdc3c7", linestyle="--", linewidth=1, alpha=0.5)
ax1.axhline(y=best, color="#3498db", linestyle=":", linewidth=1, alpha=0.5)

# Only annotate major milestones (>5 ms improvement from previous kept)
milestones = {
    0: "Baseline",
    4: "Metal trace",
    14: "BF16 RoPE",
    15: "Cache freqs",
    27: "CCL 4-link",
    45: "L1 norms",
    50: "L1 reshapes",
}
offsets = {
    0: (10, 8),
    4: (10, 8),
    14: (10, 10),
    15: (10, -18),
    27: (10, 8),
    45: (10, 8),
    50: (10, -18),
}
for it, ms, desc in kept:
    if it in milestones:
        label = f"{milestones[it]}\n{ms:.0f} ms"
        ox, oy = offsets[it]
        ax1.annotate(
            label,
            (it, ms),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8,
            fontweight="bold",
            color="#2c3e50",
            arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bdc3c7", alpha=0.9),
        )

ax1.set_xlabel("Experiment #", fontsize=11)
ax1.set_ylabel("Avg Latency (ms)", fontsize=11)
ax1.legend(
    ["Kept trajectory", "Baseline", "Current best"],
    loc="upper right",
    fontsize=9,
)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1.grid(axis="y", alpha=0.2)
ax1.set_xlim(-1, max(r[0] for r in rows) + 2)
ax1.set_ylim(min(kept_ms) - 15, baseline + 15)

# ── Bottom panel: per-experiment improvement waterfall ─────────────────────────
deltas = []
labels = []
for i in range(1, len(kept)):
    it, ms, desc = kept[i]
    prev_ms = kept[i - 1][1]
    d = prev_ms - ms
    short = desc.split("—")[0].strip() if "—" in desc else desc[:30]
    if d >= 1.0:
        deltas.append(d)
        labels.append(f"#{it}: {short}")

colors = ["#27ae60" if d >= 5 else "#2ecc71" for d in deltas]
bars = ax2.barh(range(len(deltas)), deltas, color=colors, edgecolor="white", height=0.7)
ax2.set_yticks(range(len(deltas)))
ax2.set_yticklabels(labels, fontsize=8)
ax2.set_xlabel("Latency Reduction (ms)", fontsize=10)
ax2.set_title("Per-Experiment Improvement (kept, ≥1 ms only)", fontsize=11)
ax2.invert_yaxis()
ax2.grid(axis="x", alpha=0.2)

for i, d in enumerate(deltas):
    ax2.text(d + 0.8, i, f"{d:.1f}", va="center", fontsize=8, fontweight="bold", color="#2c3e50")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved plot to {OUT_PATH}")
