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

# Short names for all kept experiments
SHORT_NAMES = {
    0: "Baseline",
    1: "MM blocking",
    2: "HiFi2+L1acc",
    3: "math_approx norms",
    4: "Metal trace",
    7: "BFP8 MLP",
    13: "Remove blocking",
    14: "BF16 RoPE",
    15: "Cache freqs",
    19: "Fused w1+w3",
    22: "nlp_qkv_heads",
    23: "fp32acc off norms",
    24: "BFP8 adaLN",
    25: "CCL 2-link",
    26: "CCL 3-link",
    27: "CCL 4-link",
    36: "AG no hyperparams",
    45: "L1 norms",
    46: "L1 eltwise",
    47: "L1 SDPA",
    50: "L1 reshapes",
}

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

fig.suptitle(
    f"DiT Single Pass Optimization — {baseline:.0f} ms → {best:.0f} ms"
    f"  (−{baseline - best:.0f} ms / {(baseline - best) / baseline * 100:.1f}%)",
    fontsize=14,
    fontweight="bold",
    y=0.97,
)

# ── Top panel ──────────────────────────────────────────────────────────────────
for r in rows:
    it, ms, status = r[0], r[1], r[2]
    is_keep = status in ("baseline", "keep")
    ax1.scatter(
        it,
        ms,
        color="#2ecc71" if is_keep else "#e74c3c",
        marker="o" if is_keep else "x",
        s=80 if is_keep else 40,
        zorder=5,
        linewidths=1.5,
        alpha=1.0 if is_keep else 0.5,
    )

ax1.plot(kept_its, kept_ms, color="#2ecc71", linewidth=2.5, alpha=0.7, zorder=4)
ax1.axhline(y=baseline, color="#bdc3c7", linestyle="--", linewidth=1, alpha=0.4)
ax1.axhline(y=best, color="#3498db", linestyle=":", linewidth=1, alpha=0.4)

# Label ALL kept experiments — alternate above/below to reduce overlap
for idx, (it, ms, desc) in enumerate(kept):
    name = SHORT_NAMES.get(it, f"#{it}")
    above = idx % 2 == 0
    oy = 14 if above else -16
    va = "bottom" if above else "top"

    ax1.annotate(
        f"{name} ({ms:.0f})",
        (it, ms),
        textcoords="offset points",
        xytext=(0, oy),
        fontsize=6.5,
        ha="center",
        va=va,
        color="#2c3e50",
        arrowprops=dict(arrowstyle="-", color="#bdc3c7", lw=0.5),
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#ddd", alpha=0.85, lw=0.5),
    )

ax1.set_xlabel("Experiment #", fontsize=11)
ax1.set_ylabel("Avg Latency (ms)", fontsize=11)
ax1.legend(["Kept trajectory", "Baseline", "Current best"], loc="upper right", fontsize=9)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1.grid(axis="y", alpha=0.2)
ax1.set_xlim(-2, max(r[0] for r in rows) + 3)
ax1.set_ylim(min(kept_ms) - 25, baseline + 25)

# ── Bottom panel: waterfall ────────────────────────────────────────────────────
deltas = []
labels = []
for i in range(1, len(kept)):
    it, ms, desc = kept[i]
    prev_ms = kept[i - 1][1]
    d = prev_ms - ms
    full = desc.split("—")[0].strip() if "—" in desc else desc
    deltas.append(d)
    labels.append(f"#{it}: {full}")

colors = ["#1a7a3a" if d >= 10 else "#27ae60" if d >= 3 else "#2ecc71" for d in deltas]
ax2.barh(range(len(deltas)), deltas, color=colors, edgecolor="white", height=0.7)
ax2.set_yticks(range(len(deltas)))
ax2.set_yticklabels(labels, fontsize=7)
ax2.set_xlabel("Latency Reduction (ms)", fontsize=10)
ax2.set_title("Per-Experiment Improvement (all kept)", fontsize=11)
ax2.invert_yaxis()
ax2.grid(axis="x", alpha=0.2)

for i, d in enumerate(deltas):
    ax2.text(d + 0.5, i, f"{d:.1f}", va="center", fontsize=7, fontweight="bold", color="#2c3e50")

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved plot to {OUT_PATH}")
