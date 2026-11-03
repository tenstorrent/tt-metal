"""Plot iter51_power_cap_repro.py's run: per chip AICLK, TDP, TDC, temperature and the achieved matmul rate over time,
the phases shaded, the limits drawn, and the checkpoints marked (load start, the first sample under 1350 MHz, the
clock settled, the minimum clock). Usage: plot.py <out-prefix> -> <out-prefix>.png"""
import csv
import json
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

prefix = sys.argv[1]
run = json.load(open(prefix + ".json"))
import gzip  # noqa: E402
import os  # noqa: E402

_csv = prefix + "_telemetry.csv"
rows = list(csv.DictReader(open(_csv) if os.path.exists(_csv) else gzip.open(_csv + ".gz", "rt")))
t0 = run["phases"][0]["start"]
per = defaultdict(lambda: defaultdict(list))
for r in rows:
    c = int(r["chip"])
    per[c]["t"].append(float(r["t"]) - t0)
    for k in ("aiclk", "tdp", "tdc", "temp"):
        per[c][k].append(float(r[k]))
colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
labels = {"aiclk": "AICLK (MHz)", "tdp": "TDP (W)", "tdc": "TDC (A)", "temp": "ASIC temperature (C)"}
limits = {"aiclk": (1350, "1350 MHz nominal"), "tdp": (125, "125 W TDP limit"), "temp": (110, "110 C thermal limit")}
for ax, key in zip(axes[:4], ("aiclk", "tdp", "tdc", "temp")):
    for c in sorted(per):
        ax.plot(per[c]["t"], per[c][key], color=colors[c], lw=1.2, label=f"chip {c}")
    if key in limits:
        ax.axhline(limits[key][0], color="k", ls="--", lw=1, label=limits[key][1])
    ax.set_ylabel(labels[key])
    ax.grid(alpha=0.3)
ax = axes[4]
for name, color in (("HiFi4", "tab:purple"), ("HiFi2", "tab:brown")):
    b = [x for x in run["batches"] if x["phase"] == name]
    ax.plot(
        [x["t"] - t0 for x in b], [x["tflops_per_chip"] for x in b], color=color, lw=1.2, label=f"{name} matmul rate"
    )
ax.axhline(148.5, color="k", ls="--", lw=1, label="148.5 TFLOP/s: bf16 peak per chip at 1350 MHz")
ax.set_ylabel("achieved TFLOP/s per chip\n(slowest chip, host-timed)")
ax.set_xlabel("seconds since start")
ax.grid(alpha=0.3)
# phases shaded, checkpoints marked on the clock axis
shade = {"idle": "0.92", "HiFi4": "#e8d8f5", "cool-down": "0.92", "HiFi2": "#f3e3d3", "idle (end)": "0.92"}
for p in run["phases"]:
    for a in axes:
        a.axvspan(p["start"] - t0, p["end"] - t0, color=shade.get(p["name"], "0.95"), alpha=0.6, lw=0)
    axes[0].text((p["start"] + p["end"]) / 2 - t0, 1390, p["name"], ha="center", va="bottom", fontsize=10)
for p in run["phases"]:
    if p["name"] not in ("HiFi4", "HiFi2"):
        continue
    s, e = p["start"] - t0, p["end"] - t0
    axes[0].axvline(s, color="k", lw=0.8)
    for c in sorted(per):
        ts, ai = per[c]["t"], per[c]["aiclk"]
        inside = [(t, a) for t, a in zip(ts, ai) if s <= t <= e]
        if not inside:
            continue
        first_low = next(((t, a) for t, a in inside if a < 1350), None)
        settled = [(t, a) for t, a in inside if t >= s + 5]
        if first_low:
            axes[0].plot(first_low[0], first_low[1], "v", color=colors[c], ms=8)
        if settled:
            mean = sum(a for _, a in settled) / len(settled)
            axes[0].hlines(mean, s + 5, e, color=colors[c], ls=":", lw=1.5)
            axes[0].text(e, mean, f" {mean:.0f}", color=colors[c], fontsize=8, va="center")
        tmin, amin = min(inside, key=lambda x: x[1])
        axes[0].plot(tmin, amin, "o", color=colors[c], ms=5)
axes[0].plot([], [], "kv", label="first sample under 1350 MHz after load start")
axes[0].plot([], [], "k:", label="mean clock from 5 s into the phase")
axes[0].plot([], [], "ko", ms=5, label="minimum clock in the phase")
for a in axes:
    a.legend(loc="upper right", fontsize=8, ncol=2)
M, K, N = run["shape"]
fig.suptitle(
    f"Power-cap repro: [{M}, {K}] x [{N}, {K}]^T bf16 HiFi4 then HiFi2 on the 4-chip QB2 mesh (p300c, TDP 125 W per chip)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(prefix + ".png", dpi=80)
print("wrote", prefix + ".png")
