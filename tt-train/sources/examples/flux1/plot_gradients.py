"""Plot gradient CosSim from gradients_comparison.csv.

Output:
  gradients_plot.png       — all params in one row with names
  gradients_by_group.png   — split by logical group, full width rows
"""
import csv
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def layer_sort_key(name):
    parts = name.split(".")
    if parts[0] == "time_text_embed":
        return (0, 0, name)
    if parts[0] == "x_embedder":
        return (1, 0, name)
    if parts[0] == "context_embedder":
        return (2, 0, name)
    m = re.match(r"transformer_blocks\.(\d+)\.", name)
    if m:
        return (3, int(m.group(1)), name)
    m = re.match(r"single_transformer_blocks\.(\d+)\.", name)
    if m:
        return (4, int(m.group(1)), name)
    if parts[0] in ("norm_out", "proj_out"):
        return (5, 0, name)
    return (6, 0, name)

def cos_color(c):
    if c > 0.95: return "green"
    if c > 0.5:  return "orange"
    return "red"

def get_group(name):
    """Merge into logical groups: combine bias+weight, spatial+context."""
    m = re.match(r"(?:single_)?transformer_blocks\.\d+\.(.*)", name)
    if not m:
        return "global (embedders, norm_out, proj_out)"
    suf = m.group(1)

    # Strip .bias / .weight suffix to merge them
    base = re.sub(r"\.(bias|weight)$", "", suf)

    GROUP_MAP = {
        "attn.add_qkv_proj":          "attn QKV (spatial + context)",
        "attn.to_qkv":                "attn QKV (spatial + context)",
        "attn.norm_added_k":          "attn norm Q/K (spatial + context)",
        "attn.norm_added_q":          "attn norm Q/K (spatial + context)",
        "attn.norm_k":                "attn norm Q/K (spatial + context)",
        "attn.norm_q":                "attn norm Q/K (spatial + context)",
        "attn.to_add_out":            "attn output proj (spatial + context)",
        "attn.to_out.0":              "attn output proj (spatial + context)",
        "ff.net.0.proj":              "feedforward (spatial + context)",
        "ff.net.2":                   "feedforward (spatial + context)",
        "ff_context.net.0.proj":      "feedforward (spatial + context)",
        "ff_context.net.2":           "feedforward (spatial + context)",
        "norm1.linear":               "AdaLN modulation (spatial + context)",
        "norm1_context.linear":       "AdaLN modulation (spatial + context)",
        "norm.linear":                "AdaLN modulation (single blocks)",
        "proj_mlp":                   "proj_mlp + proj_out (single blocks)",
        "proj_out":                   "proj_mlp + proj_out (single blocks)",
        "time_embed":                 "time_embed (single blocks)",
    }
    return GROUP_MAP.get(base, base)

# ── Load & sort ──
rows = []
with open("gradients_comparison.csv") as f:
    for r in csv.DictReader(f):
        rows.append(r)
rows.sort(key=lambda r: layer_sort_key(r["Parameter"]))

names = [r["Parameter"] for r in rows]
cos_sim = [float(r["CosSim"]) for r in rows]

# ═══════════════════════════════════════════════════════════════════
# Plot 1: All params in one line
# ═══════════════════════════════════════════════════════════════════
hf_norm = [float(r["HF Norm"]) for r in rows]
tt_norm = [float(r["TTML Norm"]) for r in rows]
norm_ratio = [t / (h + 1e-12) for h, t in zip(hf_norm, tt_norm)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(48, 10), gridspec_kw={"height_ratios": [2, 1]})

ax1.bar(range(len(cos_sim)), cos_sim, color=[cos_color(c) for c in cos_sim], width=1.0, edgecolor="none")
ax1.axhline(y=0.95, color="green", linestyle="--", alpha=0.5, linewidth=0.8)
ax1.axhline(y=0.0, color="black", linewidth=0.5)
ax1.set_ylabel("CosSim", fontsize=12)
ax1.set_title("Per-parameter gradient CosSim (HF vs TTML) — all params in layer order", fontsize=14)
ax1.set_xlim(-1, len(cos_sim))
ax1.set_ylim(-0.8, 1.05)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=90, fontsize=4, ha="center")

ax2.bar(range(len(norm_ratio)), norm_ratio, color="steelblue", width=1.0, edgecolor="none")
ax2.axhline(y=1.0, color="black", linewidth=1)
ax2.set_ylabel("TTML / HF Norm", fontsize=12)
ax2.set_title("Gradient norm ratio", fontsize=14)
ax2.set_xlim(-1, len(norm_ratio))
ax2.set_ylim(0, min(max(norm_ratio) * 1.1, 4.0))
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels(names, rotation=90, fontsize=4, ha="center")

plt.tight_layout()
plt.savefig("gradients_plot.png", dpi=250)
print("Saved gradients_plot.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# Plot 2: Grouped — one full-width row per logical group
# ═══════════════════════════════════════════════════════════════════
group_order = []
group_data = {}
for n, c in zip(names, cos_sim):
    g = get_group(n)
    if g not in group_data:
        group_order.append(g)
        group_data[g] = []
    group_data[g].append((n, c))

n_groups = len(group_order)
fig, axes = plt.subplots(n_groups, 1, figsize=(36, 3.0 * n_groups), squeeze=False)

for i, g in enumerate(group_order):
    ax = axes[i][0]
    items = group_data[g]
    xs = range(len(items))
    ys = [c for _, c in items]
    labels = [n for n, _ in items]

    ax.bar(xs, ys, color=[cos_color(y) for y in ys], width=0.9, edgecolor="none")
    ax.axhline(y=0.0, color="black", linewidth=0.4)
    ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_ylim(-0.8, 1.1)
    ax.set_xlim(-0.5, len(items) - 0.5)
    ax.set_ylabel("CosSim", fontsize=8)

    avg = sum(ys) / len(ys)
    ax.set_title(f"{g}  (n={len(items)}, avg={avg:.3f})", fontsize=12, fontweight="bold", loc="left", pad=2)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=90, fontsize=4.5, ha="center")
    ax.tick_params(axis="y", labelsize=7)

plt.tight_layout(h_pad=0.5)
plt.savefig("gradients_by_group.png", dpi=200)
print("Saved gradients_by_group.png")
plt.close()

# ── Summary ──
print(f"\nTotal params: {len(cos_sim)}")
print(f"Avg CosSim:   {sum(cos_sim)/len(cos_sim):.4f}")
for g in group_order:
    items = group_data[g]
    avg = sum(c for _, c in items) / len(items)
    print(f"  {g:>50s}  n={len(items):>3d}  avg={avg:.4f}")
