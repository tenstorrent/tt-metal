#!/usr/bin/env python3
"""Build REAP_FULL_SWEEP_COMPARE.md + PNGs from old_baseline_inventory + reap_full_new_summary."""
from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent
ISLS = [128, 512, 1024, 2048, 4096]
BATCHES = [4, 8, 16, 32]


def _f(x: str | None):
    if x is None or str(x).strip() == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def load_old() -> dict[tuple[int, int], dict]:
    path = OUT / "old_baseline_inventory.csv"
    by = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            isl, b = int(r["isl"]), int(r["batch"])
            by[(isl, b)] = {
                "prefill_s": _f(r.get("old_prefill_s")),
                "source": (r.get("source") or "").strip(),
                "note": (r.get("date_or_note") or "").strip(),
            }
    return by


def load_new() -> dict[tuple[int, int], dict]:
    path = OUT / "reap_full_new_summary.csv"
    by = {}
    if not path.exists():
        return by
    with open(path) as f:
        for r in csv.DictReader(f):
            isl, b = int(r["isl"]), int(r["batch"])
            by[(isl, b)] = {
                "prefill_s": _f(r.get("prefill_s")),
                "status": (r.get("status") or "").strip(),
                "decode_mean_ms": _f(r.get("decode_mean_ms")),
                "detail": (r.get("detail") or "").strip(),
                "T": int(r["T"]) if r.get("T") else isl * b,
            }
    return by


def speedup(old, new):
    if old is None or new is None or new <= 0:
        return None
    return old / new


def write_md(old, new):
    lines = []
    lines.append("# REAP full ISL×batch sweep — old vs new (chunked)")
    lines.append("")
    lines.append("**Mesh:** 8×4 · **Model:** `cerebras/GLM-4.7-REAP-218B-A32B`")
    lines.append("**New path:** batched + `PCM=1` / `CHUNK=4096` + `BATCHED_PREFILL_MAX_TOKENS=131072`")
    lines.append("**Old path:** pre-chunk PCM0/CHUNK0 (or Jul17 matrix equivalent)")
    lines.append("")
    lines.append("Artifacts:")
    lines.append(f"- New CSV: `{OUT / 'reap_full_new_summary.csv'}`")
    lines.append(f"- Old inventory: `{OUT / 'old_baseline_inventory.csv'}`")
    lines.append("")
    lines.append("## Prefill seconds (old vs new) + speedup")
    lines.append("")
    lines.append("| ISL | B | T | Old prefill_s | New prefill_s | Speedup | New status | Old source |")
    lines.append("|----:|--:|--:|--------------:|--------------:|--------:|:-----------|:-----------|")

    ok_n = fail_n = 0
    pairs = []
    for isl in ISLS:
        for b in BATCHES:
            T = isl * b
            o = old.get((isl, b), {})
            n = new.get((isl, b), {})
            op = o.get("prefill_s")
            np_ = n.get("prefill_s")
            st = n.get("status") or "missing"
            if st == "ok":
                ok_n += 1
            elif st != "missing":
                fail_n += 1
            sp = speedup(op, np_)
            op_s = f"{op:.3f}" if op is not None else "n/a"
            np_s = f"{np_:.3f}" if np_ is not None else ("—" if st == "missing" else "—")
            sp_s = f"**{sp:.2f}×**" if sp is not None else "—"
            src = o.get("source") or "n/a"
            if src and len(src) > 48:
                src = src.split("/")[-1]
            lines.append(f"| {isl} | {b} | {T} | {op_s} | {np_s} | {sp_s} | {st} | {src} |")
            if sp is not None:
                pairs.append((isl, b, T, op, np_, sp))

    lines.append("")
    lines.append(f"**New cells:** {ok_n} ok · {fail_n} fail/timeout/OOM · {20 - ok_n - fail_n} missing")
    lines.append("")
    if pairs:
        lines.append("## Speedup highlights (both arms present)")
        lines.append("")
        pairs_sorted = sorted(pairs, key=lambda x: -x[5])
        lines.append("| ISL×B | T | Old | New | Speedup |")
        lines.append("|------:|--:|----:|----:|--------:|")
        for isl, b, T, op, np_, sp in pairs_sorted:
            lines.append(f"| {isl}×{b} | {T} | {op:.1f} | {np_:.1f} | **{sp:.2f}×** |")
        lines.append("")
        median = sorted(p[5] for p in pairs)[len(pairs) // 2]
        geomean = 1.0
        for p in pairs:
            geomean *= p[5]
        geomean **= 1.0 / len(pairs)
        lines.append(f"- Paired cells: **{len(pairs)}**")
        lines.append(f"- Median speedup: **{median:.2f}×**")
        lines.append(f"- Geo-mean speedup: **{geomean:.2f}×**")
        lines.append("")

    lines.append("## Old data sources")
    lines.append("")
    lines.append("| Preference | Source | Notes |")
    lines.append("|------------|--------|-------|")
    lines.append("| Primary (paired) | `validation_summary.csv` | 128×4 & 512×8 PCM0/0 on 2026-07-21 |")
    lines.append("| Bulk old matrix | `batched_prefill_smoke/matrix_summary.csv` | Jul17 pre-chunk REAP batched |")
    lines.append(
        "| Not used as old batched | `g1_multilink_4_ring_isl_sweep/` | Different harness / closer to serial-ish times |"
    )
    lines.append("")
    lines.append("Missing old cells left as **n/a** — not invented.")
    lines.append("")
    lines.append("## Graphs")
    lines.append("")
    lines.append("- `reap_full_prefill_vs_isl_by_batch.png` — old vs new lines per batch")
    lines.append("- `reap_full_speedup_heatmap.png` — speedup where both exist")
    lines.append("- `reap_full_prefill_vs_T.png` — prefill vs aggregate tokens T")
    lines.append("")

    (OUT / "REAP_FULL_SWEEP_COMPARE.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT / 'REAP_FULL_SWEEP_COMPARE.md'}")
    return pairs


def plot(old, new):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    # Prefill vs ISL for each batch
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()
    for ax, b in zip(axes, BATCHES):
        old_xs, old_ys, new_xs, new_ys = [], [], [], []
        for isl in ISLS:
            o = old.get((isl, b), {}).get("prefill_s")
            n = new.get((isl, b), {})
            np_ = n.get("prefill_s") if n.get("status") == "ok" else None
            if o is not None:
                old_xs.append(isl)
                old_ys.append(o)
            if np_ is not None:
                new_xs.append(isl)
                new_ys.append(np_)
        if old_xs:
            ax.plot(old_xs, old_ys, "o--", color="#c44e52", label="old (PCM0)", linewidth=1.5)
        if new_xs:
            ax.plot(new_xs, new_ys, "s-", color="#4c72b0", label="new (PCM1/4096)", linewidth=1.8)
        ax.set_title(f"batch={b}")
        ax.set_xlabel("ISL")
        ax.set_ylabel("prefill_s")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("REAP batched prefill: old vs new (chunked MoE)", fontsize=13)
    fig.tight_layout()
    p1 = OUT / "reap_full_prefill_vs_isl_by_batch.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p1}")

    # Speedup heatmap
    M = np.full((len(ISLS), len(BATCHES)), np.nan)
    for i, isl in enumerate(ISLS):
        for j, b in enumerate(BATCHES):
            op = old.get((isl, b), {}).get("prefill_s")
            n = new.get((isl, b), {})
            np_ = n.get("prefill_s") if n.get("status") == "ok" else None
            sp = speedup(op, np_)
            if sp is not None:
                M[i, j] = sp
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(BATCHES)))
    ax.set_xticklabels(BATCHES)
    ax.set_yticks(range(len(ISLS)))
    ax.set_yticklabels([str(x) for x in ISLS])
    ax.set_xlabel("Batch")
    ax.set_ylabel("ISL")
    ax.set_title("Speedup (old/new) — blank = missing arm")
    for i in range(len(ISLS)):
        for j in range(len(BATCHES)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}×", ha="center", va="center", color="white", fontsize=9)
    plt.colorbar(im, ax=ax, label="speedup")
    fig.tight_layout()
    p2 = OUT / "reap_full_speedup_heatmap.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p2}")

    # Prefill vs T
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, data, style, color in (
        ("old", old, "o--", "#c44e52"),
        ("new", new, "s-", "#4c72b0"),
    ):
        xs, ys = [], []
        for isl in ISLS:
            for b in BATCHES:
                T = isl * b
                if label == "old":
                    v = data.get((isl, b), {}).get("prefill_s")
                else:
                    cell = data.get((isl, b), {})
                    v = cell.get("prefill_s") if cell.get("status") == "ok" else None
                if v is not None:
                    xs.append(T)
                    ys.append(v)
        if xs:
            order = sorted(range(len(xs)), key=lambda k: xs[k])
            ax.plot([xs[k] for k in order], [ys[k] for k in order], style, color=color, label=label, linewidth=1.6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("T = ISL × batch")
    ax.set_ylabel("prefill_s")
    ax.set_title("REAP batched prefill vs aggregate tokens")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p3 = OUT / "reap_full_prefill_vs_T.png"
    fig.savefig(p3, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p3}")


def main():
    old = load_old()
    new = load_new()
    write_md(old, new)
    plot(old, new)


if __name__ == "__main__":
    main()
