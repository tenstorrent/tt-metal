#!/usr/bin/env python3
"""frontier_scatter.py — merge frontier_sweep shard CSVs into ULP-vs-runtime
Pareto scatters per activation, with optional TTNN reference.

Usage:
  frontier_scatter.py frontier_chip*.csv [--ttnn ttnn_ref.csv] [--outdir plots]

Input rows (from frontier_sweep.sh):
  csv,activation,method,degree,segments,precision,bf16_maxulp,runtime_us,compiles,range
  or an equivalent schema with dtype instead of precision.
TTNN ref (optional): activation,ttnn_maxulp,ttnn_us[,dtype|precision]

Per activation it draws Tracy runtime(x) vs Goldberg ULP(y), colored by
method/config family. Did-not-compile rows are omitted, the TTNN op is a red
star when a reference is supplied, and the "we win" quadrant below+left of TTNN
is shaded. If there is no dtype column, the precision column is used as the
dtype group. Inputs with neither dtype nor precision are treated as one BF16
sweep.
"""
import argparse
import csv as _csv
import glob
import os
import re
import sys
from collections import defaultdict

try:
    from .style import apply_tufte_style
except ImportError:
    from style import apply_tufte_style


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def row_dtype(row):
    dtype = (row.get("dtype") or row.get("precision") or "").strip()
    return dtype or None


def has_dtype_or_precision_column(path):
    with open(path) as f:
        reader = _csv.reader(f)
        header = next(reader, [])
    return "dtype" in header or "precision" in header


def config_family(row):
    method = (row.get("method") or "unknown").strip() or "unknown"
    degree = (row.get("degree") or "").strip()
    segments = (row.get("segments") or "").strip()
    if degree and segments:
        return f"{method} {degree}/s{segments}"
    return method


def load(paths):
    rows = []
    explicit_dtype = any(has_dtype_or_precision_column(p) for p in paths)
    precisions = set()
    for p in paths:
        with open(p) as f:
            for r in _csv.DictReader(f):
                r["_ulp"] = fnum(r.get("maxulp", r.get("bf16_maxulp")))
                r["_us"] = fnum(r.get("runtime_us"))
                r["_ok"] = str(r.get("compiles", "0")).strip() in ("1", "True", "true")
                precision = (r.get("precision") or "").strip().lower()
                if precision:
                    precisions.add(precision)
                r["_dtype"] = row_dtype(r)
                r["_family"] = config_family(r)
                rows.append(r)
    if not explicit_dtype and precisions - {"bf16"}:
        raise SystemExit(
            "frontier_scatter: input rows do not have a dtype column and are not a BF16-only sweep. "
            f"Found precision values: {', '.join(sorted(precisions))}"
        )
    return rows, explicit_dtype


def load_ttnn(path):
    ttnn = {}
    if not path:
        return ttnn, False
    if not os.path.exists(path):
        raise SystemExit(f"frontier_scatter: TTNN reference not found: {path}")
    has_typed_rows = False
    with open(path) as f:
        for r in _csv.DictReader(f):
            act = (r.get("activation") or "").strip()
            if not act:
                continue
            dtype = row_dtype(r)
            has_typed_rows = has_typed_rows or dtype is not None
            us = fnum(r.get("ttnn_us"))
            ulp = fnum(r.get("ttnn_maxulp"))
            if us is None or ulp is None:
                continue
            ttnn[(act, dtype)] = (us, ulp)
            if dtype is None:
                ttnn[(act, None)] = (us, ulp)
    return ttnn, has_typed_rows


def load_picked(path):
    picked = {}
    if not path:
        return picked
    if not os.path.exists(path):
        raise SystemExit(f"frontier_scatter: picked manifest not found: {path}")
    with open(path) as f:
        for r in _csv.DictReader(f):
            act = (r.get("activation") or "").strip()
            method = (r.get("method") or "").strip()
            role = (r.get("role") or "").strip()
            status = (r.get("status") or "").strip()
            if not act or method == "ttnn" or role == "ttnn" or status == "ttnn_ref":
                continue
            dtype = row_dtype(r)
            picked[(act, dtype)] = r
    return picked


def picked_for(picked, act, dtype):
    return picked.get((act, dtype)) or picked.get((act, None))


def rows_match_pick(row, pick):
    for key in ("csv", "method", "degree", "segments"):
        rv = (row.get(key) or "").strip()
        pv = (pick.get(key) or "").strip()
        if pv and rv != pv:
            return False
    return True


def picked_point(comp, pick):
    if not pick:
        return None
    matches = [r for r in comp if rows_match_pick(r, pick)]
    if matches:
        return min(matches, key=lambda r: abs(r["_us"] - (fnum(pick.get("runtime_us")) or r["_us"])))
    us = fnum(pick.get("runtime_us"))
    ulp = fnum(pick.get("max_ulp") or pick.get("bf16_maxulp"))
    if us is None or ulp is None:
        return None
    return {"_us": us, "_ulp": ulp}


def ttnn_for(ttnn, act, dtype, has_typed_rows):
    if dtype is None:
        return ttnn.get((act, None)) or (None, None)
    if has_typed_rows:
        return ttnn.get((act, dtype)) or (None, None)
    # Untyped refs in this tree are BF16 legacy baselines. Do not overlay them
    # onto typed non-BF16 frontier plots.
    if dtype.lower() == "bf16":
        return ttnn.get((act, None)) or (None, None)
    return (None, None)


def dtype_label(dtype):
    return (dtype or "bf16").upper()


def shade_ttnn_win_region(ax, tus, tulp):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = max(0.0, xmin)
    y0 = max(0.0, ymin)
    if tus <= x0 or tulp <= y0:
        return False
    ax.fill_between([x0, tus], [y0, y0], [tulp, tulp], alpha=0.08, color="#777777", zorder=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return True


def legend_outside_two_rows(ax, handles=None, labels=None):
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label and label not in seen:
            seen[label] = handle
    if not seen:
        return
    ncols = max(1, (len(seen) + 1) // 2)
    ax.legend(
        seen.values(),
        seen.keys(),
        ncols=ncols,
        mode="expand",
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.22),
        borderaxespad=0.0,
    )


def load_tier_csvs(specs):
    tiers = []
    for spec in specs or []:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = os.path.basename(path).replace("_native_vs_embedded.csv", "")
        if not os.path.exists(path):
            raise SystemExit(f"frontier_scatter: tier CSV not found: {path}")
        rows = []
        with open(path) as f:
            for r in _csv.DictReader(f):
                if (r.get("precision") or "").lower() != "bf16":
                    continue
                if (r.get("status") or "") != "pass":
                    continue
                rows.append(r)
        tiers.append((label, rows))
    return tiers


def pareto(points):
    """points: list of (us, ulp, row). Return the non-dominated (lower-left) set."""
    pts = sorted([p for p in points if p[0] is not None and p[1] is not None], key=lambda p: (p[0], p[1]))
    front, best_ulp = [], float("inf")
    for us, ulp, row in pts:
        if ulp < best_ulp - 1e-12:
            front.append((us, ulp, row))
            best_ulp = ulp
    return front


def add_one_ulp_inset(ax, comp, front, tus, tulp, method_color, method_marker, pick=None):
    low_ulp = [r for r in comp if r["_us"] is not None and r["_ulp"] is not None and 0 <= r["_ulp"] <= 1.0]
    ttnn_in_band = tus is not None and tulp is not None and 0 <= tulp <= 1.0
    picked_in_band = (
        pick is not None and pick["_us"] is not None and pick["_ulp"] is not None and 0 <= pick["_ulp"] <= 1.0
    )
    if not low_ulp and not ttnn_in_band and not picked_in_band:
        return

    xs = [r["_us"] for r in low_ulp]
    if ttnn_in_band:
        xs.append(tus)
    if picked_in_band:
        xs.append(pick["_us"])
    if not xs:
        return

    q90 = percentile(xs, 0.90) or max(xs)
    x_max = max(q90, tus * 1.15 if tus else q90)
    x_max = max(x_max, min(xs) + 0.25)

    inset = ax.inset_axes([0.58, 0.52, 0.34, 0.28])
    inset.set_facecolor("#F7F7F7")
    for r in low_ulp:
        if r["_us"] > x_max:
            continue
        method = r.get("method") or "unknown"
        marker = method_marker.get(method, "o")
        inset.scatter(
            r["_us"],
            r["_ulp"],
            c=method_color.get(method, "gray"),
            marker=marker,
            s=11,
            alpha=0.75,
            edgecolors="none",
            clip_on=False,
        )

    front_low = [(us, ulp) for us, ulp, _ in front if 0 <= ulp <= 1.0 and us <= x_max]
    if front_low:
        inset.plot(
            [p[0] for p in front_low],
            [p[1] for p in front_low],
            color="#333333",
            ls="--",
            lw=0.7,
            alpha=0.65,
            clip_on=False,
        )

    if ttnn_in_band and tus <= x_max:
        inset.scatter([tus], [tulp], c="#D62728", s=64, marker="*", zorder=5, clip_on=False)
        inset.axhline(tulp, color="#D62728", ls=":", lw=0.7, alpha=0.5)
        inset.axvline(tus, color="#D62728", ls=":", lw=0.7, alpha=0.5)

    if picked_in_band and pick["_us"] <= x_max:
        inset.scatter(
            [pick["_us"]],
            [pick["_ulp"]],
            s=42,
            marker="P",
            facecolor="#F1C232",
            edgecolor="#222222",
            linewidth=0.45,
            zorder=6,
            clip_on=False,
        )

    inset.set_xlim(0, x_max)
    inset.set_ylim(0, 1.02)
    inset.set_title("0-1 ULP inset", fontsize=8, pad=2)
    inset.tick_params(axis="both", labelsize=7, length=2, pad=1)
    inset.grid(True, color="#CCCCCC", alpha=0.12, linewidth=0.4)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_linewidth(0.5)
    inset.spines["bottom"].set_linewidth(0.5)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot frontier_sweep shard CSVs as per-activation scatters.")
    parser.add_argument("csvs", nargs="+", help="Shard CSV paths or globs")
    parser.add_argument("--ttnn", help="Optional TTNN reference CSV")
    parser.add_argument("--picked", help="Optional pareto_winners.csv manifest to highlight the IO-dumped frontier row")
    parser.add_argument(
        "--frontier-subdir",
        default="scatter",
        help="Optional subdirectory under --outdir for per-activation frontier plots",
    )
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Optional native-vs-embedded tier CSV for summary overlays; can be repeated",
    )
    parser.add_argument("--outdir", default="plots", help="Output plot directory")
    return parser.parse_args()


def expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    if not paths:
        raise SystemExit("frontier_scatter: no input CSVs")
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"frontier_scatter: input CSV not found: {missing[0]}")
    return paths


def main():
    args = parse_args()
    paths = expand_paths(args.csvs)
    rows, explicit_dtype = load(paths)
    ttnn, has_typed_ttnn = load_ttnn(args.ttnn)
    picked = load_picked(args.picked)
    tiers = load_tier_csvs(args.tier)

    by_group = defaultdict(list)
    for r in rows:
        key = (r["activation"], r["_dtype"]) if explicit_dtype else (r["activation"], None)
        by_group[key].append(r)
    dtypes = {dtype for _, dtype in by_group if dtype}
    split_dtype_dirs = len(dtypes) > 1
    os.makedirs(args.outdir, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        apply_tufte_style(plt, compact=True)
        have_mpl = True
    except Exception as e:
        have_mpl = False
        print(f"(matplotlib unavailable: {e} — emitting text frontier only)", file=sys.stderr)

    METHOD_COLOR = {
        "poly": "#4E79A7",
        "rational": "#59A14F",
        "exponent_alu_exp2": "#F28E2B",
        "exponent_alu_log2": "#E15759",
        "exponent_alu_pow": "#B07AA1",
        "newton_root": "#76B7B2",
    }
    METHOD_MARKER = {
        "poly": "o",
        "rational": "s",
        "exponent_alu_exp2": "^",
        "exponent_alu_log2": "v",
        "exponent_alu_pow": "D",
        "newton_root": "P",
        "trig": "X",
        "tan": "h",
    }
    n_win = 0
    n_with_ttnn = 0
    activations = {act for act, _ in by_group}
    dtype_msg = "explicit dtype groups" if explicit_dtype else "BF16-only sweep"
    print(f"# {len(rows)} configs, {len(activations)} activations, {dtype_msg}\n")
    for act, dtype in sorted(by_group):
        rs = by_group[(act, dtype)]
        comp = [r for r in rs if r["_ok"] and r["_us"] and r["_ulp"] is not None]
        front = pareto([(r["_us"], r["_ulp"], r) for r in comp])
        tus, tulp = ttnn_for(ttnn, act, dtype, has_typed_ttnn)
        if tus is not None and tulp is not None:
            n_with_ttnn += 1
        wins = [r for (_, _, r) in front if tus and tulp is not None and r["_us"] <= tus and r["_ulp"] <= tulp]
        if wins:
            n_win += 1
        print(
            f"{act:16s} {len(comp):4d} compile / {len(rs):4d} total | frontier {len(front):2d} pts"
            + (f" | TTNN {tus:.2f}us ulp{tulp}" if tus else "")
            + (
                f" | WINS: {','.join(sorted(set(r['_family'] for r in wins)))} "
                f"(fastest {min(r['_us'] for r in wins):.2f}us)"
                if wins
                else ""
            )
        )
        if not have_mpl:
            continue
        fig, ax = plt.subplots(figsize=(3.7, 2.4))
        methods = sorted(set(r.get("method") or "unknown" for r in rs))
        plotted_methods = set()
        for r in rs:
            if not (r["_ok"] and r["_us"] and r["_ulp"] is not None):
                continue
            method = r.get("method") or "unknown"
            c = METHOD_COLOR.get(method, "gray")
            marker = METHOD_MARKER.get(method, "o")
            label = method if method not in plotted_methods else None
            ax.scatter(
                r["_us"],
                r["_ulp"],
                c=c,
                marker=marker,
                s=13,
                alpha=0.72,
                edgecolors="none",
                label=label,
                clip_on=False,
            )
            plotted_methods.add(method)
        for method in methods:
            if method not in plotted_methods:
                ax.scatter(
                    [],
                    [],
                    c=METHOD_COLOR.get(method, "#666666"),
                    marker=METHOD_MARKER.get(method, "o"),
                    s=13,
                    label=method,
                )
        fxs = [p[0] for p in front]
        fys = [p[1] for p in front]
        if fxs:
            ax.plot(fxs, fys, color="#333333", ls="--", lw=0.8, alpha=0.65, label="Pareto frontier", clip_on=False)
        pick = picked_point(comp, picked_for(picked, act, dtype))
        if pick:
            ax.scatter(
                [pick["_us"]],
                [pick["_ulp"]],
                s=74,
                marker="P",
                facecolor="#F1C232",
                edgecolor="#222222",
                linewidth=0.65,
                zorder=6,
                label="picked",
                clip_on=False,
            )
        if tus:
            shade_ttnn_win_region(ax, tus, tulp)
            ax.scatter([tus], [tulp], c="#D62728", s=90, marker="*", zorder=5, label="TTNN", clip_on=False)
            ax.axhline(tulp, color="#D62728", ls=":", lw=0.7, alpha=0.55)
            ax.axvline(tus, color="#D62728", ls=":", lw=0.7, alpha=0.55)
        add_one_ulp_inset(ax, comp, front, tus, tulp, METHOD_COLOR, METHOD_MARKER, pick)
        ax.set_xlabel("Tracy runtime (us)")
        ax.set_ylabel(f"Maximum ULP error ({dtype_label(dtype)})")
        ax.set_ylim(bottom=0)
        ymin, ymax = ax.get_ylim()
        if ymax > 0:
            ax.set_ylim(0, ymax * 1.08)
        legend_outside_two_rows(ax)
        ax.grid(True, color="#CCCCCC", alpha=0.12, linewidth=0.5)
        fig.tight_layout()
        plot_dir = args.outdir
        if args.frontier_subdir:
            plot_dir = os.path.join(plot_dir, args.frontier_subdir)
        if split_dtype_dirs and dtype:
            plot_dir = os.path.join(plot_dir, slug(dtype))
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(os.path.join(plot_dir, f"{slug(act)}.png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    if n_with_ttnn:
        print(f"\n# {n_win}/{n_with_ttnn} activation groups with TTNN refs have a config that beats TTNN on both axes")
    else:
        print("\n# no matching TTNN refs for these activation groups")
    if have_mpl and tiers:
        write_tier_plots(args.outdir, tiers, ttnn, plt)
    if have_mpl:
        frontier_dir = os.path.join(args.outdir, args.frontier_subdir) if args.frontier_subdir else args.outdir
        if split_dtype_dirs:
            frontier_dir = os.path.join(frontier_dir, "<dtype>")
        print(f"# per-activation scatters -> {frontier_dir}/")


def write_tier_plots(outdir, tiers, ttnn, plt):
    by_act = defaultdict(list)
    for label, rows in tiers:
        for r in rows:
            act = r.get("activation")
            if act:
                by_act[act].append((label, r))
    if not by_act:
        return
    tier_dir = os.path.join(outdir, "tiers")
    os.makedirs(tier_dir, exist_ok=True)
    colors = {"best": "#4E79A7", "best99": "#59A14F", "best95": "#F28E2B"}
    for act in sorted(by_act):
        fig, ax = plt.subplots(figsize=(3.7, 2.4))
        tus, tulp = ttnn_for(ttnn, act, None, False)
        if tus and tulp is not None:
            ax.scatter([tus], [tulp], c="#D62728", s=90, marker="*", zorder=5, label="TTNN", clip_on=False)
            ax.axhline(tulp, color="#D62728", ls=":", lw=0.7, alpha=0.55)
            ax.axvline(tus, color="#D62728", ls=":", lw=0.7, alpha=0.55)
        for label, r in by_act[act]:
            ours_us = fnum(r.get("ours_us"))
            ours_ulp = fnum(r.get("ours_maxulp"))
            if ours_us is None or ours_ulp is None:
                continue
            ax.scatter(
                [ours_us],
                [ours_ulp],
                s=26,
                marker="D",
                color=colors.get(label, "gray"),
                label=label,
                alpha=0.8,
                edgecolors="none",
                clip_on=False,
            )
        if tus and tulp is not None:
            shade_ttnn_win_region(ax, tus, tulp)
        ax.set_xlabel("Tracy runtime (us)")
        ax.set_ylabel("Maximum ULP error (BF16)")
        ax.set_ylim(bottom=0)
        ymin, ymax = ax.get_ylim()
        if ymax > 0:
            ax.set_ylim(0, ymax * 1.08)
        ax.grid(True, color="#CCCCCC", alpha=0.12, linewidth=0.5)
        legend_outside_two_rows(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(tier_dir, f"{slug(act)}.png"), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"# tier comparison plots -> {tier_dir}/")


if __name__ == "__main__":
    main()
