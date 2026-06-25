#!/usr/bin/env python3
"""Generate ULP-by-input plots from a Pareto winner dump manifest."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

import numpy as np

try:
    from .style import apply_tufte_style
    from .ulp_by_input import (
        add_low_ulp_inset,
        compute_ulp_errors,
        downsample,
        legend_handles,
        load_dump,
        main as ulp_main,
        series_draw_params,
        series_style,
    )
except ImportError:
    from style import apply_tufte_style
    from ulp_by_input import (
        add_low_ulp_inset,
        compute_ulp_errors,
        downsample,
        legend_handles,
        load_dump,
        main as ulp_main,
        series_draw_params,
        series_style,
    )


def existing(path):
    return Path(path).expanduser().resolve()


OLD_TT_METAL_ROOT = Path("/Users/nachiket/workspace/tt-metal")


def repo_root():
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def resolve_repo_path(path, repo):
    p = Path(path or "")
    if not p:
        return p
    try:
        return repo / p.relative_to(OLD_TT_METAL_ROOT)
    except ValueError:
        return p


def plot_path(row, manifest, repo):
    p = resolve_repo_path(row.get("plot_png") or "", repo)
    if p and str(p).startswith(str(repo)):
        return p
    dtype = row.get("dtype") or "bf16"
    act = row.get("activation")
    run_dir = manifest.parent.parent.parent
    return run_dir / "plots" / "ulp_by_input" / dtype / f"{act}.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot raw IO dumps selected by select_pareto_winners.py.")
    parser.add_argument(
        "--manifest",
        required=True,
        action="append",
        type=existing,
        help="Pareto dump manifest. Pass once for per-dtype plots or multiple times for side-by-side dtype panels.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any selected dump is missing. Default: skip incomplete activations.",
    )
    parser.add_argument(
        "--outdir",
        type=existing,
        help="Output directory for combined multi-manifest plots. Defaults to results/frontier/plots/ulp_by_input.",
    )
    parser.add_argument("--max-points", type=int, default=50000)
    return parser.parse_args()


def fnum(value):
    try:
        v = float(value)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def choose_ours(rows):
    frontier = [row for row in rows if row.get("role") != "ttnn"]
    if not frontier:
        return None, "no_frontier"

    ttnn_ulp = None
    for row in rows:
        if row.get("role") == "ttnn":
            ttnn_ulp = fnum(row.get("ttnn_maxulp") or row.get("max_ulp"))
            break
    if ttnn_ulp is None:
        ttnn_ulp = fnum(frontier[0].get("ttnn_maxulp"))

    def sort_key(row):
        runtime = fnum(row.get("runtime_us"))
        ulp = fnum(row.get("max_ulp"))
        return (
            float("inf") if runtime is None else runtime,
            float("inf") if ulp is None else ulp,
            row.get("csv") or row.get("dump_csv") or "",
        )

    if ttnn_ulp is not None:
        matches = [
            row for row in frontier if (fnum(row.get("max_ulp")) is not None and fnum(row.get("max_ulp")) <= ttnn_ulp)
        ]
        if matches:
            return min(matches, key=sort_key), "fastest_ttnn_ulp_match"

    ulps = [fnum(row.get("max_ulp")) for row in frontier]
    ulps = [ulp for ulp in ulps if ulp is not None]
    if not ulps:
        return min(frontier, key=sort_key), "fastest_no_numeric_ulp"
    best_ulp = min(ulps)
    best = [row for row in frontier if fnum(row.get("max_ulp")) == best_ulp]
    return min(best, key=sort_key), "fastest_min_ulp"


def choose_rows(rows):
    selected, reason = choose_ours(rows)
    if selected is None:
        return [], reason
    selected = dict(selected)
    selected["_label"] = "ours"
    chosen = [selected]
    ttnn = next((dict(row) for row in rows if row.get("role") == "ttnn"), None)
    if ttnn:
        ttnn["_label"] = "TTNN"
        chosen.append(ttnn)
    return chosen, reason


def load_manifest(manifest, repo):
    if not manifest.exists():
        raise SystemExit(f"plot_pareto_io: manifest not found: {manifest}")
    groups = defaultdict(list)
    with manifest.open() as f:
        for row in csv.DictReader(f):
            act = row.get("activation")
            dtype = row.get("dtype") or "bf16"
            dump = resolve_repo_path(row.get("dump_csv") or "", repo)
            plot = plot_path(row, manifest, repo)
            if not act or not dump or not plot:
                continue
            row = dict(row)
            row["dump_csv"] = str(dump)
            row["plot_png"] = str(plot)
            groups[(act, dtype, plot)].append(row)
    return groups


def plot_single_manifests(args, repo):
    made = skipped = missing_count = fallback_count = 0
    groups = {}
    for manifest in args.manifest:
        groups.update(load_manifest(manifest, repo))

    for (act, dtype, plot), rows in sorted(groups.items()):
        rows, reason = choose_rows(rows)
        if reason != "fastest_ttnn_ulp_match":
            fallback_count += 1
            print(f"# {act} {dtype}: {reason}", file=sys.stderr)
        if not rows:
            skipped += 1
            continue
        missing = [row for row in rows if not Path(row["dump_csv"]).exists()]
        if missing:
            missing_count += len(missing)
            msg = f"plot_pareto_io: {act} {dtype} missing {len(missing)}/{len(rows)} dumps"
            if args.strict:
                first = missing[0]["dump_csv"]
                raise SystemExit(f"{msg}; first missing: {first}")
            print(f"# skip {msg}", file=sys.stderr)
            skipped += 1
            continue

        argv = [
            "ulp_by_input.py",
            "--activation",
            act,
            "--precision",
            dtype,
            "--out",
            str(plot),
            "--max-points",
            str(args.max_points),
        ]
        for row in rows:
            argv += ["--series", f"{row['_label']}={row['dump_csv']}"]

        old_argv = sys.argv
        try:
            sys.argv = argv
            ulp_main()
        finally:
            sys.argv = old_argv
        made += 1

    print(
        f"# pareto IO plots made={made} skipped={skipped} " f"missing_dumps={missing_count} fallbacks={fallback_count}"
    )


def combined_outdir(manifests, explicit):
    if explicit:
        return explicit
    roots = [manifest.parent.parent.parent.parent for manifest in manifests]
    root = roots[0]
    if any(other != root for other in roots):
        raise SystemExit("plot_pareto_io: pass --outdir when manifests do not share a frontier root")
    return root / "plots" / "ulp_by_input"


def prepare_series(act, dtype, rows, max_points):
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759", "#B07AA1", "#76B7B2"]
    plotted = []
    for i, row in enumerate(rows):
        label = row["_label"]
        inputs, outputs = load_dump(Path(row["dump_csv"]))
        x, ulp = compute_ulp_errors(act, dtype, inputs, outputs)
        x, ulp = downsample(x, ulp, max_points)
        color, marker = series_style(label, colors[i % len(colors)])
        plotted.append((label, x, ulp, color, marker))
    return plotted


def draw_panel(ax, plotted, dtype, xlim, ylim):
    for label, x, ulp, color, marker in plotted:
        size, alpha, zorder = series_draw_params(label)
        kwargs = {}
        if marker != "x":
            kwargs["edgecolors"] = "none"
        ax.scatter(
            x,
            ulp,
            s=size,
            alpha=alpha,
            color=color,
            label=label,
            marker=marker,
            rasterized=True,
            zorder=zorder,
            **kwargs,
        )
    ax.set_title(dtype.upper(), pad=2.0)
    ax.set_xlabel("Input domain")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    ax.grid(True, color="#CCCCCC", alpha=0.12, linewidth=0.5)
    add_low_ulp_inset(ax, plotted)


def plot_combined(args, repo):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_dtype = defaultdict(dict)
    for manifest in args.manifest:
        for (act, dtype, _plot), rows in load_manifest(manifest, repo).items():
            by_dtype[act][dtype] = rows

    outdir = combined_outdir(args.manifest, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    made = skipped = missing_count = fallback_count = 0
    dtypes = ["bf16", "fp32"]
    for act in sorted(by_dtype):
        panels = []
        for dtype in dtypes:
            rows = by_dtype[act].get(dtype)
            if not rows:
                continue
            chosen, reason = choose_rows(rows)
            if reason != "fastest_ttnn_ulp_match":
                fallback_count += 1
                print(f"# {act} {dtype}: {reason}", file=sys.stderr)
            missing = [row for row in chosen if not Path(row["dump_csv"]).exists()]
            if missing:
                missing_count += len(missing)
                msg = f"plot_pareto_io: {act} {dtype} missing {len(missing)}/{len(chosen)} dumps"
                if args.strict:
                    raise SystemExit(f"{msg}; first missing: {missing[0]['dump_csv']}")
                print(f"# skip {msg}", file=sys.stderr)
                continue
            panels.append((dtype, prepare_series(act, dtype, chosen, args.max_points)))

        if not panels:
            skipped += 1
            continue

        xs = [x for _, plotted in panels for _, x, _, _, _ in plotted if len(x)]
        ys = [y[np.isfinite(y)] for _, plotted in panels for _, _, y, _, _ in plotted if np.any(np.isfinite(y))]
        if not xs or not ys:
            skipped += 1
            continue
        xmin = min(float(np.nanmin(x)) for x in xs)
        xmax = max(float(np.nanmax(x)) for x in xs)
        ymax = max(float(np.nanmax(y)) for y in ys)
        xpad = (xmax - xmin) * 0.02 if xmax > xmin else 1.0
        ylim = (0, ymax * 1.04 if ymax > 0 else 1.0)

        apply_tufte_style(plt, compact=True)
        fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 2.35), sharex=True, sharey=True)
        if len(panels) == 1:
            axes = [axes]
        for ax, (dtype, plotted) in zip(axes, panels):
            draw_panel(ax, plotted, dtype, (xmin - xpad, xmax + xpad), ylim)
        axes[0].set_ylabel("ULP error")
        for ax in axes[1:]:
            ax.set_ylabel("")

        handles = legend_handles(plt, [entry for _, plotted in panels for entry in plotted])
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=min(4, len(handles)),
            frameon=False,
            handletextpad=0.45,
            columnspacing=0.95,
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))

        out = outdir / f"{act}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"# ULP-by-input paired plot -> {out}")
        made += 1

    print(
        f"# pareto IO paired plots made={made} skipped={skipped} "
        f"missing_dumps={missing_count} fallbacks={fallback_count}"
    )


def main():
    args = parse_args()
    repo = repo_root()
    if len(args.manifest) == 1:
        plot_single_manifests(args, repo)
    else:
        plot_combined(args, repo)


if __name__ == "__main__":
    main()
