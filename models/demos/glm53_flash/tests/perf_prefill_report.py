# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op / per-module / per-layer-kind tables of a ``test_mesh_prefill_profile.py`` Tracy run (one prefill chunk).

    python models/demos/glm53_flash/tests/perf_prefill_report.py <ops_perf_results_*.csv> --out <dir> \\
        --rows 2 --cols 4 [--ids 1,0,2,3,5,4,6,7] [--window PASS_ATTRIBUTED] \\
        [--counts "KDA+dense=3,DSA+MoE=5,KDA+MoE=14"] [--top 15]

The mesh profiler writes one ops CSV with a ``DEVICE ID`` column; the script splits it per device, runs
``tt-perf-report --start-signpost <window> --end-signpost <window>_END`` on each split (``--csv``: the per-op
rows with ``Device Time`` in us, joined back to the raw rows on the global call count) and labels every op of the
window with the innermost signpost open when it was issued: the layer (``L<idx>_<kind>_BEGIN`` .. ``_END``) and
the module labels the test's ``_ModuleTimer`` emits (``KDA`` / ``DSA_*`` / ``MOE_*`` / ``ATTN_HC_PRE`` ... /
``CCL_*`` / ``EMBEDDING`` / ``HEAD``; ops outside every module label inside a layer are ``LAYER_GLUE``).
Writes ``summary.md``: per layer, per layer kind (mean per layer), per (kind, module), the top ops per kind
(same op code collapsed), the op classes, the collectives in order, and the composition of a full chunk from
the per-kind means and ``--counts`` (the layer kinds of the 22-layer model). The device time of every chip is
listed; the tables are chip (0,0)'s.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

import pandas as pd

CCL_NAMES = ("AllGather", "ReduceScatter", "AllReduce", "AllBroadcast", "AllToAll")
LAYER_RE = re.compile(r"^L(\d+)_(.+?)_(BEGIN|END)$")


def _is_ccl(op: str) -> bool:
    return any(n in op for n in CCL_NAMES)


def _signpost_window(df: pd.DataFrame, start: str, end: str) -> tuple[int, int]:
    sp = df[df["OP TYPE"] == "signpost"]
    t0 = sp[sp["OP CODE"] == start]["HOST START TS"].min()
    t1 = sp[sp["OP CODE"] == end]["HOST START TS"].min()
    if pd.isna(t0) or pd.isna(t1):
        raise SystemExit(f"signposts {start}/{end} not found")
    return int(t0), int(t1)


def label_window(df: pd.DataFrame, device_id: int, start: str, end: str) -> pd.DataFrame:
    """Rows (global call count -> layer idx, kind, module, stack) of device ``device_id``'s ops in the window."""
    t0, t1 = _signpost_window(df, start, end)
    win = df[(df["HOST START TS"] >= t0) & (df["HOST START TS"] <= t1)]
    win = win[(win["OP TYPE"] == "signpost") | (win["DEVICE ID"] == device_id)].sort_values("HOST START TS")
    layer_idx, kind = -1, "OUTSIDE"
    stack: list[str] = []
    rows = []
    for _, row in win.iterrows():
        code = str(row["OP CODE"])
        if row["OP TYPE"] == "signpost":
            m = LAYER_RE.match(code)
            if m:
                if m.group(3) == "BEGIN":
                    layer_idx, kind = int(m.group(1)), m.group(2)
                else:
                    layer_idx, kind = -1, "OUTSIDE"
                stack.clear()
            elif code.endswith("_BEGIN"):
                stack.append(code[: -len("_BEGIN")])
            elif code.endswith("_END"):
                name = code[: -len("_END")]
                if stack and stack[-1] == name:
                    stack.pop()
            continue
        if stack:
            module = stack[-1]
        elif kind in ("OUTSIDE",):
            module = "OUTSIDE"
        else:
            module = "LAYER_GLUE"
        if module in ("EMBEDDING", "HEAD"):
            kind = module
        rows.append(
            {
                "Global Call Count": int(row["GLOBAL CALL COUNT"]),
                "layer": layer_idx,
                "kind": kind,
                "module": module,
                "stack": "/".join(stack),
            }
        )
    return pd.DataFrame(rows)


def run_perf_report(tool: str, csv: Path, txt: Path, out_csv: Path, start: str, end: str) -> None:
    base = [tool, str(csv), "--start-signpost", start, "--end-signpost", end, "--no-advice", "--no-color"]
    res = subprocess.run(base + ["--csv", str(out_csv), "--no-summary"], capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"tt-perf-report failed on {csv}: rc={res.returncode}\n{res.stderr[-2000:]}")
    res = subprocess.run(base + ["--no-summary"], capture_output=True, text=True)
    txt.write_text(res.stdout + ("\n[stderr]\n" + res.stderr if res.stderr.strip() else ""))


def _op_class(code: str) -> str:
    return code.split(" ")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--ids", default="", help="device ids in row-major mesh order (default: id = r * cols + c)")
    ap.add_argument("--window", default="PASS_ATTRIBUTED", help="signpost name; the end is <window>_END")
    ap.add_argument("--counts", default="KDA+dense=3,DSA+MoE=5,KDA+MoE=14", help="layer kinds of the full model")
    ap.add_argument("--tool", default=shutil.which("tt-perf-report") or "tt-perf-report")
    ap.add_argument("--title", default="")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    start, end = args.window, args.window + "_END"
    df = pd.read_csv(args.csv, low_memory=False)
    df["DEVICE ID"] = pd.to_numeric(df["DEVICE ID"], errors="coerce")
    present = sorted(int(d) for d in df[df["OP TYPE"] == "tt_dnn_device"]["DEVICE ID"].dropna().unique())
    dev_ids = [int(v) for v in args.ids.split(",")] if args.ids else present
    assert sorted(dev_ids) == present, (dev_ids, present)
    coord = {d: divmod(i, args.cols) for i, d in enumerate(dev_ids)}
    counts = OrderedDict()
    for part in args.counts.split(","):
        k, _, v = part.partition("=")
        counts[k] = int(v)
    print(f"devices in the CSV: {present}; (r,c) -> id: {dev_ids}; window {start}..{end}")

    per_dev: "OrderedDict[int, pd.DataFrame]" = OrderedDict()
    for d in dev_ids:
        r, c = coord[d]
        sub = df[(df["OP TYPE"] != "tt_dnn_device") | (df["DEVICE ID"] == d)]
        split = out / f"ops_dev{d}.csv"
        sub.to_csv(split, index=False)
        txt = out / f"dev{d}_r{r}c{c}_prefill_perf_report.txt"
        ops_csv = out / f"dev{d}_r{r}c{c}_prefill_ops.csv"
        run_perf_report(args.tool, split, txt, ops_csv, start, end)
        rep = pd.read_csv(ops_csv)
        rep = rep[rep["OP Code"].astype(str) != "nan"]
        rep = rep[~rep["OP Code"].astype(str).str.startswith(("PERF_", "PASS_", "L"))]
        rep["Device Time"] = pd.to_numeric(rep["Device Time"], errors="coerce").fillna(0.0)
        rep["Global Call Count"] = pd.to_numeric(rep["Global Call Count"], errors="coerce").astype("Int64")
        labels = label_window(df, d, start, end)
        rep = rep.merge(labels, on="Global Call Count", how="left")
        rep["kind"] = rep["kind"].fillna("UNLABELLED")
        rep["module"] = rep["module"].fillna("UNLABELLED")
        rep["layer"] = rep["layer"].fillna(-1).astype(int)
        rep["class"] = rep["OP Code"].astype(str).map(_op_class)
        per_dev[d] = rep
        split.unlink()
        rep.to_csv(out / f"dev{d}_r{r}c{c}_prefill_ops_labelled.csv", index=False)

    lines = [f"# {args.title or args.csv.name}", "", f"window `{start}` .. `{end}` (one prefill chunk, eager)", ""]
    lines += [
        "## Device time per chip",
        "",
        "| chip | (r,c) | ops | device ms | CCL kernel ms |",
        "|---|---|---|---|---|",
    ]
    for d, rep in per_dev.items():
        r, c = coord[d]
        ccl = rep.loc[rep["OP Code"].astype(str).map(_is_ccl), "Device Time"].sum()
        lines.append(f"| {d} | ({r},{c}) | {len(rep)} | {rep['Device Time'].sum() / 1e3:.1f} | {ccl / 1e3:.2f} |")
    lines.append("")
    d0 = dev_ids[0]
    rep = per_dev[d0]
    tot = rep["Device Time"].sum()

    lines += [f"## Chip {d0} (0,0): per layer", "", "| layer | kind | ops | device ms |", "|---|---|---|---|"]
    per_layer = rep.groupby(["layer", "kind"], sort=True)["Device Time"].agg(["sum", "count"])
    for (li, kind), row in per_layer.iterrows():
        lines.append(f"| {li if li >= 0 else '-'} | {kind} | {int(row['count'])} | {row['sum'] / 1e3:.2f} |")
    lines.append("")

    lines += [
        f"## Chip {d0} (0,0): per layer kind (mean per layer of the kind)",
        "",
        "| kind | layers in window | ops / layer | device ms / layer |",
        "|---|---|---|---|",
    ]
    kind_mean = {}
    for kind, grp in rep.groupby("kind"):
        n_layers = max(1, grp["layer"].nunique()) if kind not in ("EMBEDDING", "HEAD", "OUTSIDE") else 1
        kind_mean[kind] = (grp["Device Time"].sum() / n_layers, len(grp) / n_layers, n_layers)
        lines.append(
            f"| {kind} | {n_layers} | {len(grp) / n_layers:.0f} | {grp['Device Time'].sum() / n_layers / 1e3:.2f} |"
        )
    lines.append("")
    comp = 0.0
    lines += [
        "### Composition of a full chunk from the per-kind means",
        "",
        "| kind | layers | ms / layer | ms |",
        "|---|---|---|---|",
    ]
    for kind, n in counts.items():
        if kind in kind_mean:
            ms = kind_mean[kind][0] / 1e3
            comp += n * ms
            lines.append(f"| {kind} | {n} | {ms:.2f} | {n * ms:.1f} |")
    for kind in ("EMBEDDING", "HEAD"):
        if kind in kind_mean:
            ms = kind_mean[kind][0] / 1e3
            comp += ms
            lines.append(f"| {kind} | 1 | {ms:.2f} | {ms:.1f} |")
    lines.append(f"| **total** | | | **{comp:.1f}** |")
    lines.append("")

    lines += [f"## Chip {d0} (0,0): per (kind, module), exclusive device time", ""]
    lines += ["| kind | module | ops / layer | device ms / layer | % of the kind |", "|---|---|---|---|---|"]
    for kind, grp in rep.groupby("kind"):
        n_layers = kind_mean[kind][2]
        ktot = grp["Device Time"].sum()
        mods = grp.groupby("module")["Device Time"].agg(["sum", "count"]).sort_values("sum", ascending=False)
        for mod, row in mods.iterrows():
            lines.append(
                f"| {kind} | {mod} | {row['count'] / n_layers:.0f} | {row['sum'] / n_layers / 1e3:.3f} | {100 * row['sum'] / max(ktot, 1e-9):.1f} % |"
            )
    lines.append("")

    for kind, grp in rep.groupby("kind"):
        n_layers = kind_mean[kind][2]
        ktot = grp["Device Time"].sum()
        lines += [f"### {kind}: top-{args.top} ops (same op code collapsed; per layer of the kind)", ""]
        lines += [
            "| op | module | cores | count | us each | ms total | % of the kind |",
            "|---|---|---|---|---|---|---|",
        ]
        g = grp.groupby(["OP Code", "module"]).agg(
            n=("Device Time", "count"), us=("Device Time", "sum"), cores=("Cores", "max")
        )
        for (op, mod), row in g.sort_values("us", ascending=False).head(args.top).iterrows():
            lines.append(
                f"| {op} | {mod} | {row['cores']} | {row['n'] / n_layers:.0f} | {row['us'] / row['n']:.0f} | "
                f"{row['us'] / n_layers / 1e3:.2f} | {100 * row['us'] / max(ktot, 1e-9):.1f} % |"
            )
        lines.append("")
        lines += [
            f"### {kind}: by op class (per layer of the kind)",
            "",
            "| op class | count | ms | % of the kind |",
            "|---|---|---|---|",
        ]
        cls = grp.groupby("class")["Device Time"].agg(["sum", "count"]).sort_values("sum", ascending=False)
        for name, row in cls.head(16).iterrows():
            lines.append(
                f"| {name} | {row['count'] / n_layers:.0f} | {row['sum'] / n_layers / 1e3:.2f} | {100 * row['sum'] / max(ktot, 1e-9):.1f} % |"
            )
        lines.append("")

    lines += [
        f"## Chip {d0} (0,0): collectives in order",
        "",
        "| ID | layer | op | module | device us |",
        "|---|---|---|---|---|",
    ]
    for _, row in rep[rep["OP Code"].astype(str).map(_is_ccl)].iterrows():
        lines.append(
            f"| {int(row['ID'])} | {row['layer']} | {row['OP Code']} | {row['module']} | {row['Device Time']:.0f} |"
        )
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
