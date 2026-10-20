# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-device ``tt-perf-report`` tables of a ``test_perf_mesh.py`` Tracy run, with per-module attribution.

    python models/demos/glm53_flash/tests/perf_mesh_report.py <ops_perf_results_*.csv> --out <dir> --rows 2 --cols 4

The mesh profiler writes one ops CSV with a ``DEVICE ID`` column (local chip id; row-major mesh order on
the (2,4) block: id = r * cols + c). This script splits it per device, runs
``tt-perf-report --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END`` on each split (text table
+ ``--csv``), labels every op of the traced window with the module that issued it (from the ``PERF_EAGER``
window's nested ``<LABEL>_BEGIN`` / ``<LABEL>_END`` signposts: the eager step issues the same ops in the
same order as the captured graph) and writes ``summary.md``: step time per chip, the top-10 ops of chip
(0,0), and the CCL / expert / attention-composite shares. Per-op CSVs are git-excluded; commit the text.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

import pandas as pd

CCL_NAMES = (
    "AllGather",
    "ReduceScatter",
    "AllReduce",
    "AllBroadcast",
)  # all_gather over a 2-row axis lowers to AllBroadcast


def _is_ccl(op: str) -> bool:
    return any(n in op for n in CCL_NAMES)


def _signpost_window(df: pd.DataFrame, start: str, end: str) -> tuple[int, int]:
    sp = df[df["OP TYPE"] == "signpost"]
    t0 = sp[sp["OP CODE"] == start]["HOST START TS"].min()
    t1 = sp[sp["OP CODE"] == end]["HOST START TS"].min()
    if pd.isna(t0) or pd.isna(t1):
        raise SystemExit(f"signposts {start}/{end} not found")
    return int(t0), int(t1)


def eager_labels(df: pd.DataFrame, device_id: int) -> list[list[str]]:
    """Label stack of every device op of ``device_id`` inside the PERF_EAGER window, in host order."""
    t0, t1 = _signpost_window(df, "PERF_EAGER", "PERF_EAGER_END")
    win = df[(df["HOST START TS"] >= t0) & (df["HOST START TS"] <= t1)]
    win = win[(win["OP TYPE"] == "signpost") | (win["DEVICE ID"] == device_id)].sort_values("HOST START TS")
    stack: list[str] = []
    out: list[list[str]] = []
    for _, row in win.iterrows():
        code = str(row["OP CODE"])
        if row["OP TYPE"] == "signpost":
            if code.endswith("_BEGIN"):
                stack.append(code[: -len("_BEGIN")])
            elif code.endswith("_END"):
                name = code[: -len("_END")]
                if stack and stack[-1] == name:
                    stack.pop()
            continue
        out.append(list(stack))
    return out


def run_perf_report(tool: str, csv: Path, txt: Path, out_csv: Path, summary: Path | None) -> None:
    """Two passes: ``--csv`` (machine-readable rows; the stacked summary for chip (0,0)) and the plain
    text table (``--csv`` suppresses the table on stdout)."""
    base = [
        tool,
        str(csv),
        "--start-signpost",
        "PERF_DECODE",
        "--end-signpost",
        "PERF_DECODE_END",
        "--no-advice",
        "--no-color",
    ]
    cmd = (
        base + ["--csv", str(out_csv)] + (["--summary-file", str(summary)] if summary is not None else ["--no-summary"])
    )
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"tt-perf-report failed on {csv}: rc={res.returncode}\n{res.stderr[-2000:]}")
    res = subprocess.run(base + ["--no-summary"], capture_output=True, text=True)
    txt.write_text(res.stdout + ("\n[stderr]\n" + res.stderr if res.stderr.strip() else ""))
    if res.returncode != 0:
        raise SystemExit(f"tt-perf-report (text) failed on {csv}: rc={res.returncode}\n{res.stderr[-2000:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tool", default=shutil.which("tt-perf-report") or "tt-perf-report")
    ap.add_argument("--title", default="")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument(
        "--ids",
        default="",
        help="device ids in row-major mesh order as the test logs them ('mesh RxC device ids (row-major): [...]'); "
        "default: id = r * cols + c",
    )
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv, low_memory=False)
    df["DEVICE ID"] = pd.to_numeric(df["DEVICE ID"], errors="coerce")
    present = sorted(int(d) for d in df[df["OP TYPE"] == "tt_dnn_device"]["DEVICE ID"].dropna().unique())
    dev_ids = [int(v) for v in args.ids.split(",")] if args.ids else present  # row-major mesh order
    assert sorted(dev_ids) == present, (dev_ids, present)
    coord = {d: divmod(i, args.cols) for i, d in enumerate(dev_ids)}
    print(f"devices in the CSV: {present}; (r,c) -> id: {dev_ids}")

    labels0 = None
    try:
        labels0 = eager_labels(df, dev_ids[0])
        print(f"eager window on device {dev_ids[0]}: {len(labels0)} ops labelled")
    except SystemExit as e:
        print(f"no eager window ({e}); name-based attribution only")

    per_dev: "OrderedDict[int, pd.DataFrame]" = OrderedDict()
    for d in dev_ids:
        r, c = coord[d]
        sub = df[(df["OP TYPE"] != "tt_dnn_device") | (df["DEVICE ID"] == d)]
        split = out / f"ops_dev{d}.csv"
        sub.to_csv(split, index=False)
        txt = out / f"dev{d}_r{r}c{c}_decode_perf_report.txt"
        ops_csv = out / f"dev{d}_r{r}c{c}_decode_ops.csv"
        summary = (out / f"dev{d}_r{r}c{c}_decode_summary") if d == dev_ids[0] else None
        run_perf_report(args.tool, split, txt, ops_csv, summary)
        rep = pd.read_csv(ops_csv)
        rep = rep[rep["OP Code"].astype(str) != "nan"]
        rep = rep[~rep["OP Code"].astype(str).str.startswith("PERF_")]  # drop signpost rows if present
        rep["Device Time"] = pd.to_numeric(rep["Device Time"], errors="coerce").fillna(0.0)
        if labels0 is not None and len(labels0) == len(rep):
            rep["Labels"] = ["/".join(s) if s else "MHC" for s in labels0]
        else:
            if labels0 is not None:
                print(f"device {d}: traced window has {len(rep)} ops, eager window {len(labels0)}: name-based only")
            rep["Labels"] = ["CCL" if _is_ccl(str(o)) else "" for o in rep["OP Code"]]
        per_dev[d] = rep
        split.unlink()  # the per-device raw split is reproducible from the ops CSV

    def share(rep: pd.DataFrame, pred) -> tuple[float, int]:
        m = rep.apply(pred, axis=1)
        return float(rep.loc[m, "Device Time"].sum()), int(m.sum())

    groups = OrderedDict(
        [
            ("CCL ops (AllGather/AllBroadcast/ReduceScatter/AllReduce kernels)", lambda r: _is_ccl(str(r["OP Code"]))),
            (
                "CCL composites (all_reduce/all_gather/reduce_scatter incl. local reduce ops)",
                lambda r: "CCL_" in r["Labels"],
            ),
            ("sparse_matmul (expert matmuls)", lambda r: "SparseMatmul" in str(r["OP Code"])),
            (
                "EXPERTS local (router-weight select, sparse_matmul, slicing; no CCL)",
                lambda r: "EXPERTS" in r["Labels"] and "CCL_" not in r["Labels"],
            ),
            ("ROUTER (exact fp32 composite top-8)", lambda r: r["Labels"].endswith("ROUTER")),
            ("SHARED expert", lambda r: r["Labels"].endswith("SHARED")),
            ("MoE total (router + experts + shared + CCL + add)", lambda r: r["Labels"].startswith("MOE")),
            (
                "ATTN composite incl. its all_reduce (KDA recurrence / DSA flash-MLA + index)",
                lambda r: r["Labels"].startswith("ATTN"),
            ),
            ("ATTN local only (no CCL)", lambda r: r["Labels"].startswith("ATTN") and "CCL_" not in r["Labels"]),
            ("NORM (RMSNorm x2)", lambda r: r["Labels"] == "NORM"),
            ("MHC (hyper-connections x2, everything outside the sublayers)", lambda r: r["Labels"] == "MHC"),
        ]
    )

    lines = [f"# {args.title or args.csv.name}", ""]
    lines.append("## Device time per chip (traced decode window, one replay)")
    lines.append("")
    lines.append("| chip | (r,c) | ops | device ms/step | CCL kernel ms | CCL composite ms |")
    lines.append("|---|---|---|---|---|---|")
    for d, rep in per_dev.items():
        r, c = coord[d]
        tot = rep["Device Time"].sum() / 1e3
        ccl, _ = share(rep, groups["CCL ops (AllGather/AllBroadcast/ReduceScatter/AllReduce kernels)"])
        cclc, _ = share(rep, groups["CCL composites (all_reduce/all_gather/reduce_scatter incl. local reduce ops)"])
        lines.append(f"| {d} | ({r},{c}) | {len(rep)} | {tot:.3f} | {ccl / 1e3:.3f} | {cclc / 1e3:.3f} |")
    lines.append("")
    for d in [dev_ids[0]] + ([dev_ids[args.cols]] if len(dev_ids) > args.cols else []):
        rep = per_dev[d]
        r, c = coord[d]
        tot = rep["Device Time"].sum()
        lines.append(f"## Chip {d} ({r},{c}): shares of the {tot / 1e3:.3f} ms step")
        lines.append("")
        lines.append("| group | device us | % of step | ops |")
        lines.append("|---|---|---|---|")
        for name, pred in groups.items():
            us, n = share(rep, pred)
            lines.append(f"| {name} | {us:.0f} | {100 * us / tot:.1f} % | {n} |")
        lines.append("")
        lines.append(f"### Chip {d} ({r},{c}): top-{args.top} ops")
        lines.append("")
        lines.append("| ID | op | module | device us | % of step |")
        lines.append("|---|---|---|---|---|")
        top = rep.sort_values("Device Time", ascending=False).head(args.top)
        for _, row in top.iterrows():
            lines.append(
                f"| {int(row['ID'])} | {row['OP Code']} | {row['Labels']} | {row['Device Time']:.0f} | "
                f"{100 * row['Device Time'] / tot:.1f} % |"
            )
        lines.append("")
        lines.append(f"### Chip {d} ({r},{c}): by op class")
        lines.append("")
        lines.append("| op class | device us | % of step | count |")
        lines.append("|---|---|---|---|")
        cls = rep.groupby(rep["OP Code"].astype(str).str.split(" ").str[0])["Device Time"].agg(["sum", "count"])
        for name, row in cls.sort_values("sum", ascending=False).head(14).iterrows():
            lines.append(f"| {name} | {row['sum']:.0f} | {100 * row['sum'] / tot:.1f} % | {int(row['count'])} |")
        lines.append("")
        lines.append(f"### Chip {d} ({r},{c}): CCL ops in order")
        lines.append("")
        lines.append("| ID | op | module | device us |")
        lines.append("|---|---|---|---|")
        for _, row in rep[rep["OP Code"].astype(str).map(_is_ccl)].iterrows():
            lines.append(f"| {int(row['ID'])} | {row['OP Code']} | {row['Labels']} | {row['Device Time']:.0f} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
