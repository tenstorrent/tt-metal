# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Create per-cell reports from Tracy operation data."""

import argparse
import contextlib
import csv
import getpass
import io
import json
import os
import re
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path

MANIFEST_DIR = "layer_perf"
SUMMARY_NAME = "gemma4_d_p_layer_perf.md"
SIGNPOST_PATTERN = re.compile(r"^gemma4-layer-(global|local)-sz([0-9]+)-chunk([0-9]+)-(start|stop)$")
GAP_COLUMNS = ("OP TO OP LATENCY [ns]", "OP TO OP LATENCY BR/NRISC START [ns]")
# GitHub caps a job step summary at 1 MiB.
SUMMARY_TEXT_LIMIT = 900_000


def summaries_root():
    return Path(os.getenv("PREFILL_SUMMARIES", f"/tmp/prefill_summaries_{getpass.getuser()}"))


def is_primary_rank():
    for var in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK"):
        rank = os.environ.get(var)
        if rank is not None:
            return rank == "0"
    return True


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def write_manifest(run_id, cells, **meta):
    if not is_primary_rank():
        return None
    out_dir = summaries_root() / MANIFEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"manifest_{_safe_name(run_id)}.json"
    path.write_text(json.dumps({"run_id": run_id, **meta, "cells": cells}, indent=2))
    return path


def find_ops_csvs(profiler_dir):
    return sorted(
        Path(profiler_dir).glob("reports/**/ops_perf_results_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )


def signposts_in(ops_csv):
    with open(ops_csv, newline="") as f:
        return {r["OP CODE"] for r in csv.DictReader(f) if r.get("OP TYPE") == "signpost"}


def match_ops_csvs(manifest_paths, manifests, ops_csvs):
    # Each chunk size runs in its own Tracy session, which writes its ops CSV after the test writes the manifest.
    signposts = {p: signposts_in(p) for p in ops_csvs}
    matched = []
    for path, m in zip(manifest_paths, manifests):
        need = {s for c in m["cells"] for s in (c["start_signpost"], c["stop_signpost"])}
        newer = (p for p in ops_csvs if p.stat().st_mtime >= path.stat().st_mtime)
        matched.append(next((p for p in newer if need <= signposts[p]), None))
    return matched


def read_ops_csv(ops_csv):
    with open(ops_csv, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_cell_ops_csv(fieldnames, rows, start_signpost, stop_signpost, path):
    def signpost_at(name, begin):
        return next(
            (i for i in range(begin, len(rows)) if rows[i].get("OP TYPE") == "signpost" and rows[i]["OP CODE"] == name),
            None,
        )

    start = signpost_at(start_signpost, 0)
    stop = None if start is None else signpost_at(stop_signpost, start + 1)
    if stop is None:
        return False
    window = [dict(r) for r in rows[start : stop + 1]]
    # Each device's first replayed op (lowest call count) carries the idle gap from before the replay.
    leading = {}
    for row in window:
        if row.get("OP TYPE") == "signpost":
            continue
        call_count = _float(row.get("GLOBAL CALL COUNT"))
        key = (call_count is None, call_count if call_count is not None else 0)
        device = row.get("DEVICE ID")
        if device not in leading or key < leading[device][0]:
            leading[device] = (key, row)
    for _, row in leading.values():
        for column in GAP_COLUMNS:
            if column in row:
                row[column] = ""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window)
    return True


def _perf_report():
    from tt_perf_report import perf_report

    return perf_report


def _validate_signpost(signpost, expected_edge):
    if not isinstance(signpost, str):
        raise ValueError(f"signpost must be a string: {signpost!r}")
    match = SIGNPOST_PATTERN.fullmatch(signpost)
    if match is None or match.group(4) != expected_edge:
        raise ValueError(f"invalid {expected_edge} signpost: {signpost!r}")
    return signpost


def run_tt_perf_report(ops_csv, start_signpost, stop_signpost, out_csv):
    _validate_signpost(start_signpost, "start")
    _validate_signpost(stop_signpost, "stop")
    perf_report = _perf_report()
    perf_report.set_color_output(False, True)

    # In-process with the CLI's defaults; the library calls sys.exit on unusable input.
    def report(csv_output_file):
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                perf_report.generate_perf_report(
                    csv_files=[str(ops_csv)],
                    start_signpost=start_signpost,
                    end_signpost=stop_signpost,
                    ignore_signposts=False,
                    print_signposts=False,
                    min_percentage=0.5,
                    id_range=None,
                    arch=None,
                    csv_output_file=csv_output_file,
                    no_advice=False,
                    tracing_mode=False,
                    raw_op_codes=False,
                    no_host_ops=False,
                    no_summary=False,
                    group_by="memory",
                    classic_colors=False,
                    summary_file=None,
                    no_stacked_report=False,
                    no_stack_by_in0=False,
                    stacked_csv=None,
                    no_merge_devices=False,
                )
        except (Exception, SystemExit):
            out.write(traceback.format_exc())
            return False, out.getvalue()
        return True, out.getvalue()

    text_ok, text = report(None)
    Path(out_csv).with_suffix(".txt").write_text(text)
    csv_ok, log = report(str(out_csv))
    Path(out_csv).with_suffix(".log").write_text(log)
    return text_ok and csv_ok and Path(out_csv).exists()


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_cell_csv(path, top_n=None):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if _float(r.get("Device Time")) is not None]
    # Rows are not in execution order; the lowest call count ran first and its gap precedes the replay.
    call_counts = [_float(r.get("Global Call Count")) for r in rows]
    if rows and None not in call_counts:
        leading = call_counts.index(min(call_counts))
    else:
        leading = 0
    kernel_us = gap_us = 0.0
    by_op = defaultdict(lambda: [0.0, 0])
    for i, row in enumerate(rows):
        device_us = _float(row["Device Time"])
        kernel_us += device_us
        if i != leading:
            gap_us += _float(row.get("Op-to-Op Gap")) or 0.0
        op = row.get("OP Code", "").strip().lstrip("'")
        by_op[op][0] += device_us
        by_op[op][1] += 1
    n_ops = len(rows)
    ordered_ops = sorted(by_op.items(), key=lambda kv: kv[1][0], reverse=True)
    if top_n is not None and top_n > 0:
        ordered_ops = ordered_ops[:top_n]
    return {
        "n_ops": n_ops,
        "kernel_us": kernel_us,
        "span_us": kernel_us + gap_us,
        "top_ops": [{"op": op, "device_us": us, "count": count} for op, (us, count) in ordered_ops],
    }


def _cell_text(cell):
    host = f"host {cell['measured_ms']:.2f}"
    if cell.get("report") is None:
        return f"report failed<br>{host}"
    r = cell["report"]
    return f"**{r['kernel_us'] / 1000:.2f}**<br>span {r['span_us'] / 1000:.2f}<br>{host}"


def render_markdown(manifests):
    text_size = sum(len(c.get("report_text", "")) for m in manifests for c in m["cells"])
    include_text = text_size <= SUMMARY_TEXT_LIMIT
    lines = []
    for m in manifests:
        shape = "x".join(str(d) for d in m.get("mesh_shape", ()))
        lines += [
            f"### Gemma4 layer perf: {m['run_id']}",
            "",
            f"Context {m.get('context_len')}, chunk {m.get('chunk_size')}, mesh {shape}. Each cell: "
            "**device-kernel time**, span (kernel time plus op-to-op gaps), and host-measured replay, in ms. "
            "Device times are tt-perf-report's device-merged values. The gap before each device's first replayed op "
            "is idle time before the replay and is left out.",
            "",
        ]
        chunks = sorted({c["chunk_idx"] for c in m["cells"]})
        types = list(dict.fromkeys(c["layer_type"] for c in m["cells"]))
        grid = {(c["layer_type"], c["chunk_idx"]): c for c in m["cells"]}
        lines.append("| Layer | " + " | ".join(f"Chunk {i}" for i in chunks) + " |")
        lines.append("|---" * (len(chunks) + 1) + "|")
        for lt in types:
            layer_idx = next(c["layer_idx"] for c in m["cells"] if c["layer_type"] == lt)
            row = [_cell_text(grid[(lt, i)]) if (lt, i) in grid else "–" for i in chunks]
            lines.append(f"| {lt} (layer {layer_idx}) | " + " | ".join(row) + " |")
        lines.append("")
        if not include_text:
            lines += ["tt-perf-report output is too large for the job summary; see the layer-perf artifact.", ""]
            continue
        for c in m["cells"]:
            r = c.get("report")
            if r is None or not c.get("report_text"):
                continue
            lines += [
                f"<details><summary>{c['layer_type']} chunk {c['chunk_idx']}: tt-perf-report ({r['n_ops']} ops)</summary>",
                "",
                "```text",
                c["report_text"].rstrip(),
                "```",
                "",
                "</details>",
                "",
            ]
    return "\n".join(lines) + "\n"


def _report_manifest(m, ops_csv, fieldnames, rows, out_dir, top):
    failed = []
    cell_dir = out_dir / _safe_name(m["run_id"])
    cell_dir.mkdir(parents=True, exist_ok=True)
    for c in m["cells"]:
        out_csv = cell_dir / f"{c['layer_type']}_chunk{c['chunk_idx']}.csv"
        label = f"{m['run_id']} {c['layer_type']} chunk {c['chunk_idx']}"
        cell_ops_csv = out_csv.with_name(f"{out_csv.stem}_ops.csv")
        # tt-perf-report leaves the range open when a signpost is missing, so check before slicing.
        if not write_cell_ops_csv(fieldnames, rows, c["start_signpost"], c["stop_signpost"], cell_ops_csv):
            c["report"] = None
            failed.append(label)
            print(f"{label}: signposts {c['start_signpost']}..{c['stop_signpost']} are not in {ops_csv.name}")
            continue
        ok = run_tt_perf_report(cell_ops_csv, c["start_signpost"], c["stop_signpost"], out_csv)
        c["report"] = summarize_cell_csv(out_csv, top) if ok else None
        text_path = out_csv.with_suffix(".txt")
        if ok and text_path.exists():
            c["report_text"] = text_path.read_text()
        if c["report"] is None or c["report"]["n_ops"] == 0:
            failed.append(label)
            print(f"{label}: no ops reported, see {out_csv.with_suffix('.log')}")
        else:
            print(f"{label}: {c['report']['kernel_us'] / 1000:.2f}ms device kernel")
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiler-dir", default="generated/profiler", help="Tracy artifacts folder (-o)")
    parser.add_argument("--root", type=Path, default=None, help="Summaries root (default: PREFILL_SUMMARIES)")
    parser.add_argument("--top", type=int, default=0, help="Limit ops listed per cell; 0 lists the full table")
    args = parser.parse_args(argv)
    if not is_primary_rank():
        return 0

    root = args.root or summaries_root()
    out_dir = root / MANIFEST_DIR
    manifest_paths = sorted(out_dir.glob("manifest_*.json"))
    manifests = [json.loads(p.read_text()) for p in manifest_paths]
    if not manifests:
        print(f"error: no manifest_*.json under {out_dir}; did the layer-perf test run?", file=sys.stderr)
        return 1
    ops_csvs = find_ops_csvs(args.profiler_dir)
    if not ops_csvs:
        print(f"error: no reports/**/ops_perf_results_*.csv under {args.profiler_dir}", file=sys.stderr)
        return 1
    matched = match_ops_csvs(manifest_paths, manifests, ops_csvs)

    failed = []
    for ops_csv in [p for p in ops_csvs if p in matched]:
        shutil.copy2(ops_csv, out_dir / ops_csv.name)
        print(f"ops CSV: {ops_csv}")
        fieldnames, rows = read_ops_csv(ops_csv)
        for m, m_csv in zip(manifests, matched):
            if m_csv == ops_csv:
                failed += _report_manifest(m, ops_csv, fieldnames, rows, out_dir, args.top)
        del rows
    for m, m_csv in zip(manifests, matched):
        if m_csv is None:
            print(f"{m['run_id']}: no ops CSV newer than its manifest holds its signposts")
            for c in m["cells"]:
                c["report"] = None
                failed.append(f"{m['run_id']} {c['layer_type']} chunk {c['chunk_idx']}")

    (out_dir / "summary.json").write_text(json.dumps(manifests, indent=2))
    summary_dir = root / "perf"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / SUMMARY_NAME).write_text(render_markdown(manifests))
    print(f"summary: {summary_dir / SUMMARY_NAME}")
    if failed:
        print(f"error: no valid report for {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
