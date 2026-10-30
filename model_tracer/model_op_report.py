#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Turn a tracer master JSON into a per-model ops + shapes report.

Writes three files into ``--output-dir``:

- ``model_op_report.json``: the master JSON unchanged, plus ``metadata.report``
  (model, hardware, commit, totals). It keeps the master schema, so it can be
  fed straight to sweep vector generation via ``--master-trace`` /
  ``TTNN_MASTER_JSON_PATH``.
- ``model_op_report.csv``: one row per traced configuration.
- ``model_op_report.md``: markdown summary (also appended to ``--summary``).

No timing is recorded; counts are tracer call counts (Python-level calls,
including warmup/compile, blind to metal-trace replays).
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

CSV_COLUMNS = [
    "operation",
    "config_id",
    "config_hash",
    "count",
    "num_tensor_args",
    "input_shapes",
    "input_dtypes",
    "input_layouts",
    "input_memory",
    "output_memory",
    "mesh_device_shape",
    "hardware",
    "sources",
]


def _tensor_summary(value):
    mem = value.get("memory_config") or {}
    buffer = str(mem.get("buffer_type", "")).replace("BufferType.", "")
    layout = str(mem.get("memory_layout", "")).replace("TensorMemoryLayout.", "")
    return {
        "shape": "x".join(str(d) for d in value.get("original_shape") or []),
        "dtype": str(value.get("original_dtype", "")).replace("DataType.", ""),
        "layout": str(value.get("layout", "")).replace("Layout.", ""),
        "memory": f"{buffer}/{layout}" if buffer or layout else "",
    }


def _collect_tensors(value, out):
    if isinstance(value, dict):
        if value.get("type") == "ttnn.Tensor" and "original_shape" in value:
            out.append(_tensor_summary(value))
            return
        for v in value.values():
            _collect_tensors(v, out)
    elif isinstance(value, list):
        for v in value:
            _collect_tensors(v, out)


def _output_memory(arguments):
    mem = arguments.get("memory_config") if isinstance(arguments, dict) else None
    if not isinstance(mem, dict):
        return ""
    buffer = str(mem.get("buffer_type", "")).replace("BufferType.", "")
    layout = str(mem.get("memory_layout", "")).replace("TensorMemoryLayout.", "")
    return f"{buffer}/{layout}"


def _first_machine_info(executions):
    for execution in executions or []:
        mi = execution.get("machine_info") if isinstance(execution, dict) else None
        if isinstance(mi, list):
            mi = mi[0] if mi else None
        if mi:
            return mi
    return {}


def _hardware_label(mi):
    if not mi:
        return "unknown"
    board = mi.get("board_type") or mi.get("device_series") or "?"
    return f"{board} x{mi.get('card_count', '?')}"


def build_rows(master):
    rows = []
    for op_name, op_data in sorted(master.get("operations", {}).items()):
        for cfg in op_data.get("configurations", []):
            if not isinstance(cfg, dict):
                continue
            executions = cfg.get("executions", []) or []
            tensors = []
            _collect_tensors(cfg.get("arguments", {}), tensors)
            mi = _first_machine_info(executions)
            sources = sorted({e.get("source", "") for e in executions if isinstance(e, dict)} - {""})
            rows.append(
                {
                    "operation": op_name,
                    "config_id": cfg.get("config_id", ""),
                    "config_hash": cfg.get("config_hash", ""),
                    "count": sum(int(e.get("count", 0) or 0) for e in executions if isinstance(e, dict)),
                    "num_tensor_args": len(tensors),
                    "input_shapes": ";".join(t["shape"] for t in tensors),
                    "input_dtypes": ";".join(t["dtype"] for t in tensors),
                    "input_layouts": ";".join(t["layout"] for t in tensors),
                    "input_memory": ";".join(t["memory"] for t in tensors),
                    "output_memory": _output_memory(cfg.get("arguments", {})),
                    "mesh_device_shape": "x".join(str(d) for d in mi.get("mesh_device_shape") or []),
                    "hardware": _hardware_label(mi),
                    "sources": ";".join(sources),
                }
            )
    return rows


def build_report_metadata(master, rows, args):
    mi = {}
    for op_data in master.get("operations", {}).values():
        for cfg in op_data.get("configurations", []):
            if isinstance(cfg, dict):
                mi = _first_machine_info(cfg.get("executions"))
            if mi:
                break
        if mi:
            break
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "test_path": args.test_path or None,
            "hf_model": args.hf_model or None,
            "sources": master.get("metadata", {}).get("models", []),
        },
        "hardware": {
            "runner_label": args.runner_label or None,
            "arch": args.arch or None,
            "machine_info": mi or None,
        },
        "commit": {"sha": args.commit or None, "ref": args.ref or None, "run_url": args.run_url or None},
        "totals": {
            "unique_operations": len({r["operation"] for r in rows}),
            "configurations": len(rows),
            "calls": sum(r["count"] for r in rows),
        },
    }


def render_markdown(meta, rows, title, top_n):
    totals = meta["totals"]
    model = meta["model"]
    hw = meta["hardware"]
    mi = hw.get("machine_info") or {}
    commit = meta["commit"]
    lines = [
        f"## Model op report — `{title}`",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Test path | `{model.get('test_path') or ', '.join(model.get('sources') or []) or '?'}` |",
        f"| HF_MODEL | `{model.get('hf_model') or '-'}` |",
        f"| Runner / arch | {hw.get('runner_label') or '?'} / {hw.get('arch') or '?'} |",
        f"| Hardware | {_hardware_label(mi)} ({mi.get('device_series', '?')}), mesh {mi.get('mesh_device_shape', '?')} |",
        f"| Commit | `{commit.get('sha') or '?'}` |",
        f"| Unique ops / configs / calls | {totals['unique_operations']} / {totals['configurations']} / {totals['calls']} |",
        "",
    ]

    per_op = defaultdict(lambda: [0, 0])
    for r in rows:
        per_op[r["operation"]][0] += 1
        per_op[r["operation"]][1] += r["count"]
    lines += ["### Operations", "", "| Operation | Configs | Calls |", "|---|---:|---:|"]
    for op, (n_cfg, n_calls) in sorted(per_op.items(), key=lambda kv: (-kv[1][1], kv[0])):
        lines.append(f"| `{op}` | {n_cfg} | {n_calls} |")

    lines += [
        "",
        f"### Top {min(top_n, len(rows))} configurations by call count",
        "",
        "| Operation | Calls | Input shapes | Dtypes | Input memory | config_hash |",
        "|---|---:|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (-r["count"], r["operation"]))[:top_n]:
        lines.append(
            f"| `{r['operation']}` | {r['count']} | {r['input_shapes'] or '-'} | {r['input_dtypes'] or '-'} "
            f"| {r['input_memory'] or '-'} | `{str(r['config_hash'])[:12]}` |"
        )
    lines += ["", "Counts are tracer call counts (include warmup/compile; metal-trace replays are not counted).", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("master_json", type=Path, help="Tracer master JSON (ttnn_operations_master.json)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="", help="Report title (default: test path)")
    parser.add_argument("--test-path", default="")
    parser.add_argument("--hf-model", default="")
    parser.add_argument("--runner-label", default="")
    parser.add_argument("--arch", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument("--ref", default="")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--top", type=int, default=25, help="Rows in the top-configurations table")
    parser.add_argument(
        "--summary", type=Path, help="Append the markdown summary to this file (e.g. $GITHUB_STEP_SUMMARY)"
    )
    args = parser.parse_args()

    with open(args.master_json) as f:
        master = json.load(f)
    if not isinstance(master.get("operations"), dict):
        print(f"error: {args.master_json} has no 'operations' mapping", file=sys.stderr)
        return 1

    rows = build_rows(master)
    meta = build_report_metadata(master, rows, args)
    master.setdefault("metadata", {})["report"] = meta

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "model_op_report.json", "w") as f:
        json.dump(master, f, indent=2)
    with open(args.output_dir / "model_op_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    markdown = render_markdown(meta, rows, args.title or args.test_path or args.master_json.stem, args.top)
    (args.output_dir / "model_op_report.md").write_text(markdown)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(markdown)

    t = meta["totals"]
    print(
        f"Wrote report to {args.output_dir}: {t['unique_operations']} ops, "
        f"{t['configurations']} configs, {t['calls']} calls"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
