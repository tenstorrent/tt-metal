#!/usr/bin/env python3
"""Diff what the advisor advised against what the decoder actually ships, and rank by measured cost.

This is the piece that turns "rejected in one sentence" into a numbered row. It is model-independent:
it reads only the advisor's own artifacts plus the incumbent's tt-perf-report CSV.

  reconcile.py --report shard_advise/<kind>/report.json \
               --ir     shard_advise/<kind>/final_ir.mlir \
               --perf   tracy/incumbent_ops.csv \
               --layers-of-kind 30 --total-layers 30 \
               --threshold-pct 1.0 \
               --out    reconciliation.json

What it emits, per op the advisor would place differently than the shipped graph:

  op, advised_layout, advised_program_config, shipped_layout, shipped_cores,
  device_us, window_share_pct, verdict("pending"|"below_threshold"), reason

`verdict` starts as "pending" for every material row: the STAGE has to fill in `measured_ms` and flip it
to kept/rejected. The gate refuses any row still pending, which is what stops a prose rejection from
passing for a result.

Deliberate limitation, stated rather than hidden: the shipped side is read from the perf CSV, so an op the
advisor names but which never appears in the CSV is reported with device_us=None and
`reason="op not present in the incumbent's measured window"`. That is usually a capture/shape mismatch
worth a human look, not a free win.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ---- program configs and layouts live in the IR, not the report ---------------------------------
# report.json gives per-op layout and the program-config FAMILY but omits block widths and per_core_N;
# final_ir.mlir is authoritative for those. So the IR is parsed for geometry and the report for intent.
IR_MATMUL = re.compile(
    r'"ttnn\.(?P<op>matmul|linear|sparse_matmul)".*?'
    r'(?:in0_block_w\s*=\s*(?P<blk>\d+))?.*?'
    r'(?:per_core_N\s*=\s*(?P<pcn>\d+))?',
    re.DOTALL,
)
IR_SHARD = re.compile(
    r'"ttnn\.(?P<op>[a-z_0-9]+)"[^\n]*?'
    r'#(?P<mem>l1|dram)[^\n]*?(?P<layout>block_sharded|width_sharded|height_sharded|interleaved)',
)
IR_GRID = re.compile(r"<(?P<r>\d+)x(?P<c>\d+)>")


def parse_ir(path: Path) -> dict:
    """op-name -> {memory, layout, grid, in0_block_w, per_core_N} as ADVISED."""
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    advised: dict[str, dict] = {}
    for line in text.splitlines():
        m = IR_SHARD.search(line)
        if not m:
            continue
        op = m.group("op")
        e = advised.setdefault(op, {})
        e["memory"] = m.group("mem")
        e["layout"] = m.group("layout")
        g = IR_GRID.search(line)
        if g:
            e["grid"] = f'{g.group("r")}x{g.group("c")}'
            try:
                e["cores"] = int(g.group("r")) * int(g.group("c"))
            except ValueError:
                pass
        for key, pat in (("in0_block_w", r"in0_block_w\s*=\s*(\d+)"),
                         ("per_core_N", r"per_core_N\s*=\s*(\d+)")):
            mm = re.search(pat, line)
            if mm:
                e[key] = int(mm.group(1))
    return advised


def parse_perf(path: Path) -> tuple[dict, float]:
    """op-name -> {device_us, cores, sharded} as SHIPPED, plus the total measured window."""
    if not path.exists():
        return {}, 0.0
    shipped: dict[str, dict] = {}
    total = 0.0
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}, 0.0

    def col(row, *names):
        for n in names:
            for k in row:
                if k and k.strip().lower() == n:
                    return row[k]
        # substring fallback: tt-perf-report column names vary across versions
        for n in names:
            for k in row:
                if k and n in k.strip().lower():
                    return row[k]
        return None

    for row in rows:
        name = (col(row, "op code", "op_code", "op type", "name") or "").strip()
        if not name:
            continue
        raw = col(row, "device kernel duration [ns]", "device kernel duration", "device_kernel_duration_ns")
        try:
            us = float(str(raw).replace(",", "")) / 1000.0
        except (TypeError, ValueError):
            us = 0.0
        cores_raw = col(row, "core count", "core_count", "cores")
        try:
            cores = int(float(str(cores_raw).replace(",", "")))
        except (TypeError, ValueError):
            cores = None
        ds = (col(row, "dram sharded", "dram_sharded") or "").strip().lower() in ("true", "1", "yes")
        key = name.lower().replace("ttnn.", "")
        e = shipped.setdefault(key, {"device_us": 0.0, "cores": cores, "dram_sharded": ds, "n": 0})
        e["device_us"] += us
        e["n"] += 1
        if cores is not None:
            # keep the SMALLEST core count seen: an op left on 1 core in any instance is the finding
            e["cores"] = cores if e["cores"] is None else min(e["cores"], cores)
        e["dram_sharded"] = e["dram_sharded"] or ds
        total += us
    return shipped, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--ir", required=True, type=Path)
    ap.add_argument("--perf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layer-kind", default="unknown")
    ap.add_argument("--layers-of-kind", type=int, default=1)
    ap.add_argument("--total-layers", type=int, default=1)
    ap.add_argument("--threshold-pct", type=float, default=1.0)
    a = ap.parse_args()

    try:
        report = json.loads(a.report.read_text())
    except Exception as e:
        print(f"cannot read {a.report}: {e}", file=sys.stderr)
        return 1

    advised = parse_ir(a.ir)
    shipped, window_us = parse_perf(a.perf)
    if not advised:
        print(f"warning: no advised placements parsed out of {a.ir}", file=sys.stderr)
    if window_us <= 0:
        print(f"warning: measured window is 0 us from {a.perf} -- window_share_pct will be null",
              file=sys.stderr)

    rows = []
    for op, adv in sorted(advised.items()):
        ship = shipped.get(op)
        # The finding is: advisor puts it in SHARDED L1, the shipped graph leaves it interleaved or on
        # <=2 cores. That set difference is the advisor's demonstrated strength; its geometry is not.
        advises_sharded_l1 = adv.get("memory") == "l1" and adv.get("layout", "").endswith("sharded")
        if not advises_sharded_l1:
            continue
        if ship is None:
            rows.append({
                "op": op, "layer_kind": a.layer_kind,
                "advised": adv, "shipped": None,
                "device_us": None, "window_share_pct": None,
                "verdict": "pending",
                "reason": "op not present in the incumbent's measured window -- likely a capture/shape "
                          "mismatch worth a human look, not a free win",
            })
            continue
        ship_cores = ship.get("cores")
        under_sharded = (ship_cores is not None and ship_cores <= 2) or not ship.get("dram_sharded", False)
        if not under_sharded:
            continue
        # cost is per-layer-kind; scale to the whole model so ranking reflects real impact
        per_model_us = ship["device_us"] * (a.layers_of_kind or 1)
        share = (100.0 * ship["device_us"] / window_us) if window_us > 0 else None
        row = {
            "op": op, "layer_kind": a.layer_kind,
            "advised": adv,
            "shipped": {"cores": ship_cores, "dram_sharded": ship.get("dram_sharded"),
                        "device_us": round(ship["device_us"], 3), "instances": ship["n"]},
            "device_us": round(ship["device_us"], 3),
            "per_model_us": round(per_model_us, 3),
            "layers_of_kind": a.layers_of_kind,
            "window_share_pct": None if share is None else round(share, 3),
            "verdict": "pending",
            "reason": None,
        }
        if share is not None and share < a.threshold_pct:
            row["verdict"] = "below_threshold"
            row["reason"] = f"{share:.3f}% of the measured window, under the {a.threshold_pct}% threshold"
        rows.append(row)

    rows.sort(key=lambda r: (r["window_share_pct"] or 0.0), reverse=True)
    out = {
        "layer_kind": a.layer_kind,
        "layers_of_kind": a.layers_of_kind,
        "total_layers": a.total_layers,
        "measured_window_us": round(window_us, 3),
        "threshold_pct": a.threshold_pct,
        "dram_sharded_considered": report.get("dram_sharded_considered"),
        "dram_sharded_advised": report.get("dram_sharded_advised"),
        "note": "verdict=pending rows MUST be measured and flipped to kept/rejected with a measured_ms; "
                "the stage gate refuses any row left pending. Advisor GEOMETRY is not evidence -- derive "
                "your own, or sweep its value in both directions.",
        "disagreements": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    material = [r for r in rows if r["verdict"] == "pending"]
    print(f"{a.layer_kind}: {len(rows)} disagreement(s), {len(material)} above the "
          f"{a.threshold_pct}% threshold -> {a.out}")
    for r in material:
        print(f"  {r['op']}: shipped {r['shipped']} ({r['window_share_pct']}% of window) "
              f"vs advised {r['advised'].get('layout')} {r['advised'].get('grid')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
