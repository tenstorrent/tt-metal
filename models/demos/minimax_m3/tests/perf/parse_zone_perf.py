#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Roll a tracy ops CSV up into MiniMax-M3 prefill zones.

Reads the CSV produced by `python3 -m tracy -r ...` on
models/demos/minimax_m3/tests/perf/profile_prefill.py and reconstructs the zone hierarchy from the
`M3_ZONE_START <name>` / `M3_ZONE_END <name>` signpost rows (see utils/profiler_utils.py).

How the attribution works: rows appear in host-enqueue order, so the ops between a zone's START and END
signposts are exactly the ops that zone enqueued. Every op row is charged to the innermost open zone
(and, cumulatively, to each enclosing one). Only zones nested under `profiled_chunk` are reported —
that is what excludes the warmup and cache-prefix chunks, whose ops are in the same CSV.

Per zone it reports, for each device separately and then across devices:
  * device-kernel time  (sum of DEVICE KERNEL DURATION [ns])
  * op count
  * bytes moved (in + out, from the per-op shape/dtype columns) and the implied GB/s
  * DRAM BW UTIL (%) / NOC UTIL (%) means when the run used --collect-noc-traces
The across-device MAX is the wall-clock-relevant number (the mesh waits for the slowest chip); MAX-MIN
is the skew, which is what distinguishes a genuinely slow CCL from one that is merely waiting.

Usage:
    python3 parse_zone_perf.py <ops_perf_results_*.csv> [--html report.html] [--json out.json]
    python3 parse_zone_perf.py <csv> --top 5         # + the costliest ops within each zone

The CSV is streamed in chunks (a 60-layer 12-chunk run is ~1M rows), so memory stays flat.
"""

import argparse
import html
import json
import sys
from collections import defaultdict

import pandas as pd

ZONE_START = "M3_ZONE_START"
ZONE_END = "M3_ZONE_END"
ROOT_ZONE = "profiled_chunk"
DURATION_COL = "DEVICE KERNEL DURATION [ns]"

# Bytes per element, including the per-32-element block scale for the block-float formats.
DTYPE_BYTES = {
    "BFLOAT16": 2.0,
    "FLOAT32": 4.0,
    "UINT32": 4.0,
    "INT32": 4.0,
    "UINT16": 2.0,
    "UINT8": 1.0,
    "BFLOAT8_B": 1.0625,  # 1 byte mantissa + 1 exponent per 16 (tile-row) elements
    "BFLOAT4_B": 0.5625,
}

BASE_COLS = ["OP CODE", "OP TYPE", "DEVICE ID", DURATION_COL, "CORE COUNT"]
OPTIONAL_COLS = [
    "DRAM BW UTIL (%)",
    "NOC UTIL (%)",
    "NPE CONG IMPACT (%)",
    "PM IDEAL [ns]",
    "MATH FIDELITY",
    "HOST DURATION [ns]",
]

# Ops that did NOT run as a device kernel: anything here inside the forward means the host is doing
# work (or a fallback ran on CPU) in the middle of what should be a pure device pipeline.
DEVICE_OP_TYPE = "tt_dnn_device"

# Child-call columns tracy adds per op when --child-functions is passed. A non-zero read/write_buffer
# inside the profiled chunk is literal host<->device data movement mid-forward; CompileProgram means a
# program cache miss (i.e. the warmup did not cover this shape).
HOST_MOVEMENT_COLS = {
    "HWCommandQueue_write_buffer_TT_HOST_FUNC [ns]": "H2D write_buffer",
    "HWCommandQueue_read_buffer_TT_HOST_FUNC [ns]": "D2H read_buffer",
    "EnqueueReadBuffer_TT_HOST_FUNC [ns]": "D2H EnqueueReadBuffer",
    "EnqueueWriteBuffer_TT_HOST_FUNC [ns]": "H2D EnqueueWriteBuffer",
    "CompileProgram_TT_HOST_FUNC [ns]": "CompileProgram (cache miss)",
}


def _shape_val(v):
    """'3200[3200]' or '3200' -> 3200; blank -> None."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    s = str(v)
    if "[" in s:
        s = s.split("[")[0]
    try:
        return int(float(s))
    except ValueError:
        return None


def io_byte_columns(header):
    """Group the INPUT_n / OUTPUT_n shape+dtype columns present in the CSV by tensor index."""
    groups = defaultdict(dict)
    for col in header:
        for io in ("INPUT", "OUTPUT"):
            if not col.startswith(io + "_"):
                continue
            parts = col.split("_", 2)
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            idx, field = int(parts[1]), parts[2]
            if field.startswith(("W_PAD", "Z_PAD", "Y_PAD", "X_PAD")):
                groups[(io, idx)][field[0]] = col
            elif field == "DATATYPE":
                groups[(io, idx)]["dtype"] = col
    # keep only fully-specified tensors (all 4 dims + dtype)
    return {k: v for k, v in groups.items() if all(d in v for d in "WZYX") and "dtype" in v}


def row_bytes(row, byte_cols):
    """Bytes this op touched: sum of every input and output tensor's physical size."""
    total = 0.0
    for cols in byte_cols.values():
        dims = [_shape_val(row.get(cols[d])) for d in "WZYX"]
        if any(d is None for d in dims):
            continue
        dtype = str(row.get(cols["dtype"], "")).upper().strip()
        per = DTYPE_BYTES.get(dtype)
        if per is None:
            continue
        n = 1
        for d in dims:
            n *= d
        total += n * per
    return total


def relative_path(zone):
    """'profiled_chunk/layer07_sparse/attn/indexer' -> 'sparse:attn/indexer'.

    Collapses the per-layer index so every layer of a class shares one key, which is how the op-detail
    table stays readable at 57 sparse layers.
    """
    parts = zone.split("/")
    if len(parts) >= 2 and parts[1].startswith("layer"):
        cls = "sparse" if parts[1].endswith("sparse") else "dense"
        return f"{cls}:{'/'.join(parts[2:]) or '(layer total)'}"
    return zone


class ZoneAccumulator:
    """Walks CSV rows in order, tracks the open zone stack, and charges ops to every open zone."""

    def __init__(self):
        self.stack = []  # list of zone names, outermost first
        # (zone_path, device_id) -> stats
        self.stats = defaultdict(lambda: {"ns": 0.0, "ops": 0, "bytes": 0.0, "dram": [], "noc": []})
        # relative zone path -> op_code -> device -> ns. Keyed on the layer-relative path (the layer
        # index stripped) so the 57 sparse layers collapse into one row per zone, and kept per device
        # so the report can quote the worst device rather than a meaningless all-device sum.
        self.op_detail = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.unmatched_ends = 0
        self.rows_in_root = 0
        # Host/device-movement audit, all keyed on the layer-relative zone path:
        #   host_ops[zone][op_code] = {"count", "ns"}   ops that did NOT run as a device kernel
        #   movement[zone][label]   = ns                read/write_buffer + CompileProgram child calls
        self.host_ops = defaultdict(lambda: defaultdict(lambda: {"count": 0, "ns": 0.0}))
        self.movement = defaultdict(lambda: defaultdict(float))
        # Whether the CSV even carries the child-call columns. Without them "no movement" means
        # "not measured", not "none happened" — the report must not conflate the two.
        self.movement_cols_present = False
        # Per-op timeline (execution order). Collected for every device; the writer keeps one device,
        # because interleaving 32 chips' copies of the same op destroys the sequential reading.
        self.timeline = []
        self.collect_timeline = False

    @property
    def path(self):
        return "/".join(self.stack)

    def feed(self, row, byte_cols):
        op_type = row.get("OP TYPE")
        code = row.get("OP CODE")
        if isinstance(op_type, str) and op_type == "signpost":
            name = str(code)
            if name.startswith(ZONE_START):
                self.stack.append(name[len(ZONE_START) :].strip())
            elif name.startswith(ZONE_END):
                ending = name[len(ZONE_END) :].strip()
                if self.stack and self.stack[-1] == ending:
                    self.stack.pop()
                elif ending in self.stack:
                    # Tolerate a dropped START/END (truncated CSV): unwind to the matching frame.
                    while self.stack and self.stack.pop() != ending:
                        pass
                    self.unmatched_ends += 1
                else:
                    self.unmatched_ends += 1
            return

        if not self.stack or self.stack[0] != ROOT_ZONE:
            return  # warmup / prefix-fill op, or an op outside any zone

        rel = relative_path(self.path)

        # Host/device-movement audit. Runs BEFORE the device-duration filter, because the ops we most
        # want to catch — CPU fallbacks, host ops, buffer transfers — are exactly the ones with no
        # DEVICE KERNEL DURATION. A clean device-only forward produces nothing here.
        if isinstance(op_type, str) and op_type != DEVICE_OP_TYPE:
            host_ns = row.get("HOST DURATION [ns]")
            try:
                host_ns = float(host_ns) if host_ns is not None and not pd.isna(host_ns) else 0.0
            except (TypeError, ValueError):
                host_ns = 0.0
            e = self.host_ops[rel][f"{code} [{op_type}]"]
            e["count"] += 1
            e["ns"] += host_ns
        for col, label in HOST_MOVEMENT_COLS.items():
            v = row.get(col)
            if v is None or pd.isna(v):
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v > 0:
                self.movement[rel][label] += v

        dur = row.get(DURATION_COL)
        if dur is None or pd.isna(dur):
            return
        try:
            dur = float(dur)
        except (TypeError, ValueError):
            return
        dev = row.get("DEVICE ID")
        try:
            dev = int(dev)
        except (TypeError, ValueError):
            return

        self.rows_in_root += 1
        nbytes = row_bytes(row, byte_cols)
        dram = row.get("DRAM BW UTIL (%)")
        noc = row.get("NOC UTIL (%)")
        # Charge to the innermost zone and every enclosing one, so a parent's total always covers
        # its children plus whatever ops it ran directly.
        for depth in range(1, len(self.stack) + 1):
            key = ("/".join(self.stack[:depth]), dev)
            s = self.stats[key]
            s["ns"] += dur
            s["ops"] += 1
            s["bytes"] += nbytes
            if dram is not None and not pd.isna(dram):
                s["dram"].append(float(dram))
            if noc is not None and not pd.isna(noc):
                s["noc"].append(float(noc))
        self.op_detail[relative_path(self.path)][str(code)][dev] += dur
        if self.collect_timeline:
            self.timeline.append({"dev": dev, "code": str(code), "zone": self.path, "ns": dur, "bytes": nbytes})


def summarize(acc):
    """Collapse per-(zone, device) stats into per-zone across-device aggregates."""
    per_zone = defaultdict(dict)
    for (zone, dev), s in acc.stats.items():
        per_zone[zone][dev] = s

    out = {}
    for zone, devs in per_zone.items():
        ns = [d["ns"] for d in devs.values()]
        mx, mn = max(ns), min(ns)
        worst = max(devs, key=lambda d: devs[d]["ns"])
        # Bytes/GB-s are reported on the worst device: that is the chip setting the wall clock.
        wb = devs[worst]["bytes"]
        gbs = (wb / (mx / 1e9) / 1e9) if mx > 0 else 0.0
        dram = [v for d in devs.values() for v in d["dram"]]
        noc = [v for d in devs.values() for v in d["noc"]]
        out[zone] = {
            "ms_max": mx / 1e6,
            "ms_min": mn / 1e6,
            "ms_mean": sum(ns) / len(ns) / 1e6,
            "skew_ms": (mx - mn) / 1e6,
            "worst_device": worst,
            "num_devices": len(devs),
            "ops": devs[worst]["ops"],
            "mib": wb / 2**20,
            "gbs": gbs,
            "dram_util": (sum(dram) / len(dram)) if dram else None,
            "noc_util": (sum(noc) / len(noc)) if noc else None,
        }
    return out


def layer_class(zone):
    """'layer07_sparse/attn/indexer' -> ('sparse', 'attn/indexer', 7); None for non-layer zones."""
    parts = zone.split("/")
    if len(parts) < 2 or not parts[1].startswith("layer"):
        return None
    head = parts[1]
    try:
        idx = int(head[5:7])
    except ValueError:
        return None
    cls = "sparse" if head.endswith("sparse") else "dense"
    return cls, "/".join(parts[2:]), idx


def aggregate_by_class(summary):
    """Sum each relative zone path over all layers of a class, and count the layers involved."""
    agg = defaultdict(
        lambda: defaultdict(
            lambda: {"ms": 0.0, "layers": set(), "ops": 0, "mib": 0.0, "gbs": [], "dram": [], "noc": []}
        )
    )
    for zone, s in summary.items():
        lc = layer_class(zone)
        if lc is None:
            continue
        cls, rel, idx = lc
        rel = rel or "(layer total)"
        e = agg[cls][rel]
        e["ms"] += s["ms_max"]
        e["layers"].add(idx)
        e["ops"] += s["ops"]
        e["mib"] += s["mib"]
        if s["gbs"]:
            e["gbs"].append(s["gbs"])
        # Only present when the capture ran with --collect-noc-traces AND tt-npe is importable;
        # stays empty otherwise, which the report renders as "not measured" rather than zero.
        if s.get("dram_util") is not None:
            e["dram"].append(s["dram_util"])
        if s.get("noc_util") is not None:
            e["noc"].append(s["noc_util"])
    result = {}
    for cls, rels in agg.items():
        result[cls] = {}
        for rel, e in rels.items():
            n = len(e["layers"])
            result[cls][rel] = {
                "ms_total": e["ms"],
                "ms_per_layer": e["ms"] / n if n else 0.0,
                "layers": n,
                "ops_per_layer": e["ops"] / n if n else 0,
                "mib_per_layer": e["mib"] / n if n else 0.0,
                "gbs_mean": sum(e["gbs"]) / len(e["gbs"]) if e["gbs"] else 0.0,
                "dram_util": sum(e["dram"]) / len(e["dram"]) if e["dram"] else None,
                "noc_util": sum(e["noc"]) / len(e["noc"]) if e["noc"] else None,
            }
    return result


def print_report(summary, by_class, acc, top=0):
    total = summary.get(ROOT_ZONE)
    print()
    print("=" * 100)
    if total:
        print(
            f"PROFILED CHUNK — device-kernel time {total['ms_max']:.2f} ms on the worst of "
            f"{total['num_devices']} devices (dev {total['worst_device']}); "
            f"min {total['ms_min']:.2f} ms, skew {total['skew_ms']:.2f} ms; {total['ops']} ops"
        )
    else:
        print(f"No `{ROOT_ZONE}` zone found — was M3_PROFILE_ZONES=1 set, and did the run reach the")
        print("profiled chunk? (Rows outside the root zone are ignored by design.)")
    print("=" * 100)
    if acc.unmatched_ends:
        print(f"WARNING: {acc.unmatched_ends} unmatched zone END marker(s) — CSV may be truncated.")

    for cls in ("dense", "sparse"):
        if cls not in by_class:
            continue
        rels = by_class[cls]
        nlayers = max((v["layers"] for v in rels.values()), default=0)
        cls_total = rels.get("(layer total)", {}).get("ms_total", 0.0)
        print()
        print(f"--- {cls.upper()} layers ({nlayers} layer(s), {cls_total:.2f} ms total device-kernel) ---")
        print(
            f"  {'zone':<34} {'ms/layer':>9} {'ms total':>9} {'% class':>8} " f"{'ops/L':>6} {'MiB/L':>9} {'GB/s':>8}"
        )
        print(f"  {'-'*34} {'-'*9} {'-'*9} {'-'*8} {'-'*6} {'-'*9} {'-'*8}")
        for rel, v in sorted(rels.items(), key=lambda kv: -kv[1]["ms_total"]):
            pct = (100.0 * v["ms_total"] / cls_total) if cls_total else 0.0
            print(
                f"  {rel:<34} {v['ms_per_layer']:>9.3f} {v['ms_total']:>9.2f} {pct:>7.1f}% "
                f"{v['ops_per_layer']:>6.1f} {v['mib_per_layer']:>9.1f} {v['gbs_mean']:>8.1f}"
            )

    # Per-layer detail for the first dense and first sparse layer — the two the profile is aimed at.
    for want in ("dense", "sparse"):
        zones = sorted(
            (z for z in summary if (lc := layer_class(z)) and lc[0] == want),
            key=lambda z: (layer_class(z)[2], z),
        )
        if not zones:
            continue
        first_idx = layer_class(zones[0])[2]
        print()
        print(f"--- first {want} layer (layer {first_idx}) ---")
        print(f"  {'zone':<44} {'ms':>8} {'skew ms':>8} {'ops':>6} {'MiB':>9} {'GB/s':>8} {'DRAM%':>7}")
        print(f"  {'-'*44} {'-'*8} {'-'*8} {'-'*6} {'-'*9} {'-'*8} {'-'*7}")
        for z in zones:
            if layer_class(z)[2] != first_idx:
                continue
            s = summary[z]
            rel = "/".join(z.split("/")[1:])
            du = f"{s['dram_util']:.1f}" if s["dram_util"] is not None else "-"
            print(
                f"  {rel:<44} {s['ms_max']:>8.3f} {s['skew_ms']:>8.3f} {s['ops']:>6} "
                f"{s['mib']:>9.1f} {s['gbs']:>8.1f} {du:>7}"
            )

    # --- host / device-movement audit -------------------------------------------------------
    print()
    print("--- host work & device<->host movement inside the profiled chunk ---")
    if not acc.host_ops and not acc.movement:
        print("  No non-device ops: every op in the profiled chunk ran as a device kernel (OP TYPE ==")
        print("  tt_dnn_device) — no CPU fallbacks, no host ops.")
        if not acc.movement_cols_present:
            print("  NOT MEASURED: buffer transfers / program-cache misses. Those are child calls, and this")
            print("  CSV has no *_TT_HOST_FUNC columns, so H2D/D2H copies cannot be ruled out from it.")
            print("  Re-run `python -m tracy` with:")
            print("    --child-functions HWCommandQueue_write_buffer,HWCommandQueue_read_buffer,CompileProgram")
        else:
            print("  Also zero buffer transfers and zero CompileProgram calls (both measured).")
    else:
        if acc.host_ops:
            print(f"  {'zone':<40} {'op [type]':<44} {'count':>6} {'host ms':>9}")
            print(f"  {'-'*40} {'-'*44} {'-'*6} {'-'*9}")
            rows = [(z, o, e) for z, ops in acc.host_ops.items() for o, e in ops.items()]
            for z, o, e in sorted(rows, key=lambda r: -r[2]["ns"])[:25]:
                print(f"  {z:<40} {o:<44} {e['count']:>6} {e['ns']/1e6:>9.3f}")
        if acc.movement:
            print()
            print(f"  {'zone':<40} {'movement':<32} {'ms':>9}")
            print(f"  {'-'*40} {'-'*32} {'-'*9}")
            rows = [(z, lbl, ns) for z, m in acc.movement.items() for lbl, ns in m.items()]
            for z, lbl, ns in sorted(rows, key=lambda r: -r[2])[:25]:
                print(f"  {z:<40} {lbl:<32} {ns/1e6:>9.3f}")
        else:
            print()
            print("  (no read/write_buffer or CompileProgram child calls recorded — pass")
            print("   --child-functions HWCommandQueue_write_buffer,HWCommandQueue_read_buffer,CompileProgram")
            print("   to `python -m tracy` to measure buffer transfers explicitly)")

    if top:
        # Per zone (layer index collapsed), the ops that cost the most on the WORST device — summed over
        # every layer of that class, so this is "total ms this op contributed to the profiled chunk".
        def zone_worst_ms(zone):
            per_dev = defaultdict(float)
            for by_dev in acc.op_detail[zone].values():
                for dev, ns in by_dev.items():
                    per_dev[dev] += ns
            return max(per_dev.values()) / 1e6 if per_dev else 0.0

        print()
        print(f"--- top ops by device-kernel time on the worst device, per zone (leaf zones) ---")
        for zone in sorted(acc.op_detail, key=zone_worst_ms, reverse=True)[:12]:
            ops = sorted(
                ((code, max(by_dev.values())) for code, by_dev in acc.op_detail[zone].items()),
                key=lambda kv: -kv[1],
            )[:top]
            print(f"  {zone:<48} {zone_worst_ms(zone):>9.2f} ms")
            for code, ns in ops:
                print(f"      {code:<52} {ns/1e6:>9.2f} ms")


def _html(summary, by_class, meta):
    def esc(s):
        return html.escape(str(s))

    total = summary.get(ROOT_ZONE, {})
    parts = [
        "<style>",
        "body{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:1200px}",
        "h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:2rem}",
        "table{border-collapse:collapse;width:100%;margin:.5rem 0;font-variant-numeric:tabular-nums}",
        "th,td{padding:.35rem .5rem;text-align:right;border-bottom:1px solid #8883}",
        "th:first-child,td:first-child{text-align:left;font-family:ui-monospace,monospace}",
        "th{font-weight:600;border-bottom:2px solid #8886}",
        ".bar{height:.65rem;background:#4a90d9;border-radius:2px;display:inline-block;min-width:1px}",
        ".meta{color:#8889;font-size:.9rem}",
        "@media(prefers-color-scheme:dark){body{background:#111;color:#ddd}.bar{background:#5aa}}",
        "</style>",
        "<h1>MiniMax-M3 prefill — zone profile</h1>",
        f"<p class=meta>{esc(meta)}</p>",
    ]
    if total:
        parts.append(
            f"<p><b>Profiled chunk:</b> {total['ms_max']:.2f} ms device-kernel on the worst of "
            f"{total['num_devices']} devices (dev {total['worst_device']}), "
            f"skew {total['skew_ms']:.2f} ms, {total['ops']} ops.</p>"
        )
    for cls in ("dense", "sparse"):
        if cls not in by_class:
            continue
        rels = by_class[cls]
        cls_total = rels.get("(layer total)", {}).get("ms_total", 0.0) or 1.0
        rows = sorted(rels.items(), key=lambda kv: -kv[1]["ms_total"])
        mx = max((v["ms_per_layer"] for _, v in rows), default=1.0) or 1.0
        nlayers = max((v["layers"] for v in rels.values()), default=0)
        parts.append(f"<h2>{cls} layers <span class=meta>({nlayers} layers)</span></h2>")
        parts.append(
            "<table><tr><th>zone</th><th>ms/layer</th><th>ms total</th><th>% class</th>"
            "<th>ops/layer</th><th>MiB/layer</th><th>GB/s</th><th></th></tr>"
        )
        for rel, v in rows:
            w = 240 * v["ms_per_layer"] / mx
            parts.append(
                f"<tr><td>{esc(rel)}</td><td>{v['ms_per_layer']:.3f}</td><td>{v['ms_total']:.2f}</td>"
                f"<td>{100*v['ms_total']/cls_total:.1f}%</td><td>{v['ops_per_layer']:.1f}</td>"
                f"<td>{v['mib_per_layer']:.1f}</td><td>{v['gbs_mean']:.1f}</td>"
                f"<td><span class=bar style='width:{w:.1f}px'></span></td></tr>"
            )
        parts.append("</table>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Roll a tracy ops CSV up into MiniMax-M3 prefill zones")
    ap.add_argument("csv", help="ops_perf_results_*.csv from `python3 -m tracy -r ...`")
    ap.add_argument("--html", help="write an HTML report here")
    ap.add_argument("--json", help="write the raw per-zone summary here")
    ap.add_argument("--top", type=int, default=0, help="also list the top N ops per zone")
    ap.add_argument(
        "--timeline",
        help="write a per-op execution-order timeline (JSON) for one device: every op in the profiled "
        "chunk with its zone and device-kernel duration",
    )
    ap.add_argument(
        "--per-device",
        help="write per-(zone, device) device-kernel ms (JSON) — the basis for the per-chip imbalance view",
    )
    ap.add_argument(
        "--timeline-device",
        type=int,
        default=None,
        help="device id for --timeline (default: the device with the largest total device-kernel time)",
    )
    ap.add_argument("--chunksize", type=int, default=200_000, help="CSV streaming chunk size")
    args = ap.parse_args()

    header = list(pd.read_csv(args.csv, nrows=0).columns)
    missing = [c for c in BASE_COLS if c not in header]
    if missing:
        sys.exit(f"ERROR: {args.csv} is missing expected column(s): {missing}")
    byte_cols = io_byte_columns(header)
    usecols = BASE_COLS + [c for c in OPTIONAL_COLS if c in header]
    usecols += [c for c in HOST_MOVEMENT_COLS if c in header]
    usecols += sorted({c for cols in byte_cols.values() for c in cols.values()})

    acc = ZoneAccumulator()
    acc.collect_timeline = bool(args.timeline)
    acc.movement_cols_present = any(c in header for c in HOST_MOVEMENT_COLS)
    nrows = 0
    for chunk in pd.read_csv(args.csv, usecols=usecols, chunksize=args.chunksize, low_memory=False):
        for row in chunk.to_dict("records"):
            acc.feed(row, byte_cols)
        nrows += len(chunk)

    summary = summarize(acc)
    by_class = aggregate_by_class(summary)
    meta = f"{args.csv} — {nrows} CSV rows, {acc.rows_in_root} inside `{ROOT_ZONE}`, {len(summary)} zones"
    print(f"[parse] {meta}")
    print_report(summary, by_class, acc, top=args.top)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "zones": summary, "by_class": by_class}, f, indent=2, default=str)
        print(f"\n[parse] json -> {args.json}")
    if args.per_device:
        # zone -> {device -> ms}, restricted to zones inside a layer so the view is per-layer-class.
        out = {}
        for (zone, dev), st in acc.stats.items():
            if not zone.startswith(ROOT_ZONE + "/"):
                continue
            out.setdefault(zone, {})[str(dev)] = round(st["ns"] / 1e6, 5)
        with open(args.per_device, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"[parse] per-device ({len(out)} zones) -> {args.per_device}")

    if args.timeline:
        dev = args.timeline_device
        if dev is None:
            root = summary.get(ROOT_ZONE)
            dev = root["worst_device"] if root else (acc.timeline[0]["dev"] if acc.timeline else 0)
        ops, cum = [], 0.0
        for r in acc.timeline:
            if r["dev"] != dev:
                continue
            ms = r["ns"] / 1e6
            ops.append(
                {
                    "i": len(ops),
                    "code": r["code"],
                    "zone": r["zone"].split("/", 1)[1] if "/" in r["zone"] else r["zone"],
                    "ms": round(ms, 6),
                    "start_ms": round(cum, 6),
                    "mib": round(r["bytes"] / 2**20, 3),
                }
            )
            cum += ms
        with open(args.timeline, "w") as f:
            json.dump({"device": dev, "total_ms": round(cum, 4), "ops": ops}, f)
        print(f"[parse] timeline ({len(ops)} ops on device {dev}, {cum:.2f} ms) -> {args.timeline}")

    if args.html:
        with open(args.html, "w") as f:
            f.write(_html(summary, by_class, meta))
        print(f"[parse] html -> {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
