# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Roll a tracy ops CSV up into prefill zones.

Reads the CSV produced by ``python3 -m tracy -r ...`` on a model's ``tests/perf/profile_prefill.py`` and
reconstructs the zone hierarchy from the ``<PREFIX>_START <name>`` / ``<PREFIX>_END <name>`` signpost rows
(see zones.py). Which prefix, which layer classes and which zones count as communication come from the
model's :class:`~.spec.ZoneSpec`.

How the attribution works: rows appear in host-enqueue order, so the ops between a zone's START and END
signposts are exactly the ops that zone enqueued. Every op row is charged to the innermost open zone
(and, cumulatively, to each enclosing one). Only zones nested under the root zone (``profiled_chunk``)
are reported — that is what excludes the warmup and cache-prefix chunks, whose ops are in the same CSV.

Two views come out of one pass:

* :func:`summarize` — per zone path, per device, then across devices: max / min / skew. The across-
  device MAX is the wall-clock-relevant number for a single zone (the mesh waits for the slowest chip);
  MAX-MIN is the skew, which is what distinguishes a genuinely slow CCL from one that is merely waiting.
* :func:`aggregate_by_class` — per layer class, per layer-relative zone: ms per layer, on ONE device per
  layer (the one with the largest layer total), so the zones of a layer sum exactly to its total. Parents
  get a ``(self)`` bucket for the ops that ran directly in them, so every detail level accounts for 100%.

The CSV is streamed in chunks (a full-model multi-chunk run is ~1M rows), so memory stays flat, and it
is read exactly once — :func:`parse_csv` is the single entry point the CLI and visualize_zones share.

Usage (through a model's shim, e.g. models/demos/gpt_oss_d_p/tests/perf/parse_zone_perf.py):
    python3 parse_zone_perf.py <ops_perf_results_*.csv> [--json out.json] [--top 5]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

import pandas as pd

from models.demos.common.prefill.profiling.spec import ZoneSpec

DURATION_COL = "DEVICE KERNEL DURATION [ns]"
FW_DURATION_COL = "DEVICE FW DURATION [ns]"

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
    FW_DURATION_COL,
    "DRAM BW UTIL (%)",
    "NOC UTIL (%)",
    "NPE CONG IMPACT (%)",
    "PM IDEAL [ns]",
    "MATH FIDELITY",
    "HOST DURATION [ns]",
]

# Ops that did NOT run as a device kernel: anything else inside the forward means the host is doing
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

# The per-layer zone tag the model's layer emits: layer<NN>_<class>. The class is whatever follows the
# underscore, so the parser needs no per-model list to split a tag; the spec only orders the classes.
LAYER_TAG = re.compile(r"^layer(\d+)_(.+)$")
LAYER_TOTAL = "(layer total)"  # the layer zone itself, in the layer-relative view
SELF = "(self)"  # ops that ran directly in a parent zone, not in any captured child


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


def _float(v):
    """Cell -> float, or None for blank / NaN / unparsable."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
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


def split_layer_tag(head):
    """'layer07_full' -> (7, 'full'); None when `head` is not a layer tag."""
    m = LAYER_TAG.match(head)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def layer_class(zone):
    """'profiled_chunk/layer07_full/attn/sdpa' -> ('full', 'attn/sdpa', 7); None for non-layer zones.

    Operates on FULL zone paths (root first): the layer head is parts[1], after the root zone.
    """
    parts = zone.split("/")
    if len(parts) < 2:
        return None
    tag = split_layer_tag(parts[1])
    if tag is None:
        return None
    idx, cls = tag
    return cls, "/".join(parts[2:]), idx


def relative_path(zone):
    """'profiled_chunk/layer07_full/attn/sdpa' -> 'full:attn/sdpa'.

    Collapses the per-layer index so every layer of a class shares one key, which is how the op-detail
    table stays readable with many layers per class.
    """
    lc = layer_class(zone)
    if lc is None:
        return zone
    cls, rel, _ = lc
    return f"{cls}:{rel or LAYER_TOTAL}"


def _parent_rel(rel):
    """Immediate parent of a layer-relative path: 'attn/qkv_proj' -> 'attn', 'attn' -> '' (the layer)."""
    return rel.rsplit("/", 1)[0] if "/" in rel else ""


class ZoneAccumulator:
    """Walks CSV rows in order, tracks the open zone stack, and charges ops to every open zone."""

    def __init__(self, spec: ZoneSpec):
        self.spec = spec
        self.root = spec.root_zone
        self.stack = []  # list of zone names, outermost first
        # (zone_path, device_id) -> stats
        self.stats = defaultdict(lambda: {"ns": 0.0, "ops": 0, "bytes": 0.0, "dram": [], "noc": []})
        # relative zone path -> op_code -> device -> ns. Keyed on the layer-relative path (the layer
        # index stripped) so all layers of a class collapse into one row per zone, and kept per device
        # so the report can quote the worst device rather than a meaningless all-device sum.
        self.op_detail = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.rows_in_root = 0
        # --- capture-integrity counters; anything non-zero here is a warning in every report ---
        self.unmatched_ends = 0  # END with no matching open START (dropped START, or truncated CSV)
        self.ops_no_device_data = 0  # device ops inside the root with an EMPTY duration: profiler overflow
        self.root_opened = False
        self.root_closed = False
        # Host/device-movement audit, all keyed on the layer-relative zone path:
        #   host_ops[zone][op_code] = {"count", "ns"}   ops that did NOT run as a device kernel
        #   movement[zone][label]   = ns                read/write_buffer + CompileProgram child calls
        self.host_ops = defaultdict(lambda: defaultdict(lambda: {"count": 0, "ns": 0.0}))
        self.movement = defaultdict(lambda: defaultdict(float))
        # Whether the CSV even carries the child-call / firmware columns. Without them "no movement"
        # means "not measured", not "none happened" — the report must not conflate the two.
        self.movement_cols_present = False
        self.fw_col_present = False
        # Root-level per-device sums for the device-busy accounting (kernel vs per-op firmware).
        self.root_kernel_ns = defaultdict(float)
        self.root_fw_ns = defaultdict(float)
        self.root_ops = defaultdict(int)
        # Per-op timeline (execution order). Collected for every device; the writer keeps one device,
        # because interleaving 32 chips' copies of the same op destroys the sequential reading.
        self.timeline = []
        self.collect_timeline = False

    @property
    def path(self):
        return "/".join(self.stack)

    @property
    def in_root(self):
        return bool(self.stack) and self.stack[0] == self.root

    @property
    def devices(self):
        return sorted({dev for (_, dev) in self.stats})

    def _signpost(self, name):
        if name.startswith(self.spec.zone_start):
            opened = name[len(self.spec.zone_start) :].strip()
            if not self.stack and opened == self.root:
                self.root_opened = True
            self.stack.append(opened)
        elif name.startswith(self.spec.zone_end):
            ending = name[len(self.spec.zone_end) :].strip()
            if self.stack and self.stack[-1] == ending:
                self.stack.pop()
            elif ending in self.stack:
                # Tolerate a dropped START/END (truncated CSV): unwind to the matching frame.
                while self.stack and self.stack.pop() != ending:
                    pass
                self.unmatched_ends += 1
            else:
                self.unmatched_ends += 1
            if ending == self.root and not self.stack:
                self.root_closed = True

    def feed(self, row, byte_cols):
        op_type = row.get("OP TYPE")
        code = row.get("OP CODE")
        if isinstance(op_type, str) and op_type == "signpost":
            self._signpost(str(code))
            return

        if not self.in_root:
            return  # warmup / prefix-fill op, or an op outside any zone

        rel = relative_path(self.path)

        # Host/device-movement audit. Runs BEFORE the device-duration filter, because the ops we most
        # want to catch — CPU fallbacks, host ops, buffer transfers — are exactly the ones with no
        # DEVICE KERNEL DURATION. A clean device-only forward produces nothing here.
        is_device_op = isinstance(op_type, str) and op_type == DEVICE_OP_TYPE
        if not is_device_op:
            e = self.host_ops[rel][f"{code} [{op_type}]"]
            e["count"] += 1
            e["ns"] += _float(row.get("HOST DURATION [ns]")) or 0.0
        for col, label in HOST_MOVEMENT_COLS.items():
            v = _float(row.get(col))
            if v is not None and v > 0:
                self.movement[rel][label] += v

        dur = _float(row.get(DURATION_COL))
        if dur is None:
            if is_device_op:
                # A device op with no kernel time is what a device-profiler buffer overflow looks like:
                # the op ran, its markers were dropped. Counted so the report can say the totals are low.
                self.ops_no_device_data += 1
            return
        dev = row.get("DEVICE ID")
        try:
            dev = int(dev)
        except (TypeError, ValueError):
            return

        self.rows_in_root += 1
        nbytes = row_bytes(row, byte_cols)
        dram = _float(row.get("DRAM BW UTIL (%)"))
        noc = _float(row.get("NOC UTIL (%)"))
        # Charge to the innermost zone and every enclosing one, so a parent's total always covers
        # its children plus whatever ops it ran directly.
        for depth in range(1, len(self.stack) + 1):
            key = ("/".join(self.stack[:depth]), dev)
            s = self.stats[key]
            s["ns"] += dur
            s["ops"] += 1
            s["bytes"] += nbytes
            if dram is not None:
                s["dram"].append(dram)
            if noc is not None:
                s["noc"].append(noc)
        self.op_detail[rel][str(code)][dev] += dur
        self.root_kernel_ns[dev] += dur
        self.root_ops[dev] += 1
        fw = _float(row.get(FW_DURATION_COL)) if self.fw_col_present else None
        if fw is not None:
            self.root_fw_ns[dev] += fw
        if self.collect_timeline:
            self.timeline.append({"dev": dev, "code": str(code), "zone": self.path, "ns": dur, "bytes": nbytes})

    def warnings(self):
        """Capture-integrity problems, as report-ready sentences. Empty for a clean, complete capture."""
        w = []
        if not self.root_opened:
            w.append(
                f"no `{self.root}` zone found — was {self.spec.zones_env}=1 set, and did the run reach the "
                "profiled chunk? (Rows outside the root zone are ignored by design.)"
            )
        elif not self.root_closed or self.stack:
            still_open = ", ".join(self.stack[-3:]) or self.root
            w.append(
                f"capture TRUNCATED: {max(len(self.stack), 1)} zone(s) still open at the end of the CSV "
                f"(innermost: {still_open}) — the profiled chunk did not finish; every total is partial"
            )
        if self.unmatched_ends:
            w.append(
                f"{self.unmatched_ends} unmatched zone END marker(s) — a START was dropped or the CSV is "
                "truncated; attribution around them is unreliable"
            )
        if self.ops_no_device_data:
            w.append(
                f"{self.ops_no_device_data} device op(s) inside the profiled chunk have no "
                f"{DURATION_COL} — the device profiler buffer overflowed (raise "
                "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT or drain more often); per-zone totals are too low"
            )
        return w


def summarize(acc):
    """Collapse per-(zone, device) stats into per-zone across-device aggregates (max / min / skew)."""
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


def _new_class_entry():
    return {"ns": 0.0, "layers": set(), "ops": 0, "bytes": 0.0, "gbs": [], "dram": [], "noc": []}


def aggregate_by_class(acc, summary):
    """Per layer class, per layer-relative zone: totals over the sampled layers, each layer read on ONE device.

    For every sampled layer the worst device is the one with the largest layer-total kernel time, and
    every zone of that layer is read on that same device. Summing per-zone across-device maxima instead
    (as `summarize` does for the spread view) mixes chips: when one chip is slowest in `attn` and
    another in `mlp`, the children add up to more than any real layer took.

    Parents get a ``<parent>/(self)`` leaf (``(self)`` for the layer itself) holding the ops that ran
    directly in them and not in any captured child — the norms and residual adds at LEVEL=1, the glue
    between the MoE stages at LEVEL=2 — so the leaves of every detail level sum to the layer total.
    Utilization means (DRAM% / NOC%) are taken from `summary`, i.e. across all devices.
    """
    # (cls, layer idx) -> layer-relative path -> device -> stats
    layers = defaultdict(lambda: defaultdict(dict))
    for (zone, dev), s in acc.stats.items():
        lc = layer_class(zone)
        if lc is None:
            continue
        cls, rel, idx = lc
        layers[(cls, idx)][rel][dev] = s

    agg = defaultdict(lambda: defaultdict(_new_class_entry))
    for (cls, idx), rels in layers.items():
        total = rels.get("")
        if not total:
            continue  # a layer zone whose START was lost: no layer total to anchor on
        worst = max(total, key=lambda d: total[d]["ns"])
        inclusive = {rel: devs[worst] for rel, devs in rels.items() if worst in devs}
        zone_prefix = f"{acc.root}/layer{idx:02d}_{cls}"

        for rel, s in inclusive.items():
            e = agg[cls][rel or LAYER_TOTAL]
            e["ns"] += s["ns"]
            e["ops"] += s["ops"]
            e["bytes"] += s["bytes"]
            e["layers"].add(idx)
            if s["ns"] > 0:
                e["gbs"].append(s["bytes"] / (s["ns"] / 1e9) / 1e9)
            # Only present when the capture ran with --collect-noc-traces AND tt-npe is importable;
            # stays empty otherwise, which the report renders as "not measured" rather than zero.
            zs = summary.get(zone_prefix + (f"/{rel}" if rel else ""), {})
            if zs.get("dram_util") is not None:
                e["dram"].append(zs["dram_util"])
            if zs.get("noc_util") is not None:
                e["noc"].append(zs["noc_util"])

        # Parent-exclusive buckets: inclusive minus the immediate children, on the same device.
        for rel, s in inclusive.items():
            children = [c for c in inclusive if c != rel and _parent_rel(c) == rel]
            if not children:
                continue
            self_ns = s["ns"] - sum(inclusive[c]["ns"] for c in children)
            self_ops = s["ops"] - sum(inclusive[c]["ops"] for c in children)
            self_bytes = s["bytes"] - sum(inclusive[c]["bytes"] for c in children)
            if self_ops <= 0 and self_ns <= 0:
                continue
            e = agg[cls][f"{rel}/{SELF}" if rel else SELF]
            e["ns"] += max(self_ns, 0.0)
            e["ops"] += max(self_ops, 0)
            e["bytes"] += max(self_bytes, 0.0)
            e["layers"].add(idx)
            if self_ns > 0:
                e["gbs"].append(max(self_bytes, 0.0) / (self_ns / 1e9) / 1e9)

    result = {}
    for cls, rels in agg.items():
        result[cls] = {}
        for rel, e in rels.items():
            n = len(e["layers"])
            result[cls][rel] = {
                "ms_total": e["ns"] / 1e6,
                "ms_per_layer": e["ns"] / 1e6 / n if n else 0.0,
                "layers": n,
                "ops_per_layer": e["ops"] / n if n else 0,
                "mib_per_layer": e["bytes"] / 2**20 / n if n else 0.0,
                "gbs_mean": sum(e["gbs"]) / len(e["gbs"]) if e["gbs"] else 0.0,
                "dram_util": sum(e["dram"]) / len(e["dram"]) if e["dram"] else None,
                "noc_util": sum(e["noc"]) / len(e["noc"]) if e["noc"] else None,
            }
    return result


def parse_csv(csv_path, spec: ZoneSpec, *, collect_timeline=False, chunksize=200_000):
    """Stream the ops CSV through a ZoneAccumulator exactly once. Returns (acc, csv_row_count)."""
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    missing = [c for c in BASE_COLS if c not in header]
    if missing:
        raise ValueError(f"{csv_path} is missing expected column(s): {missing}")
    byte_cols = io_byte_columns(header)
    usecols = BASE_COLS + [c for c in OPTIONAL_COLS if c in header]
    usecols += [c for c in HOST_MOVEMENT_COLS if c in header]
    usecols += sorted({c for cols in byte_cols.values() for c in cols.values()})

    acc = ZoneAccumulator(spec)
    acc.collect_timeline = collect_timeline
    acc.movement_cols_present = any(c in header for c in HOST_MOVEMENT_COLS)
    acc.fw_col_present = FW_DURATION_COL in header
    nrows = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        for row in chunk.to_dict("records"):
            acc.feed(row, byte_cols)
        nrows += len(chunk)
    return acc, nrows


def print_warnings(acc, out=sys.stdout):
    for w in acc.warnings():
        print(f"WARNING: {w}", file=out)


def print_report(summary, by_class, acc, spec: ZoneSpec, top=0):
    total = summary.get(spec.root_zone)
    print()
    print("=" * 100)
    if total:
        print(
            f"PROFILED CHUNK — device-kernel time {total['ms_max']:.2f} ms on the worst of "
            f"{total['num_devices']} devices (dev {total['worst_device']}); "
            f"min {total['ms_min']:.2f} ms, skew {total['skew_ms']:.2f} ms; {total['ops']} ops"
        )
    print("=" * 100)
    print_warnings(acc)

    for lcls in spec.layer_classes:
        cls = lcls.key
        if cls not in by_class:
            continue
        rels = by_class[cls]
        nlayers = max((v["layers"] for v in rels.values()), default=0)
        cls_total = rels.get(LAYER_TOTAL, {}).get("ms_total", 0.0)
        print()
        print(f"--- {lcls.label.upper()} ({nlayers} layer(s), {cls_total:.2f} ms total device-kernel) ---")
        print(
            f"  {'zone':<34} {'ms/layer':>9} {'ms total':>9} {'% class':>8} "
            f"{'ops/L':>6} {'MiB/L':>9} {'GB/s':>8}  kind"
        )
        print(f"  {'-'*34} {'-'*9} {'-'*9} {'-'*8} {'-'*6} {'-'*9} {'-'*8}  {'-'*7}")
        for rel, v in sorted(rels.items(), key=lambda kv: -kv[1]["ms_total"]):
            pct = (100.0 * v["ms_total"] / cls_total) if cls_total else 0.0
            kind = "" if rel == LAYER_TOTAL else spec.cat(rel)
            print(
                f"  {rel:<34} {v['ms_per_layer']:>9.3f} {v['ms_total']:>9.2f} {pct:>7.1f}% "
                f"{v['ops_per_layer']:>6.1f} {v['mib_per_layer']:>9.1f} {v['gbs_mean']:>8.1f}  {kind}"
            )

    # Per-layer detail for the first layer of each class, with the per-zone across-chip skew.
    for lcls in spec.layer_classes:
        want = lcls.key
        zones = sorted(
            (z for z in summary if (lc := layer_class(z)) and lc[0] == want),
            key=lambda z: (layer_class(z)[2], z),
        )
        if not zones:
            continue
        first_idx = layer_class(zones[0])[2]
        print()
        print(f"--- first {want} layer (layer {first_idx}): per-zone worst chip and skew ---")
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
            print("  HOST<->DEVICE MOVEMENT / PROGRAM-CACHE MISSES inside the chunk — this capture is NOT clean:")
            print(f"  {'zone':<40} {'movement':<32} {'ms':>9}")
            print(f"  {'-'*40} {'-'*32} {'-'*9}")
            rows = [(z, lbl, ns) for z, m in acc.movement.items() for lbl, ns in m.items()]
            for z, lbl, ns in sorted(rows, key=lambda r: -r[2])[:25]:
                print(f"  {z:<40} {lbl:<32} {ns/1e6:>9.3f}")
        elif not acc.movement_cols_present:
            print()
            print("  (buffer transfers / CompileProgram NOT measured: no *_TT_HOST_FUNC columns — pass")
            print("   --child-functions HWCommandQueue_write_buffer,HWCommandQueue_read_buffer,CompileProgram")
            print("   to `python -m tracy` to measure them)")

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
        print("--- top ops by device-kernel time on the worst device, per zone (leaf zones) ---")
        for zone in sorted(acc.op_detail, key=zone_worst_ms, reverse=True)[:12]:
            ops = sorted(
                ((code, max(by_dev.values())) for code, by_dev in acc.op_detail[zone].items()),
                key=lambda kv: -kv[1],
            )[:top]
            print(f"  {zone:<48} {zone_worst_ms(zone):>9.2f} ms")
            for code, ns in ops:
                print(f"      {code:<52} {ns/1e6:>9.2f} ms")


def main(spec: ZoneSpec, argv=None):
    ap = argparse.ArgumentParser(description=f"Roll a tracy ops CSV up into {spec.model_name} prefill zones")
    ap.add_argument("csv", help="ops_perf_results_*.csv from `python3 -m tracy -r ...`")
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
    args = ap.parse_args(argv)

    try:
        acc, nrows = parse_csv(args.csv, spec, collect_timeline=bool(args.timeline), chunksize=args.chunksize)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    summary = summarize(acc)
    by_class = aggregate_by_class(acc, summary)
    meta = f"{args.csv} — {nrows} CSV rows, {acc.rows_in_root} inside `{spec.root_zone}`, {len(summary)} zones"
    print(f"[parse] {meta}")
    print_report(summary, by_class, acc, spec, top=args.top)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"meta": meta, "warnings": acc.warnings(), "zones": summary, "by_class": by_class},
                f,
                indent=2,
                default=str,
            )
        print(f"\n[parse] json -> {args.json}")
    if args.per_device:
        # zone -> {device -> ms}, restricted to zones inside a layer so the view is per-layer-class.
        out = {}
        for (zone, dev), st in acc.stats.items():
            if not zone.startswith(spec.root_zone + "/"):
                continue
            out.setdefault(zone, {})[str(dev)] = round(st["ns"] / 1e6, 5)
        with open(args.per_device, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"[parse] per-device ({len(out)} zones) -> {args.per_device}")

    if args.timeline:
        dev = args.timeline_device
        if dev is None:
            root = summary.get(spec.root_zone)
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

    # No root zone means nothing was profiled: that is a failed capture, not an empty report.
    return 0 if acc.root_opened else 1
