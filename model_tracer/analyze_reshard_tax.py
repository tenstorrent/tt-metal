# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Reshard Tax Analyzer (read-only, per tenstorrent/tt-metal#50943).

Ranks sharded-support gaps by their real "reshard tax" -- aggregate device time lost
to sharded->interleaved->sharded fallback round-trips -- across traced production models.
This is discovery + prioritization only; it does NOT generate program factories or kernels.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Reuse the master-JSON tensor-spec parsing helpers rather than reimplementing them.
try:
    from analyze_operations import (
        load_master_file,
        format_argument,
        format_shard_config,
        format_grid,
    )
except ImportError:
    # Allow running from the repo root (python model_tracer/analyze_reshard_tax.py ...)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_operations import (  # noqa: E402
        load_master_file,
        format_argument,
        format_shard_config,
        format_grid,
    )


DEFAULT_MASTER_FILE = "./model_tracer/traced_operations/ttnn_operations_master.json"

# Device-op OP CODE strings that mark a layout-conversion / staging event.
# The struct-name forms are what device_operation.hpp emits live; the spaced human-readable
# forms are matched defensively in case older profiler output uses them.
SHARDED_TO_INTERLEAVED_MARKERS = ("shardedtointerleaveddeviceoperation", "sharded to interleaved")
INTERLEAVED_TO_SHARDED_MARKERS = ("interleavedtoshardeddeviceoperation", "interleaved to sharded")
RESHARD_MARKERS = ("resharddeviceoperation", "reshard")

# Known sharded-support gap ops (from the issue). Used to recognize the "responsible" op
# in a fallback triple; unrecognized ops are still reported, not dropped.
KNOWN_GAP_OPS = (
    "permute",
    "gather",
    "scatter",
    "sort",
    "fold",
    "split",
    "stack",
    "transpose",
    "pad",
    "slice",
    "repeat",
)

# Duration columns. Prefer the combined FW duration; fall back to the Quasar DM+TRISC split.
FW_DURATION_COL = "DEVICE FW DURATION [ns]"
QUASAR_DM_FW_COL = "DEVICE DM FW DURATION [ns]"
QUASAR_TRISC_FW_COL = "DEVICE TRISC FW DURATION [ns]"


def _norm(value: Optional[str]) -> str:
    """Lowercase + strip whitespace for tolerant marker matching."""
    return (value or "").strip().lower()


def _is_sharded_to_interleaved(op_code: str) -> bool:
    n = _norm(op_code)
    return any(m in n for m in SHARDED_TO_INTERLEAVED_MARKERS)


def _is_interleaved_to_sharded(op_code: str) -> bool:
    n = _norm(op_code)
    return any(m in n for m in INTERLEAVED_TO_SHARDED_MARKERS)


def _is_reshard(op_code: str) -> bool:
    n = _norm(op_code)
    # A pure reshard, but not the two staging halves above (which also contain "sharded").
    if _is_sharded_to_interleaved(op_code) or _is_interleaved_to_sharded(op_code):
        return False
    return any(m in n for m in RESHARD_MARKERS)


def _is_any_staging(op_code: str) -> bool:
    return _is_sharded_to_interleaved(op_code) or _is_interleaved_to_sharded(op_code) or _is_reshard(op_code)


def _match_gap_op(op_code: str) -> Optional[str]:
    """Return the known gap-op name if op_code plausibly refers to one, else None."""
    n = _norm(op_code)
    for gap in KNOWN_GAP_OPS:
        if gap in n:
            return gap
    return None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _row_duration_ns(row: Dict[str, str]) -> float:
    """Device FW duration for a row, tolerant of missing columns and the Quasar split."""
    primary = _to_float(row.get(FW_DURATION_COL))
    if primary is not None:
        return primary
    dm = _to_float(row.get(QUASAR_DM_FW_COL))
    trisc = _to_float(row.get(QUASAR_TRISC_FW_COL))
    if dm is not None or trisc is not None:
        return (dm or 0.0) + (trisc or 0.0)
    return 0.0


def _row_int(row: Dict[str, str], key: str) -> Optional[int]:
    raw = _to_float(row.get(key))
    return int(raw) if raw is not None else None


def _shape_signature_from_expanded(row: Dict[str, str], io_type: str, index: int = 0) -> Optional[List[int]]:
    """Recover a shape from the expanded per-tensor CSV columns (INPUT_0_W_PAD[LOGICAL] ...)."""
    dims = []
    found = False
    for field in ("W", "Z", "Y", "X"):
        col = f"{io_type}_{index}_{field}_PAD[LOGICAL]"
        val = _to_float(row.get(col))
        if val is not None:
            found = True
            dims.append(int(val))
        else:
            dims.append(None)
    if not found:
        return None
    # Trim leading None/degenerate dims but keep the meaningful tail.
    return [d for d in dims if d is not None]


def _memory_signature_from_expanded(row: Dict[str, str], io_type: str, index: int = 0) -> Optional[str]:
    """Recover a memory/shard layout string (e.g. DEV_0_L1_HEIGHT_SHARDED) from expanded columns."""
    mem = row.get(f"{io_type}_{index}_MEMORY")
    if mem is None:
        return None
    mem = mem.strip()
    return mem or None


def _extract_layout_token(memory_sig: Optional[str]) -> str:
    """Pull the TensorMemoryLayout token from a MEMORY string like DEV_0_L1_HEIGHT_SHARDED."""
    if not memory_sig:
        return "UNKNOWN"
    n = memory_sig.upper()
    for layout in ("HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED", "INTERLEAVED", "SINGLE_BANK"):
        if layout in n:
            return layout
    return "UNKNOWN"


def _parse_generic_io_column(cell: Optional[str]) -> Optional[str]:
    """Fallback parse of a single INPUTS/OUTPUTS cell (legacy/alt profiler formats).

    Returns a compact layout token if a sharded/interleaved layout can be recognized.
    """
    if not cell:
        return None
    n = cell.upper()
    for layout in ("HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED", "INTERLEAVED"):
        if layout in n:
            return layout
    return None


def _responsible_op_signature(row: Dict[str, str]) -> Dict[str, Any]:
    """Build a shape/layout/dtype signature for the responsible (enclosed) op from its own row."""
    in_shape = _shape_signature_from_expanded(row, "INPUT", 0)
    out_shape = _shape_signature_from_expanded(row, "OUTPUT", 0)
    in_mem = _memory_signature_from_expanded(row, "INPUT", 0)
    out_mem = _memory_signature_from_expanded(row, "OUTPUT", 0)

    # Fallback to the single-cell INPUTS/OUTPUTS columns if the expanded ones are absent.
    if in_mem is None:
        in_mem = _parse_generic_io_column(row.get("INPUTS"))
    if out_mem is None:
        out_mem = _parse_generic_io_column(row.get("OUTPUTS"))

    in_layout = _extract_layout_token(in_mem)
    out_layout = _extract_layout_token(out_mem)
    dtype = row.get("INPUT_0_DATATYPE") or ""

    # The shard layout "involved" in the fallback: prefer a sharded token on either side.
    shard_layout = "UNKNOWN"
    for candidate in (in_layout, out_layout):
        if candidate not in ("UNKNOWN", "INTERLEAVED"):
            shard_layout = candidate
            break
    if shard_layout == "UNKNOWN":
        # No sharded token found; fall back to whatever input layout we saw.
        shard_layout = in_layout

    return {
        "input_shape": in_shape,
        "output_shape": out_shape,
        "input_memory": in_mem,
        "output_memory": out_mem,
        "dtype": dtype.strip(),
        "shard_layout": shard_layout,
    }


def _shard_layout_from_staging(staging_rows: List[Dict[str, str]]) -> str:
    """Recover the sharded layout token from staging ops (S2I input / I2S output sides)."""
    for r in staging_rows:
        for io, idx in (("INPUT", 0), ("OUTPUT", 0)):
            mem = _memory_signature_from_expanded(r, io, idx)
            token = _extract_layout_token(mem)
            if token not in ("UNKNOWN", "INTERLEAVED"):
                return token
    return "UNKNOWN"


def _shape_str(shape: Optional[List[int]]) -> str:
    if not shape:
        return "?"
    return "x".join(str(d) for d in shape)


def load_ops_perf_csv(path: str, debug: bool = False) -> List[Dict[str, str]]:
    """Load one ops_perf_results.csv into a list of row dicts. Tolerant of empty/odd files."""
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]
    except Exception as e:
        print(f"⚠️  Could not read CSV {path}: {e}")
        return []

    if not rows:
        print(f"⚠️  CSV has no data rows: {path}")
        return []

    if "OP CODE" not in rows[0]:
        print(f"⚠️  CSV missing 'OP CODE' column, skipping: {path}")
        return []

    if debug:
        print(f"🔧 Loaded {len(rows)} rows from {path}")
        if FW_DURATION_COL not in rows[0] and QUASAR_DM_FW_COL not in rows[0]:
            print(f"⚠️  Neither '{FW_DURATION_COL}' nor Quasar DM/TRISC split present in {path}")
    return rows


def order_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Order the device-op stream by GLOBAL CALL COUNT, partitioned per DEVICE ID.

    Returns a flat list that concatenates each device's ordered stream, so fallback
    detection never spans a device boundary. Rows without GLOBAL CALL COUNT keep input order.
    """
    per_device: Dict[Any, List[Tuple[int, int, Dict[str, str]]]] = {}
    for idx, row in enumerate(rows):
        dev = row.get("DEVICE ID", "0")
        gcc = _row_int(row, "GLOBAL CALL COUNT")
        # Use input index as a stable tiebreaker / fallback when GLOBAL CALL COUNT is missing.
        sort_key = gcc if gcc is not None else idx
        per_device.setdefault(dev, []).append((sort_key, idx, row))

    ordered: List[Dict[str, str]] = []
    for dev in sorted(per_device.keys(), key=lambda d: str(d)):
        stream = sorted(per_device[dev], key=lambda t: (t[0], t[1]))
        ordered.extend(row for _, _, row in stream)
    return ordered


def detect_fallback_events(ordered_rows: List[Dict[str, str]], debug: bool = False) -> List[Dict[str, Any]]:
    """Walk the ordered device-op stream and detect reshard/fallback events.

    Two signatures:
      * ShardedToInterleaved -> 1+ intervening ops -> InterleavedToSharded  (a fallback round-trip)
      * a lone ReshardDeviceOperation                                        (pure shard-geometry reshard)

    The "reshard tax" of a fallback triple is the summed device time over the whole span
    (both staging halves plus the enclosed op(s)). The responsible op is the enclosed op.
    """
    events: List[Dict[str, Any]] = []
    n = len(ordered_rows)
    i = 0
    while i < n:
        row = ordered_rows[i]
        op_code = row.get("OP CODE", "")

        if _is_sharded_to_interleaved(op_code):
            # Look ahead for the matching InterleavedToSharded, collecting enclosed ops.
            j = i + 1
            enclosed: List[Dict[str, str]] = []
            closed = False
            while j < n:
                inner = ordered_rows[j]
                inner_code = inner.get("OP CODE", "")
                if _is_interleaved_to_sharded(inner_code):
                    closed = True
                    break
                if _is_sharded_to_interleaved(inner_code):
                    # A new fallback opens before this one closed; treat prior span as incomplete.
                    break
                enclosed.append(inner)
                j += 1

            if closed and enclosed:
                span = [ordered_rows[i]] + enclosed + [ordered_rows[j]]
                tax = sum(_row_duration_ns(r) for r in span)
                # The responsible op is the enclosed op that isn't itself a staging op.
                responsible = None
                for r in enclosed:
                    if not _is_any_staging(r.get("OP CODE", "")):
                        responsible = r
                        break
                if responsible is None:
                    responsible = enclosed[0]
                # The real shard geometry lives on the staging ops (the enclosed op runs on the
                # interleaved staging tensors), so pass them in to recover the sharded layout.
                staging_rows = [ordered_rows[i], ordered_rows[j]]
                events.append(
                    _build_event(responsible, tax, "fallback_triple", span_len=len(span), staging_rows=staging_rows)
                )
                if debug:
                    print(
                        f"🔧 Fallback triple @gcc={responsible.get('GLOBAL CALL COUNT')}: "
                        f"{responsible.get('OP CODE')} tax={tax:.0f}ns span={len(span)}"
                    )
                i = j + 1
                continue
            # Unclosed staging op: count it as a lone staging tax attributed to itself.
            events.append(_build_event(row, _row_duration_ns(row), "staging_only", span_len=1))
            i += 1
            continue

        if _is_reshard(op_code):
            tax = _row_duration_ns(row)
            events.append(_build_event(row, tax, "lone_reshard", span_len=1))
            if debug:
                print(f"🔧 Lone reshard @gcc={row.get('GLOBAL CALL COUNT')}: tax={tax:.0f}ns")
            i += 1
            continue

        i += 1

    return events


def _build_event(
    responsible_row: Dict[str, str],
    tax_ns: float,
    kind: str,
    span_len: int,
    staging_rows: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    op_code = responsible_row.get("OP CODE", "")
    sig = _responsible_op_signature(responsible_row)
    # Prefer a real sharded layout recovered from the staging ops (S2I input / I2S output),
    # since the enclosed op itself runs on interleaved staging tensors during a fallback.
    if sig.get("shard_layout") in ("UNKNOWN", "INTERLEAVED") and staging_rows:
        staged = _shard_layout_from_staging(staging_rows)
        if staged not in ("UNKNOWN", "INTERLEAVED"):
            sig["shard_layout"] = staged
    gap_op = _match_gap_op(op_code)
    return {
        "kind": kind,
        "op_code": op_code,
        "gap_op": gap_op,  # None => unrecognized (still reported)
        "recognized": gap_op is not None,
        "tax_ns": tax_ns,
        "span_len": span_len,
        "signature": sig,
        "global_call_count": responsible_row.get("GLOBAL CALL COUNT"),
        "device_id": responsible_row.get("DEVICE ID"),
    }


def _model_name_for_csv(csv_path: str, explicit: Optional[str]) -> str:
    """Derive a model name for a CSV: explicit override, else its parent directory, else basename."""
    if explicit:
        return explicit
    parent = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
    if parent and parent not in (".", "", "traced_operations"):
        return parent
    return os.path.basename(csv_path)


def _group_key(event: Dict[str, Any]) -> Tuple[str, str, str]:
    """Aggregate key: (responsible op name, shard layout involved, shape signature)."""
    op = event["gap_op"] or event["op_code"] or "unknown"
    layout = event["signature"].get("shard_layout", "UNKNOWN")
    shape = _shape_str(event["signature"].get("input_shape"))
    return (op, layout, shape)


def aggregate_events(
    per_csv_events: List[Tuple[str, List[Dict[str, Any]]]],
    master_ops: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Group events by (op, shard layout, shape) and sum tax + count across all models/CSVs."""
    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for model_name, events in per_csv_events:
        for ev in events:
            key = _group_key(ev)
            g = groups.get(key)
            if g is None:
                g = {
                    "op": key[0],
                    "shard_layout": key[1],
                    "shape": key[2],
                    "total_tax_ns": 0.0,
                    "occurrences": 0,
                    "models": {},  # model_name -> {"count", "tax_ns"}
                    "recognized": ev["recognized"],
                    "kinds": {},
                    "known_config": False,
                    "op_codes": set(),
                }
                groups[key] = g
            g["total_tax_ns"] += ev["tax_ns"]
            g["occurrences"] += 1
            g["op_codes"].add(ev["op_code"])
            g["kinds"][ev["kind"]] = g["kinds"].get(ev["kind"], 0) + 1
            m = g["models"].setdefault(model_name, {"count": 0, "tax_ns": 0.0})
            m["count"] += 1
            m["tax_ns"] += ev["tax_ns"]

    # Secondary enrichment: check whether this op is a known configuration in the master JSON.
    if master_ops:
        known_op_names = _known_op_index(master_ops)
        for g in groups.values():
            g["known_config"] = _op_in_master(g["op"], known_op_names)

    result = list(groups.values())
    for g in result:
        g["op_codes"] = sorted(g["op_codes"])
    result.sort(key=lambda g: g["total_tax_ns"], reverse=True)
    return result


def _known_op_index(master_ops: Dict[str, Any]) -> List[str]:
    """Lowercased list of master-JSON op names for substring membership checks."""
    return [name.lower() for name in master_ops.keys()]


def _op_in_master(op: str, known_op_names: List[str]) -> bool:
    op_l = op.lower()
    return any(op_l in name or name.endswith("::" + op_l) for name in known_op_names)


def _models_for_master_op(op: str, master_ops: Dict[str, Any]) -> List[str]:
    """Best-effort model membership from the master JSON executions[].source for an op."""
    models = set()
    op_l = op.lower()
    for name, op_data in master_ops.items():
        if op_l not in name.lower():
            continue
        for config in op_data.get("configurations", []):
            for execution in config.get("executions", []):
                source = execution.get("source", "")
                if not source:
                    continue
                m = re.search(r"\[HF_MODEL:([^\]]+)\]", source)
                if m:
                    models.add(m.group(1))
                else:
                    models.add(os.path.basename(source))
    return sorted(models)


def print_report(groups: List[Dict[str, Any]], top_n: int, total_events: int, master_ops: Optional[Dict[str, Any]]):
    """Print the ranked reshard-tax table to stdout."""
    print("\n📊 RESHARD TAX RANKING (aggregate device time lost to sharded fallback round-trips)")
    print("=" * 100)
    print(f"   Total fallback/reshard events detected: {total_events}")
    print(f"   Distinct (op, shard layout, shape) gap groups: {len(groups)}")
    print(f"   Showing top {min(top_n, len(groups))} by total device time (reshard tax)\n")

    if not groups:
        print("   ✅ No reshard/fallback events detected in the provided profiler output.")
        return

    for rank, g in enumerate(groups[:top_n], 1):
        models = sorted(g["models"].keys())
        recognized = "known-gap" if g["recognized"] else "UNRECOGNIZED"
        known_cfg = " [in-master]" if g["known_config"] else ""
        print(
            f"#{rank:>2}  op {g['op']} ({recognized}), shard layout {g['shard_layout']}, "
            f"shapes {g['shape']}"
        )
        print(
            f"      → {g['total_tax_ns']:.0f} ns / {g['occurrences']} occurrences "
            f"across models {{{', '.join(models)}}}{known_cfg}"
        )
        # Per-model breakdown for evidence.
        for model in models:
            md = g["models"][model]
            print(f"        · {model}: {md['tax_ns']:.0f} ns / {md['count']} occ")
        if not g["recognized"] and g["op_codes"]:
            print(f"        · op codes: {', '.join(g['op_codes'])}")
    print()


def build_json_report(
    groups: List[Dict[str, Any]],
    top_n: int,
    total_events: int,
    csv_paths: List[str],
    master_ops: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Machine-readable version of the ranked report."""
    ranked = []
    for rank, g in enumerate(groups[:top_n], 1):
        entry = {
            "rank": rank,
            "op": g["op"],
            "recognized_gap_op": g["recognized"],
            "shard_layout": g["shard_layout"],
            "shape": g["shape"],
            "total_tax_ns": g["total_tax_ns"],
            "occurrences": g["occurrences"],
            "models": {
                m: {"count": md["count"], "tax_ns": md["tax_ns"]} for m, md in sorted(g["models"].items())
            },
            "kinds": g["kinds"],
            "op_codes": g["op_codes"],
            "in_master_json": g["known_config"],
        }
        if master_ops:
            entry["master_model_membership"] = _models_for_master_op(g["op"], master_ops)
        ranked.append(entry)

    return {
        "schema": "reshard_tax_ranking_v1",
        "issue": "tenstorrent/tt-metal#50943",
        "read_only": True,
        "total_events": total_events,
        "distinct_groups": len(groups),
        "top_n": top_n,
        "csv_sources": csv_paths,
        "ranking": ranked,
    }


def _expand_csv_args(raw: List[str]) -> List[Tuple[str, Optional[str]]]:
    """Expand --ops-perf-csv args into (csv_path, optional_model_name) pairs.

    Each arg may be:
      * a file path
      * a directory (globbed for ops_perf_results*.csv, else *.csv)
      * a glob pattern
      * any of the above prefixed with 'model_name:' as 'path:model_name' (name is the suffix)
    """
    pairs: List[Tuple[str, Optional[str]]] = []
    for item in raw:
        model_override: Optional[str] = None
        path = item
        # Support 'path:model_name'. Only treat the LAST ':' as a separator, and only if the
        # left side exists as a path/glob (so Windows-style or url-ish paths aren't mangled).
        if ":" in item:
            left, right = item.rsplit(":", 1)
            if left and (os.path.exists(left) or glob.glob(left) or os.path.isdir(left)):
                path, model_override = left, right

        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, "ops_perf_results*.csv")))
            if not found:
                found = sorted(glob.glob(os.path.join(path, "*.csv")))
            for f in found:
                pairs.append((f, model_override))
        elif any(ch in path for ch in "*?[") and not os.path.exists(path):
            for f in sorted(glob.glob(path)):
                pairs.append((f, model_override))
        else:
            pairs.append((path, model_override))
    return pairs


def analyze(
    csv_specs: List[Tuple[str, Optional[str]]],
    master_ops: Optional[Dict[str, Any]],
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    """Run the full pipeline over all CSVs and return (ranked_groups, total_events, csv_paths)."""
    per_csv_events: List[Tuple[str, List[Dict[str, Any]]]] = []
    csv_paths: List[str] = []
    total_events = 0

    for csv_path, model_override in csv_specs:
        rows = load_ops_perf_csv(csv_path, debug=debug)
        if not rows:
            continue
        model_name = _model_name_for_csv(csv_path, model_override)
        ordered = order_rows(rows)
        events = detect_fallback_events(ordered, debug=debug)
        print(f"✅ {csv_path} [{model_name}]: {len(rows)} device ops, {len(events)} fallback/reshard events")
        per_csv_events.append((model_name, events))
        csv_paths.append(csv_path)
        total_events += len(events)

    groups = aggregate_events(per_csv_events, master_ops)
    return groups, total_events, csv_paths


def main():
    parser = argparse.ArgumentParser(
        description="Rank sharded-support gaps by real reshard tax across traced models (read-only, #50943)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model's profiler output
  python model_tracer/analyze_reshard_tax.py --ops-perf-csv path/to/ops_perf_results.csv

  # A directory of CSVs (globs ops_perf_results*.csv), top 30
  python model_tracer/analyze_reshard_tax.py --ops-perf-csv ./profiler_out/ --top 30

  # Multiple CSVs with explicit model names, JSON output
  python model_tracer/analyze_reshard_tax.py \\
      --ops-perf-csv llama_out/ops_perf_results.csv:llama3 \\
      --ops-perf-csv sd_out/ops_perf_results.csv:stable_diffusion \\
      --output reshard_tax.json

  # Enrich with the master trace JSON (optional; used only for membership/known-config hints)
  python model_tracer/analyze_reshard_tax.py --ops-perf-csv ./out/ \\
      --master-file ./model_tracer/traced_operations/ttnn_operations_master.json
        """,
    )
    parser.add_argument(
        "--ops-perf-csv",
        action="append",
        default=[],
        metavar="PATH[:MODEL]",
        help="Path/dir/glob to ops_perf_results.csv (repeatable). Append ':model_name' to label a CSV.",
    )
    parser.add_argument(
        "--master-file",
        default=DEFAULT_MASTER_FILE,
        help=f"Path to master JSON file for enrichment (optional; default: {DEFAULT_MASTER_FILE})",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        metavar="PATH:MODEL",
        help="Alternative way to label CSVs: 'path:model_name' pairs (repeatable).",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of top gap groups to show (default: 20)")
    parser.add_argument("--output", default=None, help="Optional path to write the ranked report as JSON")
    parser.add_argument("--debug", action="store_true", help="Verbose per-event diagnostics")

    args = parser.parse_args()

    if not args.ops_perf_csv:
        print("❌ No profiler output provided. Pass at least one --ops-perf-csv PATH.")
        print("💡 See --help for usage examples.")
        sys.exit(1)

    # Build csv specs from --ops-perf-csv (may carry inline :model) plus explicit --model-name pairs.
    csv_specs = _expand_csv_args(args.ops_perf_csv)

    name_overrides: Dict[str, str] = {}
    for pair in args.model_name:
        if ":" not in pair:
            print(f"⚠️  Ignoring malformed --model-name (expected path:model): {pair}")
            continue
        p, name = pair.rsplit(":", 1)
        name_overrides[os.path.abspath(p)] = name
    if name_overrides:
        csv_specs = [
            (path, name_overrides.get(os.path.abspath(path), override)) for path, override in csv_specs
        ]

    if not csv_specs:
        print("❌ No CSV files resolved from the provided --ops-perf-csv arguments.")
        sys.exit(1)

    # Optional master-file enrichment. Absence is not fatal.
    master_ops: Optional[Dict[str, Any]] = None
    if args.master_file and os.path.exists(args.master_file):
        print(f"📂 Loading master file (enrichment): {args.master_file}")
        try:
            master_data = load_master_file(args.master_file)
            master_ops = master_data.get("operations", {})
            print(f"✅ Loaded {len(master_ops)} operations for enrichment")
        except SystemExit:
            # load_master_file exits on hard errors; degrade to no enrichment instead of dying.
            print("⚠️  Master file could not be loaded; continuing without enrichment.")
            master_ops = None
    else:
        print(f"⚠️  Master file not found ({args.master_file}); continuing without enrichment (profiler-only).")

    print(f"🔧 Analyzing {len(csv_specs)} profiler CSV file(s)...")
    groups, total_events, csv_paths = analyze(csv_specs, master_ops, debug=args.debug)

    print_report(groups, args.top, total_events, master_ops)

    if args.output:
        report = build_json_report(groups, args.top, total_events, csv_paths, master_ops)
        try:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"✅ Wrote JSON report: {args.output}")
        except Exception as e:
            print(f"⚠️  Failed to write JSON report to {args.output}: {e}")


if __name__ == "__main__":
    main()
