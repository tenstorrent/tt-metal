#!/usr/bin/env python3
"""Reconcile the advisor's placement against the shipped graph, and account for every device op.

  reconcile.py --report shard_advise/<kind>/report.json \
               --perf   tracy/incumbent_<kind>_ops.csv \
               --layer-kind <kind> --layers-of-kind N --total-layers M \
               --out    reconciliation_<kind>.json

Emits a partition of the measured window: every device op gets exactly one bucket, and the buckets sum
to 100 %:

  chain:<id>            in a maximal L1-resident run whose placement differs from shipped
  boundary              a conversion op -- what joining the chains either side would remove
  dram_resident         the advisor placed it in DRAM; that is advice too
  agrees_with_shipped   advised == shipped
  untraced              in the profile, absent from the advisor's graph

Candidates are ranked by CONVERSION VALUE -- the microseconds of boundary ops a change would remove --
not by the advised ops' window share. A chain-lengthening change removes conversions, and those are
separate device ops that an op-share threshold never counts.

Fails loudly: an unmapped op or an accounting that does not close is an error, not a silent gap.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Pairing is automatic by normalised name (see matches()). These two tables hold only what that cannot
# reach, and both are kept SMALL and GENERAL on purpose -- enumerating the op names of the models you
# happen to have seen is how this script goes stale.
#
# RENAMES: a ttnn op whose device class is a different word.
TTNN_TO_DEVICE = {
    "rms_norm": "LayerNorm",                 # rms_norm is served by the layernorm device op
    "linear": "Matmul",
    "concatenate_heads": "NLPConcatHeads",
    "paged_scaled_dot_product_attention_decode": "SdpaDecode",
    "scaled_dot_product_attention_decode": "SdpaDecode",
}
# NO DEVICE OP: tensor creation and pure metadata. Anything here consumes no device op.
NO_DEVICE_OP = {"reshape", "to_layout", "to_memory_config", "view",
                "full", "zeros", "ones", "empty", "arange"}
# CATCH-ALL DEVICE CLASSES: tt-metal funnels whole families through one device op, so an advised op that
# pairs with nothing may still be one of these. Pairing permissively is the safer error: a mispair puts an
# op in the wrong bucket, while a missed pair pushes a real device op into `untraced` and understates what
# the advisor saw -- and understating the advisor is the bias this stage exists to avoid.
CATCH_ALL = {"Unary", "BinaryNg", "Reduce", "Copy"}
CONVERSION = {"Reshard": "l1_regrid", "ShardedToInterleaved": "l1_to_dram",
              "InterleavedToSharded": "dram_to_l1", "Untilize": "retilize", "Tilize": "retilize"}


def device_class(op_code: str) -> str:
    return op_code.split()[0].replace("DeviceOperation", "")


def parse_perf(path: Path, incumbent_ms: float | None = None):
    """Device ops in program order, plus the window total.

    Column names differ between producers and cases: tt-perf-report writes `OP Code` / `Device Time` (us),
    a raw Tracy export writes `OP CODE` / `DEVICE KERNEL DURATION [ns]` / `CORE COUNT`. Look them up
    case-insensitively by substring rather than by exact spelling.
    """
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        sys.exit(f"FATAL: {path} has no rows")
    keys = {(k or "").strip().lower(): k for k in rows[0]}

    def find(*needles, exact=True):
        for n in needles:
            if exact and n in keys:
                return keys[n]
        for n in needles:
            for k, orig in keys.items():
                if n in k:
                    return orig
        return None

    code_col = find("op code", "op type", "name")
    dur_col = find("device time", "device kernel duration [ns]", "device kernel duration")
    core_col = find("cores", "core count")
    if code_col is None or dur_col is None:
        sys.exit(f"FATAL: {path}: need an op-code and a duration column; saw {sorted(keys)}")
    ns = "ns" in dur_col.lower() or "cycle" in dur_col.lower()
    scale = 1 / 1000.0 if ns else 1.0

    ops, total = [], 0.0
    for r in rows:
        code = (r.get(code_col) or "").strip()
        if not code:
            continue
        try:
            us = float(str(r[dur_col]).replace(",", "")) * scale
        except (TypeError, ValueError):
            us = 0.0
        cores = None
        if core_col:
            try:
                cores = int(float(str(r.get(core_col, "")).replace(",", "")))
            except (TypeError, ValueError):
                cores = None
        dsc = find("dram sharded", exact=False)
        ops.append({"id": r.get(find("id") or "", None), "cls": device_class(code), "code": code,
                    "us": us, "cores": cores,
                    "dram_sharded": (r.get(dsc) or "").strip().lower() in ("true", "1", "yes")
                                    if dsc else False})
        total += us

    # A report that was not bounded to one iteration by signposts repeats its op sequence. Detect that
    # structurally rather than by a magic ratio: shares computed over N iterations are N times too small.
    seq = [o["cls"] for o in ops]
    for period in range(1, len(seq) // 2 + 1):
        if len(seq) % period == 0 and seq == seq[:period] * (len(seq) // period):
            reps = len(seq) // period
            sys.exit(f"FATAL: {path}: the op sequence repeats {reps}x (period {period} of {len(seq)} "
                     f"rows), so this report covers {reps} iterations rather than one. Re-run "
                     f"tt-perf-report with --start-signpost/--end-signpost bounding a single replay; "
                     f"every share computed here would be {reps}x too small.")

    # And a window far from the harness's own number means the CSV is not the decode window at all.
    if incumbent_ms is not None and incumbent_ms > 0:
        ratio = total / (incumbent_ms * 1000.0)
        if ratio > 5 or ratio < 0.2:
            sys.exit(f"FATAL: {path}: window {total:.1f} us is {ratio:.1f}x the harness's "
                     f"{incumbent_ms * 1000:.1f} us. Either the harness does not measure the decode "
                     f"path, or the report is not signpost-bounded to one iteration. Shares computed "
                     f"against this window would be wrong.")
    return ops, total


def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", x.lower())


def matches(ttnn_op: str, dev_cls: str) -> bool:
    """Pair a ttnn op with a device class by normalised name, either direction as a prefix.

    Prefix-either-way covers the common shapes without an entry each: slice_static/Slice, concat/Concat,
    embedding/Embeddings, topk/TopK, repeat/Repeat, the rotary variants. TTNN_TO_DEVICE holds only genuine
    renames; CATCH_ALL covers families tt-metal funnels through one device op.
    """
    a, b = norm(ttnn_op), norm(dev_cls)
    return a == b or a.startswith(b) or b.startswith(a)


def parse_advised(report: dict):
    """Advised ops in program order. Geometry comes from report.json; the IR is not needed."""
    out = []
    for o in report.get("ops", []):
        lay = o.get("layout", "")
        m = re.search(r"cores=\((\d+),(\d+)\)-\((\d+),(\d+)\)", lay)
        cores = ((int(m.group(3)) - int(m.group(1)) + 1) * (int(m.group(4)) - int(m.group(2)) + 1)
                 if m else None)
        name = o["op"].replace("ttnn.", "")
        explicit = None if name in NO_DEVICE_OP else TTNN_TO_DEVICE.get(name, "__auto__")
        out.append({"index": o["index"], "op": name, "explicit": explicit, "layout": lay,
                    "cores": cores, "program_config": o.get("program_config", "") or "",
                    "space": "dram" if lay.startswith("dram") else
                             ("l1" if lay.startswith("l1") else None)})
    return out


def align(advised, device):
    """Pair the two program-order sequences, tolerating gaps on both sides.

    Lengths differ: the advised graph omits ops the tracer never saw and includes metadata-only ops with
    no device counterpart. A positional zip misattributes everything after the first gap. An advised op
    that pairs with nothing consumes no device op, so one unknown name cannot shift the rest.
    """
    pairs, di = [], 0
    for a in advised:
        if a["explicit"] is None:
            continue                                     # metadata only; nothing to account
        want = a["explicit"]
        j = di
        while j < len(device):
            c = device[j]["cls"]
            if (c == want) if want != "__auto__" else (matches(a["op"], c) or c in CATCH_ALL):
                break
            j += 1
        if j >= len(device):
            pairs.append((a, None))                      # advised op never ran, or name unknown
            continue
        for k in range(di, j):
            pairs.append((None, device[k]))              # device ops the advisor did not place
        pairs.append((a, device[j]))
        di = j + 1
    for k in range(di, len(device)):
        pairs.append((None, device[k]))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--perf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layer-kind", default="unknown")
    ap.add_argument("--layers-of-kind", type=int, default=1)
    ap.add_argument("--total-layers", type=int, default=1)
    ap.add_argument("--incumbent-ms", type=float, default=None,
                    help="incumbent_ms from incumbent.json; enables a window sanity check")
    ap.add_argument("--ir", type=Path, help="accepted and ignored; geometry comes from report.json")
    a = ap.parse_args()

    report = json.loads(a.report.read_text())
    advised = parse_advised(report)
    device, window = parse_perf(a.perf, a.incumbent_ms)
    if window <= 0:
        sys.exit(f"FATAL: measured window is 0 us from {a.perf}")

    rows, chains, cur = [], [], None
    for adv, dev in align(advised, device):
        if dev is None:
            auto = adv["explicit"] == "__auto__"
            rows.append({"op": adv["op"],
                         "bucket": "unmapped_advised" if auto else "advised_but_not_present",
                         "us": 0.0, "share_pct": 0.0, "advised": adv["layout"],
                         "reason": "no device op paired with this advised op. Either it did not run "
                                   "(a capture/shape mismatch worth a look), or its name does not pair "
                                   "with any device class -- add it to TTNN_TO_DEVICE if so."})
            continue
        r = {"op": adv["op"] if adv else None, "device": dev["code"], "cls": dev["cls"],
             "us": round(dev["us"], 3), "share_pct": round(100 * dev["us"] / window, 3),
             "shipped_cores": dev["cores"]}
        if dev["cls"] in CONVERSION:
            # conversions live in the advisor's reshards[] list, not ops[], so they arrive unpaired;
            # classify them first or they fall through to `untraced` and the ranking loses its input
            r.update(bucket="boundary", conversion_class=CONVERSION[dev["cls"]])
            cur = None                                   # a conversion ends the current L1 run
        elif adv is None:
            r.update(bucket="untraced", reason="in the profile, absent from the advisor's graph")
            cur = None
        elif adv["space"] == "dram":
            r.update(bucket="dram_resident", advised=adv["layout"],
                     reason="advisor placed it in DRAM -- that is advice, and it disagrees with a "
                            "sharded shipped op")
            cur = None
        else:
            same = (adv["cores"] is not None and adv["cores"] == dev["cores"]) or \
                   (dev["dram_sharded"] and "dram_sharded" in adv["program_config"])
            r.update(bucket="agrees_with_shipped" if same else "chain",
                     advised=adv["layout"], advised_cores=adv["cores"])
            if not same:
                if cur is None:
                    cur = {"chain": f"{a.layer_kind}:{len(chains)}", "ops": [], "us": 0.0,
                           "boundary_us": 0.0, "verdict": "pending", "measured_ms": None,
                           "repeats_ms": None}
                    chains.append(cur)
                cur["ops"].append(adv["op"])
                cur["us"] += dev["us"]
                r["chain"] = cur["chain"]
        rows.append(r)

    # conversion value: boundary us adjacent to a chain -- what joining across it would remove
    for i, r in enumerate(rows):
        if r["bucket"] != "boundary":
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(rows) and rows[j].get("chain"):
                ch = next(c for c in chains if c["chain"] == rows[j]["chain"])
                ch["boundary_us"] += r["us"] / 2.0
    for c in chains:
        c["us"] = round(c["us"], 3)
        c["boundary_us"] = round(c["boundary_us"], 3)
        c["conversion_value_us"] = c["boundary_us"]
        c["op_share_pct"] = round(100 * c["us"] / window, 3)
        c["per_model_us"] = round((c["us"] + c["boundary_us"]) * a.layers_of_kind, 3)
    chains.sort(key=lambda c: -(c["conversion_value_us"] + c["us"]))

    acct, accounted = {}, 0.0
    for r in rows:
        if r["bucket"] in ("advised_but_not_present", "unmapped_advised"):
            continue
        accounted += r["us"]
        b = acct.setdefault(r["bucket"], {"us": 0.0, "ops": 0})
        b["us"] += r["us"]
        b["ops"] += 1
    for v in acct.values():
        v["us"] = round(v["us"], 3)
        v["share_pct"] = round(100 * v["us"] / window, 3)
    closes = abs(accounted - window) < 0.01

    low = [r for r in rows if r["bucket"] in ("chain", "agrees_with_shipped")
           and (r.get("shipped_cores") or 99) <= 2 and r["share_pct"] >= 1.0]

    out = {
        "generated_by": "advisor-challenger/scripts/reconcile.py", "tool_version": 2,
        "layer_kind": a.layer_kind, "layers_of_kind": a.layers_of_kind,
        "total_layers": a.total_layers, "measured_window_us": round(window, 3),
        "accounting": acct, "accounting_closes_100pct": closes,
        "accounted_us": round(accounted, 3),
        "ranked_by": "conversion_value_us + chain op us -- NOT advised-op window share",
        "chains": chains, "material_ops_on_le_2_cores": low,
        "note": "Screen chains in the order given, each as one unit, and record repeats_ms. Do NOT copy "
                "advised core counts: they are selected under a bytes-shaped objective and are often "
                "smaller than the divisors of the tensor's tile count. The advisor's contribution is the "
                "op SET and the DIRECTION of the change; own the geometry.",
        "disagreements": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{a.layer_kind}: window {window:.3f} us, {len(rows)} rows, closes: {closes}")
    for b, v in sorted(acct.items(), key=lambda x: -x[1]["us"]):
        print(f"   {b:22s} {v['us']:9.3f} us {v['share_pct']:6.2f} %  ({v['ops']} ops)")
    print(f"   {len(chains)} chain(s), ranked by conversion value:")
    for c in chains:
        print(f"      {c['chain']:16s} ops {c['us']:8.3f} + boundaries {c['boundary_us']:7.3f} us"
              f"   {'/'.join(c['ops'])[:52]}")
    for r in low:
        print(f"   !! {r['device']} on {r['shipped_cores']} core(s), {r['us']:.3f} us "
              f"({r['share_pct']} %) -- needs a measured attempt or a quoted hard error")
    if not closes:
        sys.exit(f"FATAL: accounting does not close: {accounted:.3f} of {window:.3f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
