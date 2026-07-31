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

# ttnn op -> device op class. Extend rather than guess: an unmapped op aborts, because a wrong guess
# misattributes every later op in the alignment.
TTNN_TO_DEVICE = {
    "rms_norm": "LayerNorm", "layer_norm": "LayerNorm",
    "linear": "Matmul", "matmul": "Matmul", "sparse_matmul": "SparseMatmul",
    "add": "BinaryNg", "multiply": "BinaryNg", "subtract": "BinaryNg", "div": "BinaryNg",
    "nlp_create_qkv_heads_decode": "NLPCreateQKVHeadsDecode",
    "nlp_concat_heads_decode": "NLPConcatHeadsDecode",
    "concatenate_heads": "NLPConcatHeads",
    "rotary_embedding_llama": "RotaryEmbeddingLlama", "rotary_embedding": "RotaryEmbedding",
    "paged_scaled_dot_product_attention_decode": "SdpaDecode",
    "scaled_dot_product_attention_decode": "SdpaDecode",
    "paged_update_cache": "PagedUpdateCache", "update_cache": "UpdateCache",
    "transpose": "Transpose", "embedding": "Embedding", "silu": "Silu", "gelu": "Gelu",
    "softmax": "Softmax", "typecast": "Typecast",
    # metadata / conversion only -- no device op of their own
    "reshape": None, "to_layout": None, "to_memory_config": None, "view": None,
}
CONVERSION = {"Reshard": "l1_regrid", "ShardedToInterleaved": "l1_to_dram",
              "InterleavedToSharded": "dram_to_l1", "Untilize": "retilize", "Tilize": "retilize"}


def device_class(op_code: str) -> str:
    return op_code.split()[0].replace("DeviceOperation", "")


def parse_perf(path: Path):
    """Device ops in program order, plus the window total. Accepts us or ns duration columns."""
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        sys.exit(f"FATAL: {path} has no rows")
    col = next((c for c in rows[0] if c and c.strip().lower() in
                ("device time", "device kernel duration [ns]", "device kernel duration",
                 "device_kernel_duration_ns")), None)
    if col is None:
        sys.exit(f"FATAL: {path} has no recognised duration column; saw {list(rows[0])}")
    scale = 1 / 1000.0 if "ns" in col.lower() else 1.0
    ops, total = [], 0.0
    for r in rows:
        code = (r.get("OP Code") or "").strip()
        if not code:
            continue
        try:
            us = float(str(r[col]).replace(",", "")) * scale
        except (TypeError, ValueError):
            us = 0.0
        try:
            cores = int(float(str(r.get("Cores", "")).replace(",", "")))
        except (TypeError, ValueError):
            cores = None
        ops.append({"id": r.get("ID"), "cls": device_class(code), "code": code,
                    "us": us, "cores": cores,
                    "dram_sharded": (r.get("DRAM Sharded") or "").strip().lower()
                                    in ("true", "1", "yes")})
        total += us
    return ops, total


def parse_advised(report: dict):
    """Advised ops in program order. Geometry comes from report.json; the IR is not needed."""
    out = []
    for o in report.get("ops", []):
        lay = o.get("layout", "")
        m = re.search(r"cores=\((\d+),(\d+)\)-\((\d+),(\d+)\)", lay)
        cores = ((int(m.group(3)) - int(m.group(1)) + 1) * (int(m.group(4)) - int(m.group(2)) + 1)
                 if m else None)
        name = o["op"].replace("ttnn.", "")
        if name not in TTNN_TO_DEVICE:
            sys.exit(f"FATAL: ttnn op {name!r} has no device-class mapping. Add it to TTNN_TO_DEVICE; "
                     f"guessing would misattribute every later op.")
        out.append({"index": o["index"], "op": name, "cls": TTNN_TO_DEVICE[name], "layout": lay,
                    "cores": cores, "program_config": o.get("program_config", "") or "",
                    "space": "dram" if lay.startswith("dram") else
                             ("l1" if lay.startswith("l1") else None)})
    return out


def align(advised, device):
    """Pair the two program-order sequences on device class, tolerating gaps on both sides.

    Lengths differ: the advised graph omits ops the tracer never saw and includes metadata-only ops with
    no device counterpart. A positional zip misattributes everything after the first gap.
    """
    pairs, di = [], 0
    for a in advised:
        if a["cls"] is None:
            continue                                     # metadata only; nothing to account
        j = di
        while j < len(device) and device[j]["cls"] != a["cls"]:
            j += 1
        if j >= len(device):
            pairs.append((a, None))                      # advised op never ran
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
    ap.add_argument("--ir", type=Path, help="accepted and ignored; geometry comes from report.json")
    a = ap.parse_args()

    report = json.loads(a.report.read_text())
    advised = parse_advised(report)
    device, window = parse_perf(a.perf)
    if window <= 0:
        sys.exit(f"FATAL: measured window is 0 us from {a.perf}")

    rows, chains, cur = [], [], None
    for adv, dev in align(advised, device):
        if dev is None:
            rows.append({"op": adv["op"], "bucket": "advised_but_not_present", "us": 0.0,
                         "share_pct": 0.0, "advised": adv["layout"],
                         "reason": "advised op absent from the measured window -- a capture/shape "
                                   "mismatch worth a look, not a free win"})
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
        if r["bucket"] == "advised_but_not_present":
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
