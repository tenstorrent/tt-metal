#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Turn one zone profile (parse_zone_perf.py --json [--per-device]) into per-op rows of per_op.csv.

  zones_to_per_op.py --zones zones.json [--per-device per_device.json] --W 4096 --h 141312 --B 1 \\
      --input prose --mesh 2x4 [--segments 4096:141312] [--run-id ID] [--out per_op.csv]

Ops come from a zone -> op map (--map, default tools/zone_to_op_map.json, else the shipped
zone_to_op_map.provisional.json). Accepted formats, per layer type ("sparse" / "dense"):
  {op: [zone, ...]}   the op is the sum of those zones (paths relative to the layer zone)
  {op: "__rest__"}    the layer total minus every other op of the layer ("misc" defaults to this)
  {zone: op}          the inverse (zone_to_op_map.json's format); zones mapped to "misc" are not summed, misc
                      is always the rest; "a+b" is a joint op (roofline a + b); " (note)" after a name is dropped
Zones nested in another mapped zone are skipped (the outer zone already contains them).
Across devices (parse_zone_perf's own convention): worst = max, mean, min over the chips. With --per-device the
per-op sums are formed per chip first, so worst/min are exact; without it worst/min are the sum of the zones'
own worst/min (an upper / lower bound) — mean is exact either way. calls = op count on each zone's worst chip.

roof_ms comes from `node roofline_ops.js --mesh <mesh> --segments <n:k,...> --layer both` (segments default
"W:h"); roof_ms / eff_* stay empty when node or the tool is missing. eff = roof / measured.
share_of_layer_worst = worst_ms / the layer's worst_ms. A "layer_total" row per layer carries the layer itself.
Rows are appended; the header is written when the file is new.
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLS = [
    "op",
    "layer",
    "layer_type",
    "W",
    "h",
    "B",
    "input",
    "calls",
    "worst_ms",
    "mean_ms",
    "min_ms",
    "roof_ms",
    "eff_worst",
    "eff_mean",
    "share_of_layer_worst",
    "mesh",
    "run_id",
]
REST = "__rest__"
ROOT = "profiled_chunk"
LAYER_RE = re.compile(r"^layer(\d+)_(sparse|dense)$")


def load_map(path):
    raw = json.load(open(path))
    raw = raw.get("zones", raw)
    out = {}
    for ltype in ("sparse", "dense"):
        m = raw.get(ltype) or raw.get({"sparse": "moe"}.get(ltype, ltype)) or {}
        m = {k: v for k, v in m.items() if not k.startswith("_")}
        if all(isinstance(v, list) or v == REST for v in m.values()):
            ops = {op: ([] if v == REST else list(v)) for op, v in m.items()}
            rest = [op for op, v in m.items() if v == REST]
        else:  # {zone: op}; an op label may carry a note: "ring (ring_c ... | ring_scan ...)" -> "ring"
            ops, rest = {}, []
            for z, op in m.items():
                op = op.split(" (")[0].strip()
                if op in (REST, "misc"):  # misc is never summed from zones: it is the rest of the layer
                    if "misc" not in rest:
                        rest.append("misc")
                    continue
                ops.setdefault(op, []).append(z)
        if "misc" not in ops and not rest:
            rest = ["misc"]
        for op in rest:
            ops[op] = []
        out[ltype] = (ops, rest)
    return out


def roofline(mesh, segments):
    tool = HERE / "roofline_ops.js"
    node = shutil.which("node")
    if not node or not tool.is_file():
        print(f"[per_op] no roofline ({'node missing' if not node else f'{tool} missing'}): roof/eff left empty")
        return {}
    cmd = [node, str(tool), "--mesh", mesh, "--segments", segments, "--layer", "both"]
    try:
        res = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError) as e:
        print(f"[per_op] roofline failed ({' '.join(cmd)}): {e}; roof/eff left empty")
        return {}
    return {k: v for k, v in res.items() if k in ("sparse", "dense")}


def layer_zones(zones):
    """{(idx, ltype): {rel_path: zone_stats}}, rel_path '' = the layer zone itself."""
    layers = {}
    for path, st in zones.items():
        parts = path.split("/")
        if len(parts) < 2 or parts[0] != ROOT:
            continue
        m = LAYER_RE.match(parts[1])
        if m:
            layers.setdefault((int(m.group(1)), m.group(2)), {})["/".join(parts[2:])] = (path, st)
    return layers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zones", required=True, help="parse_zone_perf.py --json output")
    ap.add_argument("--per-device", help="parse_zone_perf.py --per-device output (exact worst/min)")
    ap.add_argument("--W", required=True, type=int)
    ap.add_argument("--h", required=True, help="history depth; for a packed forward e.g. '141312|0'")
    ap.add_argument("--B", type=int, default=1, help="segments in the forward (1 = single request)")
    ap.add_argument("--input", required=True, help="prose / code / mixed label, e.g. 'prose|code'")
    ap.add_argument("--mesh", default="2x4", help="SPxTP of the profiled stage")
    ap.add_argument("--segments", help="roofline segments n:k[,n:k...] (default W:h)")
    ap.add_argument("--map", help="zone -> op map JSON (default tools/zone_to_op_map.json or the provisional one)")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--out", default=str(HERE.parent / "per_op.csv"))
    args = ap.parse_args()

    map_path = Path(args.map) if args.map else HERE / "zone_to_op_map.json"
    if not map_path.is_file():
        map_path = HERE / "zone_to_op_map.provisional.json"
        print(f"[per_op] using the provisional map {map_path}")
    opmap = load_map(map_path)
    zones = json.load(open(args.zones))["zones"]
    per_dev = json.load(open(args.per_device)) if args.per_device else None
    segments = args.segments or f"{args.W}:{args.h}"
    roof = roofline(args.mesh, segments)

    def stats(paths):
        """(worst, mean, min, calls) of the sum of these zone paths."""
        found = [(p, st) for p, st in paths if st is not None]
        if not found:
            return None
        calls = sum(int(st["ops"]) for _, st in found)
        mean = sum(st["ms_mean"] for _, st in found)
        if per_dev is not None and all(p in per_dev for p, _ in found):
            devs = set().union(*(per_dev[p].keys() for p, _ in found))
            tot = [sum(per_dev[p].get(d, 0.0) for p, _ in found) for d in devs]
            return max(tot), mean, min(tot), calls
        return sum(st["ms_max"] for _, st in found), mean, sum(st["ms_min"] for _, st in found), calls

    rows = []
    for (idx, ltype), rels in sorted(layer_zones(zones).items()):
        ops, rest = opmap[ltype]
        layer_path, layer_st = rels[""]
        lw, lm, ln, lc = stats([(layer_path, layer_st)])
        # A zone nested in another mapped zone is already inside that zone's time (e.g. mlp/shared_expert/
        # tp_reduce_scatter in mlp/shared_expert, mlp/gate_up_proj in mlp): count only the outermost.
        present = {z for op, zl in ops.items() if op not in rest for z in zl if z in rels}
        outer = {z for z in present if not any(z.startswith(a + "/") for a in present)}
        measured = {}
        for op, zl in ops.items():
            if op in rest:
                continue
            zl = [z for z in zl if z in outer]
            s = stats([rels[z] for z in zl])
            if s is not None:
                measured[op] = (s, [rels[z][0] for z in zl])
        listed = [p for _, ps in measured.values() for p in ps]
        for op in rest:
            # layer minus the listed ops, per chip when the per-device data is there
            if per_dev is not None and layer_path in per_dev and all(p in per_dev for p in listed):
                tot = [v - sum(per_dev[p].get(d, 0.0) for p in listed) for d, v in per_dev[layer_path].items()]
                w, n = max(tot), min(tot)
            else:
                w = lw - sum(s[0] for s, _ in measured.values())
                n = ln - sum(s[2] for s, _ in measured.values())
            m = lm - sum(s[1] for s, _ in measured.values())
            c = lc - sum(s[3] for s, _ in measured.values())
            measured[op] = ((w, m, n, c), [])
        measured["layer_total"] = ((lw, lm, ln, lc), [layer_path])
        for op, ((w, m, n, c), _) in measured.items():
            lroof = roof.get(ltype, {})
            names = {o for o in ops for o in o.split("+")} if op == "layer_total" else set(op.split("+"))
            r = sum(lroof[o] for o in names if o in lroof) if any(o in lroof for o in names) else None
            rows.append(
                {
                    "op": op,
                    "layer": idx,
                    "layer_type": ltype,
                    "W": args.W,
                    "h": args.h,
                    "B": args.B,
                    "input": args.input,
                    "calls": c,
                    "worst_ms": round(w, 4),
                    "mean_ms": round(m, 4),
                    "min_ms": round(n, 4),
                    "roof_ms": "" if r is None else round(r, 4),
                    "eff_worst": "" if r is None or w <= 0 else round(r / w, 4),
                    "eff_mean": "" if r is None or m <= 0 else round(r / m, 4),
                    "share_of_layer_worst": round(w / lw, 4) if lw > 0 else "",
                    "mesh": args.mesh,
                    "run_id": args.run_id,
                }
            )
    if not rows:
        sys.exit(f"[per_op] no layer zones under {ROOT}/ in {args.zones}")
    out = Path(args.out)
    new = not out.is_file() or out.stat().st_size == 0
    if not new:
        with open(out) as f:
            head = f.readline().strip().split(",")
        assert head == COLS, f"{out} has columns {head}, expected {COLS}"
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"[per_op] {len(rows)} rows ({len({(r['layer']) for r in rows})} layers) -> {out}")


if __name__ == "__main__":
    main()
