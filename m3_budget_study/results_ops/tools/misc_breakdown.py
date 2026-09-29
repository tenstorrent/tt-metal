#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Break Pavlo's `misc` bucket into ops, from the P0-A zone-profile ops CSVs (host only).

  misc_breakdown.py [--runs p0a_runs.csv] [--out misc_breakdown.csv] [--summary misc_breakdown.txt]
                    [--only RUN_ID ...] [--map tools/zone_to_op_map.json]

misc = every device op of a layer zone that is not inside a zone mapped to a named op (zone_to_op_map.json;
the same rule zones_to_per_op.py uses: misc = layer total - named zones). Each such op gets a role from its
op code, its zone, the last named zone before it, and its input shape (rope_q_slice, kv_write, residual_attn,
routing_setup_*, seg_slice_*, ...). Unknown ops keep '<zone>:<op>' as their role.

Per (run, layer type, role, op code): calls per layer, device ms per layer (per-chip sum over the layers of
that type / #layers; worst = max over chips, mean = mean over chips), share of that run's misc (mean).

Tag: per (W, layer type, role, op) the calls per layer of the packed forward (B segments) over the single
forward's at the same W (reference: --ref-h / --ref-input, default the h=139264 prose run): ratio ~B -> per-segment, ~1 -> whole-tensor,
0 single calls -> packed-only (segment split/merge glue). Packed rows carry their own tag (B = 2048-token
segments of the forward), single rows the tags of every packed run at their W.

Runs come from p0a_runs.csv (status OK; the last OK row per run id) or --csv RUN_ID=path pairs.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent
ROOT = "profiled_chunk"
LAYER_RE = re.compile(r"^layer(\d+)_(sparse|dense)$")
DUR = "DEVICE KERNEL DURATION [ns]"
SP = 2  # rows per chip = W / SP on the (2,4) stage

GROUPS = [  # role prefix -> coarse group (the batching work items)
    ("rope_", "rope"),
    ("q_norm", "qk_norm"),
    ("k_norm", "qk_norm"),
    ("kv_typecast", "kv_write"),
    ("kv_write", "kv_write"),
    ("index_k_", "kv_write"),
    ("input_norm", "norm"),
    ("post_attn_norm", "norm"),
    ("residual_", "residual"),
    ("add_shared", "residual"),
    ("split_heads", "heads"),
    ("concat_heads", "heads"),
    ("routing_setup", "routing_setup"),
    ("tp_allgather", "tp_allgather"),
    ("seg_", "segment_glue"),
]


def short(code):
    return re.sub(r"(Device)?Operation$", "", code)


def dim(row, t, d):
    v = row.get(f"INPUT_{t}_{d}_PAD[LOGICAL]") or ""
    try:
        return int(float(v.split("[")[0]))
    except ValueError:
        return None


def load_named(map_path):
    raw = json.load(open(map_path))
    raw = raw.get("zones", raw)
    named = {}
    for lt in ("sparse", "dense"):
        m = raw.get(lt, {})
        named[lt] = {z: op.split(" (")[0] for z, op in m.items() if not z.startswith(("_", "(")) and op != "misc"}
    return named


def named_op(named, ltype, rel):
    """The named op this zone path belongs to (innermost mapped prefix), or None (misc)."""
    parts = rel.split("/") if rel else []
    for i in range(len(parts), 0, -1):
        z = "/".join(parts[:i])
        if z in named[ltype]:
            return named[ltype][z]
    return None


def role_of(ltype, rel, code, last_named, row, W):
    s = short(code)
    heads, rows, width = dim(row, 0, "Z"), dim(row, 0, "Y"), dim(row, 0, "X")
    qk = "q" if (heads or 1) > 1 else "k"
    whole = rows is not None and rows >= W // SP
    if rel == "":
        if s == "LayerNorm":
            return {"input_norm_allgather": "input_norm", "post_attn_norm_allgather": "post_attn_norm"}.get(
                last_named, "layer_norm"
            )
        if s == "BinaryNg":
            return "residual_mlp" if last_named.startswith("mlp") else "residual_attn"
        return f"layer:{s}"
    if rel == "attn":
        idx = last_named == "attn/index_branch" or last_named.startswith("attn/index")
        if s == "Slice" and whole and heads == 1 and width in (2304, 6144):
            return f"seg_slice_{'qkv' if width == 2304 else 'x'}"
        if s == "Concat" and whole is False and heads == 1 and width and width >= 1024:
            return "seg_concat_out"
        if s == "NlpCreateHeads":
            return "split_heads"
        if s == "NLPConcatHeads":
            return "concat_heads"
        if s == "LayerNorm":
            return f"{qk}_norm"
        if s == "Slice":
            return f"rope_{qk}_slice"
        if s.startswith("RotaryEmbedding"):
            return f"rope_{qk}_rotary"
        if s == "Concat":
            return f"rope_{qk}_concat"
        if s == "Typecast":
            return "index_k_typecast" if idx else "kv_typecast"
        if s.startswith("UpdatePaddedKvCache") or "FillCache" in s or "UpdateCache" in s:
            return "index_k_write" if idx else "kv_write"
        return f"attn:{s}"
    if rel == "mlp":
        if s in ("MaskedBincount", "OffsetCumsum", "UntilizeWithUnpadding") or (
            s == "AllGather" and str(row.get("INPUT_0_DATATYPE", "")).startswith("UINT")
        ):
            return f"routing_setup_{s}"
        if s.startswith("AllGather"):
            return "tp_allgather"
        if s == "BinaryNg":
            return "add_shared"
        return f"mlp:{s}"
    return f"{rel}:{s}"


def group_of(role):
    for p, g in GROUPS:
        if role.startswith(p):
            return g
    return "other"


def scan(csv_path, named, W):
    """{(ltype, zone, role, code): {dev: [ns, calls]}}, {(ltype): set(layers)}, misc/layer totals per chip."""
    stack, last_named = [], {}
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    layers = defaultdict(set)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            code, typ = row.get("OP CODE", ""), row.get("OP TYPE", "")
            if typ == "signpost":
                if code.startswith("M3_ZONE_START"):
                    stack.append(code[len("M3_ZONE_START") :].strip())
                elif code.startswith("M3_ZONE_END"):
                    end = code[len("M3_ZONE_END") :].strip()
                    if end in stack:
                        while stack and stack.pop() != end:
                            pass
                    if len(stack) >= 2 and stack[0] == ROOT and LAYER_RE.match(stack[1]):
                        m = LAYER_RE.match(stack[1])
                        rel = "/".join(stack[2:] + [end])
                        if named_op(named, m.group(2), rel):
                            last_named[stack[1]] = rel
                    elif len(stack) == 1 and stack[0] == ROOT and LAYER_RE.match(end):
                        last_named.pop(end, None)
                continue
            if len(stack) < 2 or stack[0] != ROOT:
                continue
            m = LAYER_RE.match(stack[1])
            if not m:
                continue
            try:
                ns, dev = float(row[DUR]), int(row["DEVICE ID"])
            except (TypeError, ValueError, KeyError):
                continue
            ltype = m.group(2)
            layers[ltype].add(int(m.group(1)))
            rel = "/".join(stack[2:])
            if named_op(named, ltype, rel):
                continue
            role = role_of(ltype, rel, code, last_named.get(stack[1], ""), row, W)
            e = acc[(ltype, rel or "(layer)", role, short(code))][dev]
            e[0] += ns
            e[1] += 1
    return acc, layers


def runs_from_table(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["status"].startswith("OK") and r.get("csv"):
            out[r["run_id"]] = r
    return out


def per_request_section(rows, single):
    """Packed runs with the same W and B but different slot counts (e.g. 1 slot x 2 segments vs 2 x 1): an op's
    packed/single call ratio is ~B when it runs per 2048-token segment, ~slots when it runs per request (slot),
    ~1 when it runs once on the whole forward."""
    key = lambda x: (x["W"], x["layer_type"], x["role"], x["op"])
    runs = {}
    for x in rows:
        if x["forward"] == "packed":
            runs[x["run_id"]] = (x["W"], x["B"], x["slots"])
    groups = defaultdict(list)
    for rid, (W, B, sl) in runs.items():
        groups[(W, B)].append((sl, rid))
    out = []
    for (W, B), rs in sorted(groups.items()):
        if len({sl for sl, _ in rs}) < 2:
            continue
        rs.sort()
        out += [
            "",
            f"Per-request vs per-segment, W={W} B={B} segments: " + ", ".join(f"{rid} ({sl} slots)" for sl, rid in rs),
        ]
        calls = defaultdict(dict)
        ms = defaultdict(dict)
        for x in rows:
            if x["run_id"] in {r for _, r in rs}:
                calls[key(x)][x["run_id"]] = x["calls_per_layer"]
                ms[key(x)][x["run_id"]] = x["mean_ms_per_layer"]
        for k in sorted(calls, key=lambda k: (k[1], -max(ms[k].values()))):
            s = single.get(k, 0.0)
            ratios = {rid: (calls[k].get(rid, 0.0) / s if s else None) for _, rid in rs}
            if all(r is None for r in ratios.values()):
                verdict = "packed-only"
            else:
                fits = []
                for h in ("per-segment", "per-request", "whole-tensor"):
                    good = True
                    for sl, rid in rs:
                        r, want = ratios[rid], (B if h == "per-segment" else sl if h == "per-request" else 1)
                        good &= r is not None and abs(r - want) <= 0.25 * want
                    if good:
                        fits.append(h)
                verdict = fits[0] if len(fits) == 1 else "/".join(fits) if fits else "other"
            out.append(
                f"  {k[1]:6} {k[2]:28} {k[3]:24} {verdict:13} "
                + "  ".join(
                    (
                        f"{rid}: x{ratios[rid]:.2f} {ms[k].get(rid, 0.0):.3f} ms"
                        if ratios[rid] is not None
                        else f"{rid}: {calls[k].get(rid, 0.0):.0f} calls {ms[k].get(rid, 0.0):.3f} ms"
                    )
                    for _, rid in rs
                )
            )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(RES / "p0a_runs.csv"), help="run table(s), comma separated")
    ap.add_argument("--only", nargs="*", help="run ids to include (default all OK runs)")
    ap.add_argument("--map", default=str(HERE / "zone_to_op_map.json"))
    ap.add_argument("--ref-h", default="139264", help="history of the single run used for the tag ratio")
    ap.add_argument("--ref-input", default="prose", help="input of the single run used for the tag ratio")
    ap.add_argument("--out", default=str(RES / "misc_breakdown.csv"))
    ap.add_argument("--summary", default=str(RES / "misc_breakdown.txt"))
    args = ap.parse_args()

    named = load_named(args.map)
    runs = {}
    for p in args.runs.split(","):
        if Path(p).is_file():
            runs.update(runs_from_table(p))
    if args.only:
        runs = {k: v for k, v in runs.items() if k in args.only}
    if not runs:
        sys.exit("[misc] no OK runs with an ops CSV")

    rows = []
    for rid, r in sorted(runs.items()):
        W, B = int(r["W"]), int(r["B"])
        slots = len([x for x in r.get("segments", "").split(",") if x]) or 1
        csv_path = r["csv"]
        if not Path(csv_path).is_file():
            print(f"[misc] {rid}: {csv_path} missing, skipped")
            continue
        acc, layers = scan(csv_path, named, W)
        n_layers = {lt: len(v) for lt, v in layers.items()}
        misc_mean = defaultdict(float)
        for (lt, zone, role, code), devs in acc.items():
            per_dev = [v[0] / 1e6 / n_layers[lt] for v in devs.values()]
            calls = max(v[1] for v in devs.values()) / n_layers[lt]
            mean = statistics.mean(per_dev)
            misc_mean[lt] += mean
            rows.append(
                dict(
                    run_id=rid,
                    W=W,
                    h=r["h"],
                    B=B,
                    slots=slots,
                    input=r["input"],
                    forward="packed" if B > 1 else "single",
                    layer_type=lt,
                    n_layers=n_layers[lt],
                    zone=zone,
                    role=role,
                    group=group_of(role),
                    op=code,
                    calls_per_layer=round(calls, 3),
                    worst_ms_per_layer=round(max(per_dev), 4),
                    mean_ms_per_layer=round(mean, 4),
                )
            )
        for x in rows:
            if x["run_id"] == rid:
                x["share_of_misc_mean"] = round(x["mean_ms_per_layer"] / misc_mean[x["layer_type"]], 4)
        print(f"[misc] {rid}: misc mean ms/layer " + " ".join(f"{k}={v:.3f}" for k, v in sorted(misc_mean.items())))

    # per-segment / whole-tensor tag, per (W, B): packed calls vs the reference single run's at that W
    key = lambda x: (x["W"], x["layer_type"], x["role"], x["op"])
    single = {
        key(x): x["calls_per_layer"]
        for x in rows
        if x["forward"] == "single" and str(x["h"]) == args.ref_h and x["input"] == args.ref_input
    }
    packed = defaultdict(dict)  # key -> {run_id: (calls, B = 2048-token segments)}
    for x in rows:
        if x["forward"] == "packed":
            packed[key(x)][x["run_id"]] = (x["calls_per_layer"], x["B"])
    tag = {}  # (key, run_id) -> (tag, ratio)
    for k, byrun in packed.items():
        s = single.get(k, 0.0)
        for rid, (p, b) in byrun.items():
            if s == 0:
                tag[(k, rid)] = ("packed-only", "")
            else:
                ratio = p / s
                t = "per-segment" if ratio >= 0.75 * b else "whole-tensor" if ratio <= 1.25 else "mixed"
                tag[(k, rid)] = (t, round(ratio, 2))
    for x in rows:
        k = key(x)
        if x["forward"] == "packed":
            x["seg_tag"], x["packed_over_single_calls"] = tag.get((k, x["run_id"]), ("", ""))
        else:  # a single row carries the tags of every packed run at its W
            ts = sorted((rid, tag[(k, rid)]) for rid in packed.get(k, {}))
            x["seg_tag"] = ";".join(f"{rid}:{t}" for rid, (t, _) in ts)
            x["packed_over_single_calls"] = ";".join(f"{rid}:{r}" for rid, (_, r) in ts)

    cols = [
        "run_id", "forward", "W", "h", "B", "slots", "input", "layer_type", "n_layers", "zone", "role", "group", "op",
        "calls_per_layer", "worst_ms_per_layer", "mean_ms_per_layer", "share_of_misc_mean", "seg_tag",
        "packed_over_single_calls",
    ]  # fmt: skip
    rows.sort(key=lambda x: (x["run_id"], x["layer_type"], -x["mean_ms_per_layer"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[misc] {len(rows)} rows -> {args.out}")

    # summary: per run of interest, groups per layer type
    lines = [
        "misc breakdown (device ms per layer, mean over the 8 chips; worst chip in brackets). Source: P0-A zone",
        "profiles, (2,4) SP=2 TP=4, layers 0-6 (dense 0-2, sparse 3-6). Groups: "
        + ", ".join(sorted({g for _, g in GROUPS})),
        "",
    ]
    by_run = defaultdict(list)
    for x in rows:
        by_run[x["run_id"]].append(x)
    for rid in sorted(by_run, key=lambda r: (by_run[r][0]["W"], by_run[r][0]["B"], r)):
        rr = by_run[rid]
        x0 = rr[0]
        lines.append(f"## {rid}  W={x0['W']} B={x0['B']} h={x0['h']} input={x0['input']}")
        for lt in ("sparse", "dense"):
            lr = [x for x in rr if x["layer_type"] == lt]
            if not lr:
                continue
            tot = sum(x["mean_ms_per_layer"] for x in lr)
            g = defaultdict(lambda: [0.0, 0.0, 0.0])
            for x in lr:
                g[x["group"]][0] += x["mean_ms_per_layer"]
                g[x["group"]][1] += x["worst_ms_per_layer"]
                g[x["group"]][2] += x["calls_per_layer"]
            parts = [
                f"{k} {v[0]:.3f} [{v[1]:.3f}] ({v[0] / tot:.0%}, {v[2]:.0f} ops)"
                for k, v in sorted(g.items(), key=lambda kv: -kv[1][0])
            ]
            lines.append(f"  {lt:6} misc {tot:.3f} ms/layer: " + "; ".join(parts))
        lines.append("")
    lines.append("Per-segment vs whole-tensor (packed calls / single calls per layer at the same W; run: tag ratio):")
    seen = defaultdict(dict)
    for ((W, lt, role, op), rid), (t, ratio) in sorted(tag.items()):
        seen[(lt, role, op)][rid] = f"{t} {ratio}".strip()
    for (lt, role, op), d in sorted(seen.items()):
        lines.append(f"  {lt:6} {role:28} {op:24} " + "  ".join(f"{rid}: {v}" for rid, v in sorted(d.items())))
    lines += per_request_section(rows, single)
    Path(args.summary).write_text("\n".join(lines) + "\n")
    print(f"[misc] summary -> {args.summary}")


if __name__ == "__main__":
    main()
