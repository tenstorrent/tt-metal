# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer breakdown of one whole-model decode step from a tracy ops report (test_decode_profile.py).

A decode step runs from an Embeddings op to the next one. It is split into decoder layers at each MLP
down-projection (the only matmul with K = 17408). Each layer splits at its first gate/up matmul into the mixer
(input norm, GDN or attention, residual add, post-attention norm) and the MLP (through the residual add after
down). A mixer counts as attention if it holds an SDPA or KV-cache op.
    python decode_profile_summary.py OPS_PERF_CSV [--cpp-report .logs/cpp_device_perf_report.csv]
The cpp report holds every op's device time but no shapes; it gives each step's total as a check.
"""
import argparse
import collections
import csv
import re

ATTN_OPS = re.compile(r"ScaledDotProduct|Sdpa|UpdateCache|PagedFill|FillCache", re.I)
NORM_OPS = re.compile(r"LayerNorm|RMSNorm", re.I)
FFN = 17408


def kernel_ns(r):
    v = (r.get("DEVICE KERNEL DURATION [ns]") or "").strip()
    return float(v) if v not in ("", "-") else 0.0


def logical(r, t, d):
    v = (r.get(f"{t}_{d}_PAD[LOGICAL]") or "").split("[")[-1].rstrip("]")
    return int(v) if v.isdigit() else -1


def is_matmul(r):
    return "Matmul" in r["OP CODE"]


def steps_of(rows):
    starts = [i for i, r in enumerate(rows) if "Embedding" in r["OP CODE"]]
    return [rows[a:b] for a, b in zip(starts, starts[1:] + [len(rows)])]


def split_layers(step):
    """Returns head ops, a list of (kind, mixer ops, mlp ops) per layer, and tail ops."""
    downs = [i for i, r in enumerate(step) if is_matmul(r) and logical(r, "INPUT_1", "Y") == FFN]
    start = next((i for i, r in enumerate(step) if NORM_OPS.search(r["OP CODE"])), 0)
    head, layers = step[:start], []
    for d in downs:
        layer = step[start : d + 2]
        mlp_at = next(i for i, r in enumerate(layer) if is_matmul(r) and logical(r, "INPUT_1", "X") >= FFN)
        mixer = layer[:mlp_at]
        kind = "attention" if any(ATTN_OPS.search(r["OP CODE"]) for r in mixer) else "gdn"
        layers.append((kind, mixer, layer[mlp_at:]))
        start = d + 2
    return head, layers, step[start:]


def report(name, ops, n_units, top):
    mm = sum(kernel_ns(r) for r in ops if is_matmul(r)) / 1e3
    other = sum(kernel_ns(r) for r in ops) / 1e3 - mm
    per = f"{mm / n_units:.0f} + {other / n_units:.0f}" if n_units > 1 else ""
    print(f"  {name:<12}{len(ops):>6}{mm / 1e3:>10.2f}{other / 1e3:>10.2f}{(mm + other) / 1e3:>10.2f}{per:>22}")
    by_code = collections.Counter()
    for r in ops:
        if not is_matmul(r):
            by_code[r["OP CODE"].replace("DeviceOperation", "")] += kernel_ns(r)
    print("               other: " + ", ".join(f"{c} {ns / 1e6:.2f}" for c, ns in by_code.most_common(top)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--cpp-report")
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv)) if r["OP TYPE"] != "signpost"]
    steps = steps_of(rows)
    if not steps:
        raise SystemExit("no Embeddings op found")
    step = steps[0]
    note = "" if len(steps) > 1 else " (no later step start, so the tail may be cut off)"
    total = sum(kernel_ns(r) for r in step) / 1e6
    print(f"{a.csv}: {len(rows)} ops, {len(steps)} step start(s); first step {len(step)} ops, {total:.2f} ms{note}")
    head, layers, tail = split_layers(step)
    kinds = collections.Counter(k for k, _, _ in layers)
    print(f"  layers: {dict(kinds)}")
    print(f"  {'part':<12}{'ops':>6}{'mm ms':>10}{'other ms':>10}{'total ms':>10}{'per layer us (mm+oth)':>22}")
    report("head", head, 1, a.top)
    for kind in ("gdn", "attention"):
        report(kind, [r for k, mixer, _ in layers if k == kind for r in mixer], kinds[kind], a.top)
    report("mlp", [r for _, _, mlp in layers for r in mlp], len(layers), a.top)
    report("tail", tail, 1, a.top)
    if a.cpp_report:
        cpp = list(csv.DictReader(open(a.cpp_report)))
        for r in cpp:
            r["OP CODE"] = r["OP NAME"]
        totals = [(len(s), sum(kernel_ns(r) for r in s) / 1e6) for s in steps_of(cpp)]
        print("cpp report steps (ops, device kernel ms): " + ", ".join(f"({n}, {ms:.2f})" for n, ms in totals))


if __name__ == "__main__":
    main()
