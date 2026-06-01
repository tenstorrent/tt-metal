# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Module-wise execution-time breakdown of a ttnn ``ops_perf_results_*.csv`` profile.

The Tracy/ttnn op profile is a *flat* list of device operations with no notion of
which Python module (attention, MoE, norm, ...) issued them, so this tool attributes
each op back to a ``DeepSeekV4DecoderLayer`` sub-module
(``models/experimental/deepseek_v4_flash/tt/deepseek_v4_flash.py``) using an ordered
set of heuristic rules over the op code + tensor shapes.

Because several modules reuse the same primitive (e.g. ``MatmulDeviceOperation`` is
issued by attention, the router gate, the shared expert and the hyper-connection),
attribution relies on shape signatures (hidden_size=4096, kv=1024, n_experts=256,
hc width=16384, ...). The rules are intentionally explicit and editable at the top of
this file -- tune ``MODULE_RULES`` if your config differs. Anything that matches no
rule lands in the ``unclassified`` bucket and is reported separately so the breakdown
stays honest.

Usage:
    python analyze_profile.py path/to/ops_perf_results_*.csv
    python analyze_profile.py path/to/ops_perf_results_*.csv --metric device --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# Model-derived constants (DeepSeek-V4-Flash). Adjust if your config changes.
# --------------------------------------------------------------------------- #
HIDDEN = 4096  # config.hidden_size
KV = 1024  # kv_proj output width (self_attn.kv_proj)
N_EXPERTS = 256  # config.n_routed_experts (router gate output width)
HC_WIDTH = 16384  # hc_mult * hidden_size (hyper-connection stacked streams)


# --------------------------------------------------------------------------- #
# Shape helpers
# --------------------------------------------------------------------------- #
def _int(v: str) -> Optional[int]:
    """Parse a ``32[4]`` / ``4096[4096]`` padded-shape cell -> padded int (or None)."""
    if not v:
        return None
    s = v.split("[", 1)[0].strip()
    try:
        return int(s)
    except ValueError:
        return None


@dataclass
class Op:
    code: str
    device_ns: float
    host_ns: float
    op_to_op_ns: float
    in0: tuple[Optional[int], Optional[int]]
    in1: tuple[Optional[int], Optional[int]]
    out0: tuple[Optional[int], Optional[int]]
    attributes: str

    @property
    def in0_x(self) -> Optional[int]:
        return self.in0[1]

    @property
    def in1_x(self) -> Optional[int]:
        return self.in1[1]

    @property
    def out_x(self) -> Optional[int]:
        return self.out0[1]


# --------------------------------------------------------------------------- #
# Module classifier. First matching rule wins (order matters).
# Each rule: (module_name, predicate(op) -> bool).
# --------------------------------------------------------------------------- #
Rule = tuple[str, Callable[[Op], bool]]

MODULE_RULES: list[Rule] = [
    # --- Routed MoE experts: the single fused-experts kernel dominates. -------
    ("moe_experts", lambda o: o.code == "FusedExpertsDeviceOperation"),
    # --- Router / gate: top-k selection, scatter, softmax over experts, and ---
    #     the gate matmul (hidden -> n_experts) + tiny per-expert reductions.
    ("moe_router", lambda o: o.code in ("TopKDeviceOperation", "ScatterDeviceOperation")),
    ("moe_router", lambda o: o.code == "MatmulDeviceOperation" and o.out_x == N_EXPERTS),
    ("moe_router", lambda o: o.code == "ReduceDeviceOperation" and o.out_x == N_EXPERTS),
    # --- Hyper-connection / hyper-head: operate on the stacked (hc_width) stream.
    ("hyper_connection", lambda o: HC_WIDTH in (o.in0_x, o.in1_x, o.out_x)),
    # --- RMSNorm (LayerNorm kernel) on the hidden stream. ---------------------
    ("rmsnorm", lambda o: o.code == "LayerNormDeviceOperation"),
    # --- Attention: softmax/transpose, kv & output projections (kv=1024), ------
    #     and the score/context matmuls + softmax-reduce on hidden width.
    ("attention", lambda o: o.code in ("SoftmaxDeviceOperation", "TransposeDeviceOperation")),
    ("attention", lambda o: o.code == "MatmulDeviceOperation" and KV in (o.in1_x, o.out_x)),
    ("attention", lambda o: o.code == "MatmulDeviceOperation"),
    ("attention", lambda o: o.code == "ReduceDeviceOperation"),
    # --- Layout / data-movement plumbing (tilize, untilize, copy, slice...). --
    (
        "layout_datamovement",
        lambda o: o.code
        in (
            "CopyDeviceOperation",
            "TilizeDeviceOperation",
            "TilizeWithValPaddingDeviceOperation",
            "UntilizeDeviceOperation",
            "UntilizeWithUnpaddingDeviceOperation",
            "ReshapeViewDeviceOperation",
            "SliceDeviceOperation",
            "FillPadDeviceOperation",
            "ConcatDeviceOperation",
            "RepeatDeviceOperation",
        ),
    ),
    # --- Elementwise (swiglu, residual adds, scaling, activations). -----------
    ("elementwise", lambda o: o.code in ("BinaryNgDeviceOperation", "UnaryDeviceOperation")),
]


def classify(op: Op) -> str:
    for name, pred in MODULE_RULES:
        try:
            if pred(op):
                return name
        except Exception:
            continue
    return "unclassified"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
@dataclass
class Bucket:
    count: int = 0
    total_ns: float = 0.0
    by_opcode: dict[str, list[float]] = field(default_factory=lambda: defaultdict(lambda: [0, 0.0]))

    def add(self, op: Op, value: float) -> None:
        self.count += 1
        self.total_ns += value
        rec = self.by_opcode[op.code]
        rec[0] += 1
        rec[1] += value


def parse_csv(path: str) -> list[Op]:
    ops: list[Op] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:

            def num(col: str) -> float:
                try:
                    return float(row.get(col) or 0)
                except ValueError:
                    return 0.0

            ops.append(
                Op(
                    code=row["OP CODE"],
                    device_ns=num("DEVICE KERNEL DURATION [ns]"),
                    host_ns=num("HOST DURATION [ns]"),
                    op_to_op_ns=num("OP TO OP LATENCY [ns]"),
                    in0=(_int(row.get("INPUT_0_Y_PAD[LOGICAL]", "")), _int(row.get("INPUT_0_X_PAD[LOGICAL]", ""))),
                    in1=(_int(row.get("INPUT_1_Y_PAD[LOGICAL]", "")), _int(row.get("INPUT_1_X_PAD[LOGICAL]", ""))),
                    out0=(_int(row.get("OUTPUT_0_Y_PAD[LOGICAL]", "")), _int(row.get("OUTPUT_0_X_PAD[LOGICAL]", ""))),
                    attributes=row.get("ATTRIBUTES", ""),
                )
            )
    return ops


METRICS = {
    "device": ("DEVICE KERNEL DURATION", lambda o: o.device_ns),
    "host": ("HOST DURATION", lambda o: o.host_ns),
    "op_to_op": ("OP-TO-OP LATENCY", lambda o: o.op_to_op_ns),
}


def analyze(ops: list[Op], metric: str) -> tuple[dict[str, Bucket], float]:
    _, getter = METRICS[metric]
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    total = 0.0
    for op in ops:
        v = getter(op)
        buckets[classify(op)].add(op, v)
        total += v
    return buckets, total


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _us(ns: float) -> float:
    return ns / 1e3


def print_report(buckets: dict[str, Bucket], total: float, metric: str, n_ops: int, show_ops: bool) -> None:
    label, _ = METRICS[metric]
    total = total or 1.0
    print()
    print(f"  DeepSeek-V4-Flash profile breakdown  ({n_ops} ops, metric = {label})")
    print(f"  total {label.lower()}: {_us(total):,.1f} us")
    print("=" * 72)
    print(f"  {'module':<22}{'ops':>7}{'time [us]':>16}{'% total':>10}")
    print("-" * 72)
    order = sorted(buckets.items(), key=lambda kv: -kv[1].total_ns)
    for name, b in order:
        print(f"  {name:<22}{b.count:>7}{_us(b.total_ns):>16,.1f}{100 * b.total_ns / total:>9.1f}%")
    print("-" * 72)
    print(f"  {'TOTAL':<22}{n_ops:>7}{_us(total):>16,.1f}{100.0:>9.1f}%")

    if show_ops:
        print()
        print("  per-op-code breakdown within each module")
        print("=" * 72)
        for name, b in order:
            print(f"\n  [{name}]  {_us(b.total_ns):,.1f} us")
            for code, (cnt, t) in sorted(b.by_opcode.items(), key=lambda kv: -kv[1][1]):
                print(f"    {code:<42}{cnt:>6}{_us(t):>13,.1f} us")
    print()


def analyze_signposts(path: str, metric: str) -> tuple[dict[str, Bucket], float]:
    """Ground-truth attribution using ``_region`` Tracy signposts.

    Signpost markers land in the profile as pseudo-op rows whose ``OP CODE`` ends
    in ``_START`` / ``_END``. Walking the rows in execution order with a region
    stack assigns every real device op to its innermost enclosing region (so the
    nested ``MOE_ROUTER`` / ``MOE_EXPERTS`` / ``MOE_SHARED`` split out from their
    parent ``MOE``). Ops issued outside any region (embedding, lm_head, CCL,
    weight-load setup) land in ``(outside_regions)``.
    """
    _, getter = METRICS[metric]
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    stack: list[str] = []
    total = 0.0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            code = row["OP CODE"]
            if code.endswith("_START"):
                stack.append(code[: -len("_START")])
                continue
            if code.endswith("_END"):
                name = code[: -len("_END")]
                if name in stack:  # pop back to the matching frame (robust to gaps)
                    del stack[stack.index(name) :]
                continue
            op = Op(
                code=code,
                device_ns=float(row.get("DEVICE KERNEL DURATION [ns]") or 0),
                host_ns=float(row.get("HOST DURATION [ns]") or 0),
                op_to_op_ns=float(row.get("OP TO OP LATENCY [ns]") or 0),
                in0=(None, None),
                in1=(None, None),
                out0=(None, None),
                attributes="",
            )
            v = getter(op)
            buckets[stack[-1] if stack else "(outside_regions)"].add(op, v)
            total += v
    return buckets, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path to ops_perf_results_*.csv")
    ap.add_argument(
        "--metric",
        choices=list(METRICS),
        default="device",
        help="which time column to aggregate (default: device kernel duration)",
    )
    ap.add_argument("--no-ops", action="store_true", help="hide the per-op-code breakdown")
    ap.add_argument(
        "--mode",
        choices=("auto", "signpost", "heuristic"),
        default="auto",
        help="attribution mode: signpost (Tracy regions), heuristic (op/shape rules), "
        "or auto (signpost if markers present, else heuristic)",
    )
    ap.add_argument("--json", metavar="PATH", help="also dump the breakdown as JSON")
    args = ap.parse_args()

    ops = parse_csv(args.csv)
    has_markers = any(o.code.endswith(("_START", "_END")) for o in ops)
    use_signpost = args.mode == "signpost" or (args.mode == "auto" and has_markers)

    if use_signpost:
        if not has_markers:
            print("  no signpost markers found in this CSV; re-run with --mode heuristic")
            return 1
        buckets, total = analyze_signposts(args.csv, args.metric)
        n_ops = sum(b.count for b in buckets.values())
        print("  attribution: SIGNPOST regions (ground truth)")
    else:
        # Heuristic mode: drop the marker pseudo-ops before classifying.
        ops = [o for o in ops if not o.code.endswith(("_START", "_END"))]
        buckets, total = analyze(ops, args.metric)
        n_ops = len(ops)
        print("  attribution: HEURISTIC op/shape rules")
    print_report(buckets, total, args.metric, n_ops, show_ops=not args.no_ops)

    if args.json:
        out = {
            "metric": args.metric,
            "total_ns": total,
            "n_ops": len(ops),
            "modules": {
                name: {
                    "ops": b.count,
                    "total_ns": b.total_ns,
                    "pct": 100 * b.total_ns / (total or 1.0),
                    "by_opcode": {c: {"ops": cnt, "total_ns": t} for c, (cnt, t) in b.by_opcode.items()},
                }
                for name, b in buckets.items()
            },
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
