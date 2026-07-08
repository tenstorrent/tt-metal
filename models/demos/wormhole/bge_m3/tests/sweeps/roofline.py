# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generalized roofline analysis for a tt-metal device-profiler ops CSV.

Given an ``ops_perf_results_*.csv`` produced by ``TT_METAL_DEVICE_PROFILER=1``
tracy runs, this classifies every op as compute-bound or bandwidth-bound by
comparing its achieved FLOP/s and GB/s against Wormhole roofline peaks.

It is model-agnostic: FLOPs are inferred from tensor shapes (matmul-like ops
get ``2*M*K*N``; attention/SDPA ops get the flash-attention formula), and bytes
are summed over all input/output tensors from their shapes + datatypes. Ops it
cannot estimate FLOPs for are treated as pure bandwidth (elementwise, TM, norm).

Why it matters: it tells you *which knob* helps a given op. Compute-bound ops
(achieved FLOP/s near peak, low BW) benefit from lower math fidelity / better
block sizes; bandwidth-bound ops (achieved GB/s near peak, low FLOP/s) benefit
from smaller dtypes, fusion, or keeping tensors resident. That is exactly how
the B12/S8192 SDPA HiFi2->LoFi win was found (SDPA was compute-bound at
~33 TFLOP/s, so cutting fidelity ~halved its matmul phase).

Usage (standalone, from tt-metal root):
    # 1. produce a CSV with a profiled forward pass, e.g.:
    TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r \\
      --no-runtime-analysis -v -m pytest \\
      models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py -k b12_s8192 -sv

    # 2. analyze the newest CSV (auto-discovered) or pass one explicitly:
    python models/demos/wormhole/bge_m3/tests/sweeps/roofline.py
    python models/demos/wormhole/bge_m3/tests/sweeps/roofline.py path/to/ops.csv

    # only rows between the 'start'/'stop' signposts (default: on):
    python .../roofline.py --no-signpost   # analyze the whole CSV instead

    # override the number of forward passes captured (for per-pass numbers):
    python .../roofline.py --passes 2
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

# ── Wormhole b0 roofline peaks (approximate, single chip) ────────────────────
# These are order-of-magnitude reference peaks used only to classify an op as
# compute- vs bandwidth-bound. Adjust if you have measured numbers for your part.
WH_PEAK_TFLOPS = {
    "LoFi": 74.0,  # bf8 x bf8, lowest fidelity (1 math pass)
    "HiFi2": 37.0,  # 2 passes
    "HiFi3": 25.0,
    "HiFi4": 18.5,  # 4 passes
}
WH_DRAM_GBPS = 288.0  # aggregate DRAM bandwidth (approx)

# Bytes per element per datatype (bf8 includes the tile exponent-section overhead).
DTYPE_BYTES = {
    "BFLOAT8_B": 1.0625,
    "BFLOAT4_B": 0.5625,
    "BFLOAT16": 2.0,
    "FLOAT32": 4.0,
    "UINT32": 4.0,
    "INT32": 4.0,
    "UINT16": 2.0,
    "UINT8": 1.0,
}

DUR_COL = "DEVICE FW DURATION [ns]"


def _find_latest_csv() -> str:
    pattern = "generated/profiler/reports/*/ops_perf_results_*.csv"
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"no CSV matched {pattern!r} — run a profiler pass first")
    return files[-1]


def _num(v):
    try:
        return float(str(v).split("[")[0])
    except (ValueError, TypeError):
        return float("nan")


def _shape(row, idx: int):
    """Return (W, Z, Y, X) logical shape for INPUT_<idx> / OUTPUT_<idx>, or None."""
    dims = []
    for ax in ("W", "Z", "Y", "X"):
        col = f"{idx}_{ax}_PAD[LOGICAL]"
        if col not in row or pd.isna(row[col]):
            return None
        dims.append(int(_num(row[col])))
    return tuple(dims)


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _dtype_bytes(row, key: str) -> float:
    dt = str(row.get(f"{key}_DATATYPE", "")).strip().upper()
    return DTYPE_BYTES.get(dt, 2.0)


def _tensor_bytes(row) -> float:
    """Sum bytes over all present INPUT_* and OUTPUT_* tensors."""
    total = 0.0
    for kind in ("INPUT", "OUTPUT"):
        for i in range(4):
            key = f"{kind}_{i}"
            sh = _shape(row, key)
            if sh is None:
                continue
            total += _numel(sh) * _dtype_bytes(row, key)
    return total


def _matmul_flops(row) -> float | None:
    """2*M*K*N if INPUT_0=[..,M,K] and INPUT_1=[..,K,N] with matching K."""
    a = _shape(row, "INPUT_0")
    b = _shape(row, "INPUT_1")
    if a is None or b is None:
        return None
    m, k = a[-2], a[-1]
    k2, n = b[-2], b[-1]
    if k != k2 or k == 0 or n == 0:
        return None
    batch = max(1, a[0] * a[1])
    return 2.0 * batch * m * k * n


def _sdpa_flops(row) -> float | None:
    """Flash attention: QK^T + softmax@V ≈ 2 * 2 * B*H*S*S*d."""
    q = _shape(row, "INPUT_0")
    if q is None:
        return None
    b, h, s, d = q
    return 2.0 * 2.0 * b * h * s * s * d


def _estimate_flops(op_code: str, row) -> float | None:
    name = op_code.lower()
    if "sdpa" in name or "scaleddotproduct" in name or "attention" in name:
        f = _sdpa_flops(row)
        if f:
            return f
    if "matmul" in name or "linear" in name:
        return _matmul_flops(row)
    # generic: try matmul shape inference (covers custom matmul-like ops)
    return _matmul_flops(row)


def _peak_tflops_for(row) -> float:
    """Pick the roofline compute peak from the op's math fidelity if recorded."""
    attrs = str(row.get("ATTRIBUTES", ""))
    m = re.search(r"MathFidelity::(\w+)", attrs)
    if m and m.group(1) in WH_PEAK_TFLOPS:
        return WH_PEAK_TFLOPS[m.group(1)]
    return WH_PEAK_TFLOPS["HiFi2"]


def analyze(csv_path: str, use_signpost: bool = True, passes: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if use_signpost and (df["OP TYPE"] == "signpost").any():
        sp = df[df["OP TYPE"] == "signpost"]
        starts = sp[sp["OP CODE"] == "start"].index
        stops = sp[sp["OP CODE"] == "stop"].index
        if len(starts) and len(stops):
            df = df.loc[starts[0] + 1 : stops[-1] - 1]

    ops = df[df["OP TYPE"] == "tt_dnn_device"].copy() if "OP TYPE" in df else df.copy()
    ops[DUR_COL] = pd.to_numeric(ops[DUR_COL], errors="coerce")

    rows = []
    for _, r in ops.iterrows():
        dur_ns = r[DUR_COL]
        if pd.isna(dur_ns) or dur_ns <= 0:
            continue
        op = str(r["OP CODE"])
        flops = _estimate_flops(op, r)
        nbytes = _tensor_bytes(r)
        secs = dur_ns / 1e9
        tflops = (flops / 1e12 / secs) if flops else None
        gbps = (nbytes / 1e9 / secs) if nbytes else None
        peak_tf = _peak_tflops_for(r)
        compute_pct = (tflops / peak_tf * 100) if tflops else None
        bw_pct = (gbps / WH_DRAM_GBPS * 100) if gbps else None
        # bound classification: whichever utilization is higher
        bound = "?"
        if compute_pct is not None or bw_pct is not None:
            c = compute_pct or 0
            b = bw_pct or 0
            bound = "compute" if c >= b else "bandwidth"
        rows.append(
            {
                "op": op,
                "cores": int(_num(r.get("CORE COUNT", "nan"))) if not pd.isna(_num(r.get("CORE COUNT", "nan"))) else 0,
                "dur_us": dur_ns / 1e3,
                "tflops": tflops,
                "compute%": compute_pct,
                "gbps": gbps,
                "bw%": bw_pct,
                "bound": bound,
            }
        )

    per_op = pd.DataFrame(rows)
    if per_op.empty:
        raise RuntimeError("no device ops found in CSV")

    if passes is None:
        # infer #passes from the most frequent op count being a multiple (best-effort)
        passes = 1

    return per_op


def _fmt(x, spec):
    return format(x, spec) if x is not None and not pd.isna(x) else "   —  "


def report(per_op: pd.DataFrame, passes: int) -> None:
    dur = per_op["dur_us"]
    total_ms = dur.sum() / 1e3

    # aggregate by op code (mean of rates, sum of time)
    agg = per_op.groupby("op").agg(
        tot_us=("dur_us", "sum"),
        cnt=("dur_us", "count"),
        mean_us=("dur_us", "mean"),
        tflops=("tflops", "mean"),
        compute=("compute%", "mean"),
        gbps=("gbps", "mean"),
        bw=("bw%", "mean"),
    )
    agg = agg.sort_values("tot_us", ascending=False)

    print(f"\n{'='*104}")
    print(f"ROOFLINE  (total device-op time {total_ms:.1f} ms over {passes} pass(es) = {total_ms/passes:.1f} ms/pass)")
    print(f"peaks: compute {WH_PEAK_TFLOPS} TFLOP/s   DRAM {WH_DRAM_GBPS} GB/s")
    print("=" * 104)
    hdr = (
        f"{'op':34} {'tot_ms':>8} {'%':>5} {'cnt':>4} {'mean_us':>8} "
        f"{'TFLOP/s':>8} {'cmp%':>5} {'GB/s':>7} {'bw%':>5}  bound"
    )
    print(hdr)
    print("-" * 104)
    for op, r in agg.iterrows():
        pct = r["tot_us"] / dur.sum() * 100
        c = r["compute"] or 0
        b = r["bw"] or 0
        bound = "compute" if c >= b else "bandwidth"
        print(
            f"{op[:34]:34} {r['tot_us']/1e3:8.1f} {pct:5.1f} {int(r['cnt']):4d} {r['mean_us']:8.1f} "
            f"{_fmt(r['tflops'],'8.1f')} {_fmt(r['compute'],'5.0f')} {_fmt(r['gbps'],'7.1f')} "
            f"{_fmt(r['bw'],'5.0f')}  {bound}"
        )
    print("-" * 104)
    print(
        "read: cmp% = achieved/compute-peak, bw% = achieved/DRAM-peak. "
        "High cmp% -> lower fidelity/better blocks help; high bw% -> smaller dtype/fusion helps."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generalized roofline analysis for a tt-metal ops CSV.")
    ap.add_argument("csv", nargs="?", default=None, help="ops CSV path (default: newest under generated/profiler)")
    ap.add_argument("--no-signpost", action="store_true", help="analyze the whole CSV, not just start/stop signposts")
    ap.add_argument("--passes", type=int, default=1, help="number of forward passes captured (for per-pass timing)")
    args = ap.parse_args()

    csv_path = args.csv or _find_latest_csv()
    print(f"CSV: {csv_path}")
    per_op = analyze(csv_path, use_signpost=not args.no_signpost, passes=args.passes)
    report(per_op, args.passes)


if __name__ == "__main__":
    main()
