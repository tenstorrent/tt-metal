# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-layer device-time / op-to-op-gap post-processor for Kimi/DeepSeek chunked prefill.

Consumes a Tracy ops-perf CSV (``ops_perf_results_*.csv``) produced by
``python -m tracy -r ...`` and reports, for the warm iteration:

  * per op (one logical op runs on all 32 devices at once): min/max device kernel
    time and min/max op-to-op gap across devices;
  * per layer: the sum of device time and the sum of op-to-op gap (critical-path =
    sum of per-op max across devices; the sum of per-op min is also shown);
  * an MLA-vs-FFN/MoE sub-split within each layer;
  * a grand total across layers with the overall op2op fraction.

Segmentation relies on signposts emitted by the model / test:
  * ``iter_{i}_start`` / ``iter_{i}_end``      -- iteration boundaries (test driver)
  * ``forward_layer_{i}_start`` / ``..._end``  -- layer boundaries (tt_prefill_transformer)
  * ``MLA_START`` / ``MLA_END``                -- attention sub-region (mla.py)

Why positional segmentation: ``GLOBAL CALL COUNT`` is a per-device runtime id (it
differs across the 32 device rows of the SAME logical op, and signpost rows carry
NaN), so it cannot key ops. Instead we walk the CSV in its native host-dispatch
order: signpost rows appear at their host position and never split a logical op, and
one logical op is the run of consecutive non-signpost device rows sharing an OP CODE
until a device id repeats (the next op's first device row) or a signpost intervenes.

CLI::

    python -m models.demos.deepseek_v3_d_p.utils.perlayer_op2op <csv|dir|glob> [--iter N]

``<dir>`` resolves to the newest ``ops_perf_results_*.csv`` under it (recursively).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import pandas as pd

SIGNPOST_TYPES = {"signpost", "tt_signpost"}
KERNEL_COL = "DEVICE KERNEL DURATION [ns]"
OP2OP_COL = "OP TO OP LATENCY [ns]"
OPCODE_COL = "OP CODE"
OPTYPE_COL = "OP TYPE"
DEVID_COL = "DEVICE ID"
NS_PER_US = 1000.0


# --------------------------------------------------------------------------------------
# CSV location
# --------------------------------------------------------------------------------------
def resolve_csv(path: str) -> str:
    """Accept a CSV file, a directory (searched recursively for the newest
    ops_perf_results_*.csv), or a glob pattern, and return a single CSV path."""
    if os.path.isdir(path):
        candidates = glob.glob(os.path.join(path, "**", "ops_perf_results_*.csv"), recursive=True)
        if not candidates:
            raise FileNotFoundError(f"no ops_perf_results_*.csv found under {path}")
        # Newest by mtime (report-gen is slow; take the most recently completed).
        return max(candidates, key=os.path.getmtime)
    matches = glob.glob(path)
    if len(matches) > 1:
        return max(matches, key=os.path.getmtime)
    if matches:
        return matches[0]
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"no CSV matching {path!r}")


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
class Op:
    """One logical op = its rows across (ideally 32) devices."""

    __slots__ = ("op_code", "bucket", "device_ids", "kernels", "op2ops")

    def __init__(self, op_code: str, bucket: str):
        self.op_code = op_code
        self.bucket = bucket
        self.device_ids: set = set()
        self.kernels: list[float] = []
        self.op2ops: list[float] = []

    def add(self, device_id, kernel_ns, op2op_ns):
        self.device_ids.add(device_id)
        if kernel_ns is not None and not math.isnan(kernel_ns):
            self.kernels.append(kernel_ns)
        if op2op_ns is not None and not math.isnan(op2op_ns):
            self.op2ops.append(op2op_ns)

    @property
    def n_dev(self) -> int:
        return len(self.device_ids)

    def _mm(self, vals):
        return (min(vals), max(vals)) if vals else (float("nan"), float("nan"))

    @property
    def kern_min_max(self):
        return self._mm(self.kernels)

    @property
    def op2op_min_max(self):
        return self._mm(self.op2ops)


def _as_float(v):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return float("nan")


def parse_perlayer(csv_path: str, iteration: int = 1):
    """Return (layers, warnings) where layers is an ordered dict {layer_idx: [Op, ...]}
    for the selected iteration, in execution order."""
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for col in (OPCODE_COL, OPTYPE_COL, DEVID_COL, KERNEL_COL, OP2OP_COL):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in CSV (have: {list(df.columns)})")

    warnings: list[str] = []
    layers: dict[int, list[Op]] = {}

    iter_start = f"iter_{iteration}_start"
    iter_end = f"iter_{iteration}_end"
    headers_present = set(
        df.loc[df[OPTYPE_COL].astype(str).str.strip().isin(SIGNPOST_TYPES), OPCODE_COL].astype(str).str.strip()
    )
    have_iter = iter_start in headers_present
    if not have_iter:
        warnings.append(
            f"signpost {iter_start!r} not found; processing the WHOLE file "
            f"(no iteration filter). Present iter markers: "
            f"{sorted(h for h in headers_present if h.startswith('iter_'))}"
        )

    in_iter = not have_iter  # if no iter markers, treat everything as in-region
    cur_layer = None
    bucket = None
    cur_op: Op | None = None

    def flush():
        nonlocal cur_op
        if cur_op is not None and cur_layer is not None:
            layers.setdefault(cur_layer, []).append(cur_op)
        cur_op = None

    # Extract columns as positional lists (itertuples/_asdict mangles names with spaces/brackets).
    op_types = df[OPTYPE_COL].astype(str).str.strip().tolist()
    op_codes = df[OPCODE_COL].astype(str).str.strip().tolist()
    dev_ids = df[DEVID_COL].tolist()
    kern_list = df[KERNEL_COL].tolist()
    o2o_list = df[OP2OP_COL].tolist()

    for idx in range(len(df)):
        op_type = op_types[idx]
        op_code = op_codes[idx]

        if op_type in SIGNPOST_TYPES:
            # Signposts never belong to an op and always close the current group.
            flush()
            if op_code == iter_start:
                in_iter = True
            elif op_code == iter_end:
                in_iter = False
                cur_layer, bucket = None, None
            elif op_code.startswith("forward_layer_") and op_code.endswith("_start"):
                try:
                    cur_layer = int(op_code[len("forward_layer_") : -len("_start")])
                except ValueError:
                    cur_layer = None
                bucket = "pre"
            elif op_code.startswith("forward_layer_") and op_code.endswith("_end"):
                cur_layer, bucket = None, None
            elif op_code == "MLA_START":
                bucket = "mla"
            elif op_code == "MLA_END":
                bucket = "ffn"
            continue

        if not in_iter or cur_layer is None:
            continue

        device_id = dev_ids[idx]
        if device_id is None or (isinstance(device_id, float) and math.isnan(device_id)):
            continue  # host-only row, no device data
        kernel_ns = _as_float(kern_list[idx])
        op2op_ns = _as_float(o2o_list[idx])

        # Start a new logical op when the op code changes or this device id was already
        # seen in the current group (i.e. we have wrapped into the next op).
        if cur_op is None or cur_op.op_code != op_code or device_id in cur_op.device_ids:
            flush()
            cur_op = Op(op_code, bucket or "ffn")
        cur_op.add(device_id, kernel_ns, op2op_ns)

    flush()

    # Sanity: every op should span 32 devices.
    for li, ops in layers.items():
        bad = [(i, o.op_code, o.n_dev) for i, o in enumerate(ops) if o.n_dev != 32]
        if bad:
            warnings.append(
                f"layer {li}: {len(bad)} op(s) with n_dev != 32 (positional grouping may have "
                f"split/merged an op with missing device rows): "
                + ", ".join(f"[{i}] {c} n_dev={n}" for i, c, n in bad[:5])
                + (" ..." if len(bad) > 5 else "")
            )
    return dict(sorted(layers.items())), warnings


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def _fmt(us: float) -> str:
    return "nan" if math.isnan(us) else f"{us:10.2f}"


def perlayer_report(
    layers: dict[int, list[Op]], warnings: list[str], iteration: int, exclude_layer_entry: bool = False
) -> str:
    lines: list[str] = []
    p = lines.append

    p("=" * 118)
    p(f"Per-layer device-time / op2op report  (iteration {iteration}, warm; times in microseconds)")
    p("Per op: values are across the ~32 devices. Per layer: SUM over ops of the per-op MAX (critical")
    p("path, the wall-clock-relevant number); the SUM of per-op MIN is shown alongside for spread.")
    p("=" * 118)
    for w in warnings:
        p(f"  [warn] {w}")
    if warnings:
        p("")

    g_dev_max = g_dev_min = g_o2o_max = g_o2o_min = 0.0
    g_entry_gap = 0.0  # sum of excluded op2op gaps (region-entry sync; per-layer flush if enabled)
    bucket_tot = {"pre": [0.0, 0.0], "mla": [0.0, 0.0], "ffn": [0.0, 0.0]}  # [dev_max, o2o_max]

    first_layer = next(iter(layers)) if layers else None
    for li, ops in layers.items():
        p("")
        p("-" * 118)
        p(f"LAYER {li}   ({len(ops)} ops)")
        p("-" * 118)
        hdr = (
            f"  {'#':>3} {'bucket':<6} {'OP CODE':<40} {'ndev':>4} "
            f"{'kern_min':>10} {'kern_max':>10} {'o2o_min':>10} {'o2o_max':>10}"
        )
        p(hdr)
        l_dev_max = l_dev_min = l_o2o_max = l_o2o_min = 0.0
        entry_gap = float("nan")  # op 0's op2op = gap entering the layer (from the previous layer / iter sync)
        for i, o in enumerate(ops):
            kmin, kmax = (v / NS_PER_US for v in o.kern_min_max)
            omin, omax = (v / NS_PER_US for v in o.op2op_min_max)
            # Exclude an op's op2op from the totals only when it is the region-entry sync boundary:
            # always the very first op of the whole measured region (layer0/op0 = iter-boundary sync),
            # and — with per-layer profiler flushing — every layer's op0. Otherwise op0 of layers 1..N is
            # a genuine inter-layer dispatch gap and is counted.
            excluded = (i == 0) and (exclude_layer_entry or li == first_layer)
            tag = "  <-- region-entry sync (excl. from op2op)" if excluded else ""
            p(
                f"  {i:>3} {o.bucket:<6} {o.op_code[:40]:<40} {o.n_dev:>4} "
                f"{_fmt(kmin)} {_fmt(kmax)} {_fmt(omin)} {_fmt(omax)}{tag}"
            )
            if not math.isnan(kmax):
                l_dev_max += kmax
                bucket_tot[o.bucket][0] += kmax
            if not math.isnan(kmin):
                l_dev_min += kmin
            if i == 0:
                entry_gap = omax
            if excluded:
                continue
            if not math.isnan(omax):
                l_o2o_max += omax
                bucket_tot[o.bucket][1] += omax
            if not math.isnan(omin):
                l_o2o_min += omin

        # Per-layer MLA vs FFN split (device max = all ops; op2op max = counted ops only).
        def _counted(o, idx):
            return not ((idx == 0) and (exclude_layer_entry or li == first_layer))

        mla_dev = sum(o.kern_min_max[1] for o in ops if o.bucket == "mla") / NS_PER_US
        ffn_dev = sum(o.kern_min_max[1] for o in ops if o.bucket in ("ffn", "pre")) / NS_PER_US
        mla_o2o = sum(o.op2op_min_max[1] for i, o in enumerate(ops) if _counted(o, i) and o.bucket == "mla") / NS_PER_US
        ffn_o2o = (
            sum(o.op2op_min_max[1] for i, o in enumerate(ops) if _counted(o, i) and o.bucket in ("ffn", "pre"))
            / NS_PER_US
        )
        p("")
        p(
            f"  layer {li} SUM (critical path = per-op MAX): device={l_dev_max:10.2f} us   "
            f"op2op={l_o2o_max:10.2f} us   total={l_dev_max + l_o2o_max:10.2f} us"
        )
        p(f"           (sum of per-op MIN)      : device={l_dev_min:10.2f} us   op2op={l_o2o_min:10.2f} us")
        p(f"           MLA : device={mla_dev:10.2f} us   op2op={mla_o2o:10.2f} us")
        p(f"           FFN : device={ffn_dev:10.2f} us   op2op={ffn_o2o:10.2f} us")
        if (exclude_layer_entry or li == first_layer) and not math.isnan(entry_gap):
            p(f"           region-entry gap (op 0; sync boundary, NOT counted): {entry_gap:12.2f} us")
        g_dev_max += l_dev_max
        g_dev_min += l_dev_min
        g_o2o_max += l_o2o_max
        g_o2o_min += l_o2o_min
        if (exclude_layer_entry or li == first_layer) and not math.isnan(entry_gap):
            g_entry_gap += entry_gap

    grand_total = g_dev_max + g_o2o_max
    o2o_pct = (100.0 * g_o2o_max / grand_total) if grand_total else 0.0
    p("")
    p("=" * 118)
    p("GRAND TOTAL across layers  (critical path = sum of per-op MAX; op2op EXCLUDES the region-entry sync)")
    p("=" * 118)
    p(f"  device kernel time  : {g_dev_max:12.2f} us   (sum of per-op MIN: {g_dev_min:12.2f} us)")
    p(f"  op2op gap           : {g_o2o_max:12.2f} us   (sum of per-op MIN: {g_o2o_min:12.2f} us)")
    p(f"  total (device+op2op): {grand_total:12.2f} us")
    p(f"  op2op fraction      : {o2o_pct:6.1f} %")
    p("")
    p("  by sub-region (device max / op2op max, us):")
    for b in ("pre", "mla", "ffn"):
        p(f"    {b:<4}: device={bucket_tot[b][0]:12.2f}   op2op={bucket_tot[b][1]:12.2f}")
    p("")
    p(f"  excluded region-entry sync gap(s) total (NOT part of op2op above): {g_entry_gap:12.2f} us")
    if exclude_layer_entry:
        p("  Note: --exclude-layer-entry set -> each layer's first-op gap (per-layer profiler-flush sync)")
        p("  is excluded. Otherwise only the very first op (iter-boundary sync) is excluded, and the")
        p("  inter-layer dispatch gaps (layers 1..N op 0) ARE counted. Device kernel times are unaffected.")
    else:
        p("  Note: sync fires only per iteration, so the only excluded gap is the very first op's")
        p("  (the iter-boundary sync entering the measured region). All inter-layer gaps are counted.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="ops_perf_results_*.csv file, a directory to search, or a glob")
    ap.add_argument("--iter", type=int, default=1, help="iteration to report (default 1 = warm; 0 = cold compile)")
    ap.add_argument("--report", default=None, help="output text file (default: <csv_dir>/perlayer_report.txt)")
    ap.add_argument(
        "--exclude-layer-entry",
        action="store_true",
        help="exclude EVERY layer's first-op op2op (use when the run flushed the profiler per layer, "
        "TT_PERLAYER_PROFILER_FLUSH=1). Default: only the very first op (iter-boundary sync) is excluded.",
    )
    args = ap.parse_args(argv)

    csv_path = resolve_csv(args.path)
    print(f"[perlayer_op2op] parsing {csv_path} (iter {args.iter})", file=sys.stderr)
    layers, warnings = parse_perlayer(csv_path, iteration=args.iter)
    if not layers:
        print(
            f"[perlayer_op2op] no layers found for iter {args.iter}. " f"Warnings: {warnings}",
            file=sys.stderr,
        )
        return 1
    report = perlayer_report(layers, warnings, args.iter, exclude_layer_entry=args.exclude_layer_entry)
    print(report)

    out = args.report or os.path.join(os.path.dirname(csv_path), "perlayer_report.txt")
    with open(out, "w") as f:
        f.write(report + "\n")
    print(f"\n[perlayer_op2op] report written to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
