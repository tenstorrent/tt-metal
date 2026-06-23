# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Per-layer, per-op device-kernel duration + op-to-op gap breakdown from a tracy ops CSV.

Decomposes the measured (warm) region of a chunked-prefill device-perf capture into, for each
transformer layer, every op's on-device kernel time and the host-dispatch gap before it
(``OP TO OP LATENCY [ns]``). The sum of (kernel + op2op) over one device's op stream ≈ the
wall-clock of the measured region, so this splits "device compute" from "host/dispatch tax"
op-by-op and layer-by-layer.

Parsing model (matches the CSV emitted by the no-PCC worker under tracy):
  - Each device op appears once per ``DEVICE ID`` (32 rows on an 8x4 mesh). We LOCK to the WORST
    device — the one with the largest total (kernel + op2op) over the measured region, i.e. the
    critical-path chip — so the breakdown reflects the bottleneck that gates wall-clock. op2op is a
    host-dispatch property, so a single device's stream is the right unit to sum.
  - Signpost rows (``OP TYPE == "signpost"``) carry no device id. ``forward_layer_{i}_start`` /
    ``forward_layer_{i}_end`` (emitted in TtPrefillTransformer.forward) delimit each layer; ops
    before ``forward_layer_0_start`` are bucketed as "embed/pre", ops after the last
    ``forward_layer_*_end`` as "norm/lm_head/post".
  - Only the region between ``PROFILE_MEASURE_START`` and ``PROFILE_MEASURE_END`` is considered, so a
    preceding warmup/compile chunk is excluded.
  - The very first op of the measured region carries a spurious op2op gap (idle since the warmup
    chunk); its op2op is clipped to 0 for the totals (shown raw in the table, flagged).
  - With N measured chunks the same (layer, op-index) recurs N times; values are averaged → per-chunk.

CLI: ``python -m models.demos.deepseek_v3_d_p.utils.perlayer_op2op <ops_perf_results.csv>``
"""

import sys
from dataclasses import dataclass, field

import pandas as pd

MEASURE_START = "PROFILE_MEASURE_START"
MEASURE_END = "PROFILE_MEASURE_END"
_KERNEL = "DEVICE KERNEL DURATION [ns]"
_OP2OP = "OP TO OP LATENCY [ns]"


@dataclass
class _OpAgg:
    op_code: str
    kernel_ns: float = 0.0
    op2op_ns: float = 0.0
    count: int = 0
    region_start: bool = False  # first op of the whole measured region (op2op clipped in totals)


@dataclass
class _LayerAgg:
    label: str
    ops: list = field(default_factory=list)  # list[_OpAgg], one per op-index


def _measure_region(df: pd.DataFrame) -> pd.DataFrame:
    """Rows strictly between the last MEASURE_START and the next MEASURE_END (or EOF)."""
    starts = df.index[df["OP CODE"] == MEASURE_START].tolist()
    if not starts:
        raise ValueError(f"{MEASURE_START!r} signpost not found in CSV (was the worker run under tracy with warmup?)")
    start = starts[-1]
    ends = [i for i in df.index[df["OP CODE"] == MEASURE_END].tolist() if i > start]
    end = ends[0] if ends else len(df)
    return df.loc[start + 1 : end - 1]


def _pick_worst_device(dev_rows: pd.DataFrame):
    """The critical-path device: max total (kernel + op2op) over the region, with each device's first
    measured op2op excluded (spurious startup-idle gap that's ~equal across devices)."""
    worst_dev, worst_total = None, -1.0
    for dev, g in dev_rows.groupby("DEVICE ID"):
        kernel = g[_KERNEL].fillna(0).sum()
        op2op = g[_OP2OP].fillna(0)
        op2op_excl_first = op2op.iloc[1:].sum() if len(op2op) > 1 else 0.0
        total = kernel + op2op_excl_first
        if total > worst_total:
            worst_dev, worst_total = dev, total
    return worst_dev


def parse_perlayer(csv_path: str) -> tuple[list, dict]:
    """Return (layers, totals). ``layers`` is a list[_LayerAgg] in order (embed/pre, L0..Ln, post);
    ``totals`` has per-chunk kernel_us / op2op_us / total_us / op2op_pct / n_chunks."""
    df = pd.read_csv(csv_path, low_memory=False)
    region = _measure_region(df)

    dev_rows = region[region["OP TYPE"] == "tt_dnn_device"]
    if dev_rows.empty:
        raise ValueError("no device ops in the measured region")
    lock_dev = _pick_worst_device(dev_rows)

    # n_chunks = how many times forward_layer_0_start appears in the region (one per measured chunk).
    n_chunks = max(1, int((region["OP CODE"] == "forward_layer_0_start").sum()))

    # Walk in dispatch order; signposts switch the current layer bucket; device rows on lock_dev append.
    layers: dict[str, _LayerAgg] = {}
    order: list[str] = []

    def bucket(label: str) -> _LayerAgg:
        if label not in layers:
            layers[label] = _LayerAgg(label=label)
            order.append(label)
        return layers[label]

    cur = "embed/pre"
    op_idx = 0  # op index within the current (layer, occurrence)
    first_region_op = True
    for _, row in region.iterrows():
        op_type = row["OP TYPE"]
        code = row["OP CODE"]
        if op_type == "signpost":
            if isinstance(code, str) and code.startswith("forward_layer_") and code.endswith("_start"):
                cur = "L" + code[len("forward_layer_") : -len("_start")]
                op_idx = 0
            elif isinstance(code, str) and code.startswith("forward_layer_") and code.endswith("_end"):
                cur = "norm/lm_head/post"
                op_idx = 0
            continue
        if op_type != "tt_dnn_device" or row["DEVICE ID"] != lock_dev:
            continue
        lyr = bucket(cur)
        if op_idx >= len(lyr.ops):
            lyr.ops.append(_OpAgg(op_code=str(code)))
        agg = lyr.ops[op_idx]
        agg.kernel_ns += float(row.get(_KERNEL, 0) or 0)
        op2op = float(row.get(_OP2OP, 0) or 0)
        if first_region_op:
            agg.region_start = True  # spurious startup-idle gap, clipped in totals
            first_region_op = False
        else:
            agg.op2op_ns += op2op
        agg.count += 1
        op_idx += 1

    ordered = [layers[k] for k in order]

    tot_kernel = sum(o.kernel_ns for l in ordered for o in l.ops)
    tot_op2op = sum(o.op2op_ns for l in ordered for o in l.ops)  # region-start op already excluded
    tot_kernel_pc = tot_kernel / n_chunks
    tot_op2op_pc = tot_op2op / n_chunks
    grand = tot_kernel_pc + tot_op2op_pc
    totals = {
        "n_chunks": n_chunks,
        "lock_device": lock_dev,
        "kernel_us": tot_kernel_pc / 1e3,
        "op2op_us": tot_op2op_pc / 1e3,
        "total_us": grand / 1e3,
        "op2op_pct": (tot_op2op_pc / grand * 100) if grand else float("nan"),
    }
    return ordered, totals


def format_perlayer(layers: list, totals: dict) -> str:
    n = totals["n_chunks"]
    lines = []
    lines.append("=" * 96)
    lines.append(
        f"Per-layer device-kernel vs op2op (per-chunk avg over {n} measured chunk(s); "
        f"locked to WORST/critical-path DEVICE ID {totals['lock_device']:.0f})"
    )
    lines.append("=" * 96)
    lines.append(f"  {'idx':>3}  {'op_code':<48} {'device_us':>11} {'op2op_us':>11}")
    for lyr in layers:
        l_kernel = sum(o.kernel_ns for o in lyr.ops) / n / 1e3
        l_op2op = sum(o.op2op_ns for o in lyr.ops) / n / 1e3
        lines.append("-" * 96)
        lines.append(f"{lyr.label}  (device={l_kernel:,.1f} us  op2op={l_op2op:,.1f} us  ops={len(lyr.ops)})")
        for i, o in enumerate(lyr.ops):
            k_us = o.kernel_ns / n / 1e3
            g_us = o.op2op_ns / n / 1e3
            flag = "  <-region-start (op2op clipped)" if o.region_start else ""
            lines.append(f"  {i:>3}  {o.op_code:<48} {k_us:>11,.1f} {g_us:>11,.1f}{flag}")
    lines.append("=" * 96)
    lines.append(
        f"GRAND TOTAL (per chunk): device={totals['kernel_us']:,.1f} us  "
        f"op2op={totals['op2op_us']:,.1f} us  total={totals['total_us']:,.1f} us  "
        f"(op2op {totals['op2op_pct']:.1f}%)"
    )
    lines.append("=" * 96)
    return "\n".join(lines)


def perlayer_report(csv_path: str) -> str:
    layers, totals = parse_perlayer(csv_path)
    return format_perlayer(layers, totals)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: python -m models.demos.deepseek_v3_d_p.utils.perlayer_op2op <ops_perf_results.csv>")
        sys.exit(2)
    print(perlayer_report(sys.argv[1]))
