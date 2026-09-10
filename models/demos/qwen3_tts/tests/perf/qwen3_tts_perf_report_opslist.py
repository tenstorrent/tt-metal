#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Turn one ops_perf_results CSV into the full per-op list for a signpost window.

``tt-perf-report`` gives the ranked view; this gives the complete list with the
shapes, dtypes, memory configs and core counts you need to decide *what to change*
— plus three rollups: by op code, by repeated block, and by what the op is doing
(compute vs. moving data around).

    python3 qwen3_tts_perf_report_opslist.py --window decode_frame ops.csv > ops_list.md
    python3 qwen3_tts_perf_report_opslist.py --start cp_frame_start --end cp_frame_stop ops.csv

Reads the same CSV ``tt-perf-report`` reads, so both views come from one capture.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
from collections import Counter, defaultdict

SIGNPOST_RE = re.compile(r"TT_SIGNPOST:\s*(.+)")

# Ops that move or relayout data rather than compute with it. Kept in the report —
# they are usually the biggest addressable win — but rolled up separately so the
# split between real compute and layout churn is visible at a glance.
DATA_MOVEMENT = {
    "ShardedToInterleavedDeviceOperation",
    "InterleavedToShardedDeviceOperation",
    "ReshardDeviceOperation",
    "TransposeDeviceOperation",
    "PermuteDeviceOperation",
    "SliceDeviceOperation",
    "ConcatDeviceOperation",
    "CopyDeviceOperation",
    "TilizeDeviceOperation",
    "TilizeWithValPaddingDeviceOperation",
    "UntilizeDeviceOperation",
    "UntilizeCodegenDeviceOperation",
    "UntilizeWithUnpaddingDeviceOperation",
    "TypecastDeviceOperation",
    "FillPadDeviceOperation",
    "ReshapeDeviceOperation",
}
COLLECTIVE = {
    "AllGatherAsync",
    "ReduceScatterAsync",
    "AllReduceAsync",
    "AllGather",
    "ReduceScatter",
    "AllReduce",
}


def signpost_name(row) -> str | None:
    if (row.get("OP TYPE") or "").strip() == "signpost":
        return (row.get("OP CODE") or "").strip()
    m = SIGNPOST_RE.search(row.get("OP CODE") or "")
    return m.group(1).strip() if m else None


def slice_window(rows, start: str | None, end: str | None):
    """Rows strictly between the ``start`` and ``end`` signposts (inclusive of neither)."""
    if start is None and end is None:
        return [r for r in rows if signpost_name(r) is None], []
    out, seen, active = [], [], start is None
    for r in rows:
        name = signpost_name(r)
        if name is not None:
            seen.append(name)
            if name == start:
                active = True
                continue
            if name == end and active:
                active = False
            continue
        if active:
            out.append(r)
    return out, seen


def merge_devices(win):
    """Collapse a TP mesh's per-chip rows into one op sequence.

    Under TP every chip runs the same program, so the CSV holds one row per chip per
    op and a naive read counts each op N times. Merge positionally — the chips issue
    identical op-code sequences, which is asserted, though their GLOBAL CALL COUNTs
    differ — and keep the **max** device time and gap across chips: an op is done when
    the slowest chip is done. Falls back to chip 0 alone if the sequences diverge.

    Returns ``(rows, dev_ns, gap_ns, n_devices)``. The first op's gap is dropped: it
    measures the idle before the window opened, not anything inside it.
    """
    by_dev = defaultdict(list)
    for r in win:
        by_dev[(r.get("DEVICE ID") or "0").strip()].append(r)
    seqs = [by_dev[k] for k in sorted(by_dev)]
    codes = [[r["OP CODE"] for r in s] for s in seqs]
    if len(seqs) > 1 and any(c != codes[0] for c in codes[1:]):
        print(
            f"warning: chips issued different op sequences ({[len(c) for c in codes]} ops); " "reporting chip 0 only",
            file=sys.stderr,
        )
        seqs = seqs[:1]
    base = seqs[0]
    dev = [max(num(s[i], "DEVICE KERNEL DURATION [ns]") for s in seqs) for i in range(len(base))]
    gap = [max(num(s[i], "OP TO OP LATENCY [ns]") for s in seqs) for i in range(len(base))]
    if gap:
        gap[0] = 0
    return base, dev, gap, len(by_dev)


def num(row, key) -> int:
    try:
        return int(float(row.get(key) or 0))
    except ValueError:
        return 0


_DIM_RE = re.compile(r"^\s*(\d+)")


def shape(row, prefix) -> str:
    """`1x1x32x2048` from the four PAD[LOGICAL] columns.

    Cells read ``padded[logical]`` (e.g. ``32[27]``) when the two differ; the padded
    extent is what the kernel actually runs on, so that is the one kept.
    """
    dims = []
    for ax in ("W", "Z", "Y", "X"):
        m = _DIM_RE.match(row.get(f"{prefix}_{ax}_PAD[LOGICAL]") or "")
        if not m:
            return ""
        dims.append(m.group(1))
    return "x".join(dims)


def memcfg(row, prefix) -> str:
    """`L1:width` / `DRAM:il` — the layout facts, without the MemoryConfig repr."""
    raw = (row.get(f"{prefix}_MEMORY") or "").strip()
    if not raw:
        return ""
    buf = "DRAM" if "DRAM" in raw else ("L1" if "L1" in raw else "?")
    if "WIDTH_SHARDED" in raw:
        lay = "width"
    elif "HEIGHT_SHARDED" in raw:
        lay = "height"
    elif "BLOCK_SHARDED" in raw:
        lay = "block"
    else:
        lay = "il"
    return f"{buf}:{lay}"


def tensor_desc(row, prefix) -> str:
    s = shape(row, prefix)
    if not s:
        return ""
    dt = (row.get(f"{prefix}_DATATYPE") or "").replace("DataType::", "").lower()
    lo = "T" if "TILE" in (row.get(f"{prefix}_LAYOUT") or "") else "RM"
    return f"{s} {dt} {lo} {memcfg(row, prefix)}".strip()


# Wormhole peak, matching tt_perf_report.perf_report.ArchitectureSpec("wormhole") so the
# numbers are comparable with tt-perf-report's own columns.
DRAM_BW_GB_S = 288.0
TFLOPS_PER_CORE = {
    "HiFi4": 74 / 72,
    "HiFi2": 148 / 72,
    "LoFi": 262 / 72,
    # HiFi3 is absent from tt-perf-report's map, which is exactly why that tool dies with
    # "Unknown math fidelity: HiFi3" on any CodePredictor window (the CP defaults to HiFi3
    # since 98622104f8a). Fidelity costs one pass per level and HiFi2 is exactly 2x HiFi4
    # in that table, so the implied 4-pass base is 296/72 and HiFi3 (3 passes) is 296/72/3.
    "HiFi3": 296 / 72 / 3,
}
DTYPE_BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1, "BFLOAT4_B": 0.5, "UINT32": 4, "INT32": 4, "UINT16": 2}


def dtype_bytes(row, prefix) -> float:
    dt = (row.get(f"{prefix}_DATATYPE") or "").replace("DataType::", "").upper()
    return DTYPE_BYTES.get(dt, 2)


def tensor_elems(row, prefix) -> int:
    n = 1
    for ax in ("W", "Z", "Y", "X"):
        m = _DIM_RE.match(row.get(f"{prefix}_{ax}_PAD[LOGICAL]") or "")
        if not m:
            return 0
        n *= int(m.group(1))
    return n


def matmul_efficiency(row, dev_ns):
    """(dram_gb_s, dram_pct, tflops, flops_pct, cores) for one matmul row, or None.

    Mirrors tt_perf_report.analyze_matmul: DRAM traffic counts only the operands that
    actually live in DRAM plus the output if it lands there, and a DRAM-sharded matmul is
    charged 12 cores (the Wormhole DRAM bank count) rather than its worker-core count.
    """
    if row["OP CODE"] != "MatmulDeviceOperation" or dev_ns <= 0:
        return None
    fid = (row.get("MATH FIDELITY") or "").strip()
    if fid not in TFLOPS_PER_CORE:
        return None
    dur_s = dev_ns * 1e-9
    dram_bytes = 0.0
    for prefix in ("INPUT_0", "INPUT_1"):
        if "DRAM" in (row.get(f"{prefix}_MEMORY") or ""):
            dram_bytes += tensor_elems(row, prefix) * dtype_bytes(row, prefix)
    if "DRAM" in (row.get("OUTPUT_0_MEMORY") or ""):
        dram_bytes += tensor_elems(row, "OUTPUT_0") * dtype_bytes(row, "OUTPUT_0")

    def d(prefix, ax):
        m = _DIM_RE.match(row.get(f"{prefix}_{ax}_PAD[LOGICAL]") or "")
        return int(m.group(1)) if m else 0

    M, K, N = d("INPUT_0", "Y"), d("INPUT_0", "X"), d("INPUT_1", "X")
    W, Z = d("INPUT_0", "W"), d("INPUT_0", "Z")
    cores = int(float(row.get("CORE COUNT") or 0))
    if "DRAMShardedProgramConfig" in (row.get("ATTRIBUTES") or ""):
        cores = 12
    if not cores:
        return None
    flops = (M * K * N * W * Z * 2) / dur_s
    peak = TFLOPS_PER_CORE[fid] * 1e12 * cores
    gb_s = (dram_bytes / dur_s) / 1e9
    return gb_s, 100.0 * gb_s / DRAM_BW_GB_S, flops, 100.0 * flops / peak, cores


def op_class(code: str) -> str:
    if code in DATA_MOVEMENT:
        return "data movement"
    if code in COLLECTIVE or "AllGather" in code or "ReduceScatter" in code or "AllReduce" in code:
        return "collective"
    return "compute"


def block_key(rows):
    """Label each op with the repeated block it belongs to, by counting layer boundaries.

    A decoder layer starts at its input RMSNorm, so every LayerNorm whose predecessor
    was not a LayerNorm opens a new half-layer. Approximate on purpose — it is a
    grouping aid for reading the list, not a claim about the graph.
    """
    labels, layer, prev = [], 0, ""
    for r in rows:
        code = r["OP CODE"]
        if code == "LayerNormDeviceOperation" and prev != "LayerNormDeviceOperation":
            layer += 1
        labels.append(f"blk{(layer + 1) // 2:02d}")
        prev = code
    return labels


def table(headers, rows_, aligns=None):
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows_]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--window", default="", help="label for the report heading")
    ap.add_argument("--start", default="start", help="start signpost (default: start)")
    ap.add_argument("--end", default="stop", help="end signpost (default: stop)")
    ap.add_argument("--top", type=int, default=40, help="rows in the ranked tables")
    ap.add_argument("--json", default="", help="also write the window totals to this JSON path")
    args = ap.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    win, seen = slice_window(rows, args.start, args.end)
    if not win:
        print(
            f"error: no ops between signposts '{args.start}' and '{args.end}' "
            f"in {args.csv_path} (signposts present: {seen or 'none'})",
            file=sys.stderr,
        )
        return 1

    win, dev, gap, n_dev = merge_devices(win)
    total_dev, total_gap = sum(dev), sum(gap)
    blocks = block_key(win)

    title = args.window or f"{args.start} → {args.end}"
    o = [f"# Ops list — {title}\n\n"]
    o.append(
        f"`{args.csv_path}`, window `{args.start}` → `{args.end}`"
        + (f", {n_dev} chips merged (max per op)" if n_dev > 1 else "")
        + f": **{len(win)} device ops**, **{total_dev / 1e6:.3f} ms** device kernel time, "
        f"{total_gap / 1e6:.3f} ms op-to-op gap "
        f"(**{(total_dev + total_gap) / 1e6:.3f} ms** on device end to end).\n\n"
    )
    if args.json:
        import json

        pathlib.Path(args.json).write_text(
            json.dumps(
                {
                    "window": args.window,
                    "ops": len(win),
                    "device_ms": total_dev / 1e6,
                    "gap_ms": total_gap / 1e6,
                    "chips": n_dev,
                },
                indent=2,
            )
            + "\n"
        )

    # ── rollup by class ───────────────────────────────────────────────────────
    by_class = defaultdict(lambda: [0, 0])
    for r, d in zip(win, dev):
        c = by_class[op_class(r["OP CODE"])]
        c[0] += 1
        c[1] += d
    o.append("## Where the device time goes\n\n")
    o.append(
        table(
            ["class", "ops", "device ms", "% of device"],
            [
                [k, v[0], f"{v[1] / 1e6:.3f}", f"{100.0 * v[1] / total_dev:.1f} %"]
                for k, v in sorted(by_class.items(), key=lambda kv: -kv[1][1])
            ],
            ["---", "--:", "--:", "--:"],
        )
    )

    # ── rollup by op code ─────────────────────────────────────────────────────
    by_code = defaultdict(lambda: [0, 0, 0])
    for r, d in zip(win, dev):
        e = by_code[r["OP CODE"]]
        e[0] += 1
        e[1] += d
        e[2] = max(e[2], d)
    o.append("\n## By op code\n\n")
    o.append(
        table(
            ["op code", "class", "count", "device ms", "% ", "mean µs", "max µs"],
            [
                [
                    code,
                    op_class(code),
                    v[0],
                    f"{v[1] / 1e6:.3f}",
                    f"{100.0 * v[1] / total_dev:.1f}",
                    f"{v[1] / v[0] / 1e3:.1f}",
                    f"{v[2] / 1e3:.1f}",
                ]
                for code, v in sorted(by_code.items(), key=lambda kv: -kv[1][1])
            ],
            ["---", "---", "--:", "--:", "--:", "--:", "--:"],
        )
    )

    # ── rollup by repeated block ──────────────────────────────────────────────
    by_block = defaultdict(lambda: [0, 0])
    for b, d in zip(blocks, dev):
        by_block[b][0] += 1
        by_block[b][1] += d
    o.append(f"\n## By block ({len(by_block)} blocks — see `block_key`, a reading aid not a graph claim)\n\n")
    o.append(
        table(
            ["block", "ops", "device ms"],
            [[b, v[0], f"{v[1] / 1e6:.3f}"] for b, v in sorted(by_block.items())],
            ["---", "--:", "--:"],
        )
    )

    # ── matmul efficiency ─────────────────────────────────────────────────────
    mm = defaultdict(lambda: [0, 0.0, [], [], set(), set(), set()])
    for r, dv in zip(win, dev):
        eff = matmul_efficiency(r, dv)
        if eff is None:
            continue
        gb_s, dram_pct, flops, flops_pct, cores = eff

        def _d(prefix, ax):
            m = _DIM_RE.match(r.get(f"{prefix}_{ax}_PAD[LOGICAL]") or "")
            return m.group(1) if m else "?"

        key = f"{_d('INPUT_0','Y')}x{_d('INPUT_0','X')}x{_d('INPUT_1','X')}"
        e = mm[key]
        e[0] += 1
        e[1] += dv / 1e6
        e[2].append(dram_pct)
        e[3].append(flops_pct)
        e[4].add(cores)
        e[5].add((r.get("MATH FIDELITY") or "").strip())
        e[6].add(memcfg(r, "INPUT_1") + ("+ds" if "DRAMShardedProgramConfig" in (r.get("ATTRIBUTES") or "") else ""))
    if mm:
        mm_total = sum(v[1] for v in mm.values())
        o.append(f"\n## Matmul efficiency ({mm_total:.3f} ms over {sum(v[0] for v in mm.values())} calls)\n\n")
        o.append(
            "Peak is Wormhole: 288 GB/s DRAM, and per-core TFLOPs by fidelity from\n"
            "`tt_perf_report`'s wormhole spec (HiFi3 derived — see `TFLOPS_PER_CORE`). A\n"
            "DRAM-sharded matmul is charged 12 cores, the DRAM bank count, not its worker\n"
            "cores. A decode matmul at M=1 tile is weight-bandwidth bound, so **DRAM % is the\n"
            "number to drive**; FLOPs % only becomes meaningful once M is large.\n\n"
        )
        o.append(
            table(
                ["M x K x N", "n", "ms", "% mm", "cores", "fidelity", "in1", "DRAM %", "FLOPs %"],
                [
                    [
                        k,
                        v[0],
                        f"{v[1]:.3f}",
                        f"{100.0 * v[1] / mm_total:.1f}",
                        ",".join(str(c) for c in sorted(v[4])),
                        ",".join(sorted(v[5])),
                        ",".join(sorted(v[6])),
                        f"{sum(v[2]) / len(v[2]):.1f}",
                        f"{sum(v[3]) / len(v[3]):.1f}",
                    ]
                    for k, v in sorted(mm.items(), key=lambda kv: -kv[1][1])
                ],
                ["---", "--:", "--:", "--:", "--:", "---", "---", "--:", "--:"],
            )
        )

    # ── ranked individual ops ─────────────────────────────────────────────────
    order = sorted(range(len(win)), key=lambda i: -dev[i])[: args.top]
    o.append(f"\n## Top {len(order)} individual ops by device time\n\n")
    o.append(
        table(
            ["#", "op code", "device µs", "gap µs", "cores", "fidelity", "in0", "in1", "out0"],
            [
                [
                    i,
                    win[i]["OP CODE"],
                    f"{dev[i] / 1e3:.1f}",
                    f"{gap[i] / 1e3:.1f}",
                    win[i].get("CORE COUNT", ""),
                    win[i].get("MATH FIDELITY", "") or "-",
                    tensor_desc(win[i], "INPUT_0") or "-",
                    tensor_desc(win[i], "INPUT_1") or "-",
                    tensor_desc(win[i], "OUTPUT_0") or "-",
                ]
                for i in order
            ],
            ["--:", "---", "--:", "--:", "--:", "---", "---", "---", "---"],
        )
    )

    # ── the full list ─────────────────────────────────────────────────────────
    o.append(f"\n## Full ops list ({len(win)} ops, in issue order)\n\n")
    o.append(
        table(
            ["#", "block", "op code", "device µs", "gap µs", "cores", "fidelity", "in0", "in1", "out0"],
            [
                [
                    i,
                    blocks[i],
                    win[i]["OP CODE"],
                    f"{dev[i] / 1e3:.1f}",
                    f"{gap[i] / 1e3:.1f}",
                    win[i].get("CORE COUNT", ""),
                    win[i].get("MATH FIDELITY", "") or "-",
                    tensor_desc(win[i], "INPUT_0") or "-",
                    tensor_desc(win[i], "INPUT_1") or "-",
                    tensor_desc(win[i], "OUTPUT_0") or "-",
                ]
                for i in range(len(win))
            ],
            ["--:", "---", "---", "--:", "--:", "--:", "---", "---", "---", "---"],
        )
    )

    # ── adjacent-pair frequencies: what to fuse ───────────────────────────────
    pairs = Counter(
        (win[i]["OP CODE"], win[i + 1]["OP CODE"])
        for i in range(len(win) - 1)
        if op_class(win[i]["OP CODE"]) == "data movement" or op_class(win[i + 1]["OP CODE"]) == "data movement"
    )
    if pairs:
        o.append("\n## Most repeated adjacent pairs touching data movement\n\n")
        o.append("One fix per row removes every instance.\n\n")
        o.append(
            table(
                ["count", "pair"],
                [[c, f"`{a}` → `{b}`"] for (a, b), c in pairs.most_common(20)],
                ["--:", "---"],
            )
        )

    try:
        sys.stdout.write("".join(o))
    except BrokenPipeError:  # piped into head / less
        sys.stdout = None
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
