# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-op table with roofline columns from a Tracy ops CSV (perf.py::test_tracy_forward, one warmed eager
forward between the PERF_TEXT / PERF_TEXT_END signposts).

    python -m models.demos.embedding_gemma2.tests.perf.prefill_op_table <ops_perf_results.csv> \
        --bucket 2048 --out-md table.md --out-json table.json [--clock-mhz 1350] [--grid-cores 120]

Rows: one per op class, aggregated over layers (same op code, op variant, input / output shapes, dtypes,
fidelity and memory). Roofline per row (model, not measured):

    compute_us = FLOPs / (grid_cores * flops_per_cycle(fidelity) * clock)     matmul and SDPA only
    memory_us  = DRAM bytes / DRAM_GBPS + L1 bytes / (grid_cores * L1_BYTES_PER_CYCLE * clock)
    floor_us   = max(compute_us, memory_us); gap = device_us / floor_us; bound = the larger term

grid_cores is the compute grid the profile was captured on (130 on the P150a's 13 x 10, 120 on the
12 x 10 Galaxy grid, EMBEDDING_GEMMA2_GRID): --grid-cores if given, else the profiler's own record, the
CSV column AVAILABLE WORKER CORE COUNT of the forward's ops (must be one value), else compute_grid in a
device_info.json (perf.py) next to the CSV. No record and no --grid-cores is an error.

flops_per_cycle per Tensix core: 4096 LoFi, 2048 HiFi2, 1365 HiFi3, 1024 HiFi4 (one 8x16 x 16x16 matrix-
engine op per cycle, divided by the fidelity phases; tech_reports/GEMM_FLOPS/GEMM_FLOPS.md). Bytes are the
padded input and output tensors once each (a lower bound on traffic). SDPA FLOPs count the needed work:
4 * H * S * keys * D with keys = S for full layers and the |q - k| <= 512 band (min(S, 1025)) for sliding
layers (head dim 256 = sliding, 512 = full in this model). Eltwise / norm SFPU and FPU work is not modelled
(memory term only).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

DRAM_GBPS = 512.0  # P150a: 8 GDDR6 channels x 32 bit at 16 GT/s (tt-smi DDR_SPEED 16000)
L1_BYTES_PER_CYCLE = 64.0  # per core, one NoC read port; model value for L1-resident operands
FLOPS_PER_CYCLE = {"LoFi": 4096.0, "HiFi2": 2048.0, "HiFi3": 4096.0 / 3, "HiFi4": 1024.0}
DTYPE_BYTES = {
    "BFLOAT16": 2.0,
    "FLOAT32": 4.0,
    "BFLOAT8_B": 1088 / 1024,
    "BFLOAT4_B": 576 / 1024,
    "UINT32": 4.0,
    "INT32": 4.0,
    "UINT16": 2.0,
    "UINT8": 1.0,
}
SLIDING_RADIUS = 512


def _dim(v: str) -> int:
    m = re.match(r"(\d+)", v or "")
    return int(m.group(1)) if m else 0


def _shape(r, prefix):
    if not r.get(f"{prefix}_X_PAD[LOGICAL]"):
        return None
    return tuple(_dim(r[f"{prefix}_{a}_PAD[LOGICAL]"]) for a in ("W", "Z", "Y", "X"))


def _tensors(r):
    out = []
    for p in ("INPUT_0", "INPUT_1", "INPUT_2", "INPUT_3", "OUTPUT_0", "OUTPUT_1", "OUTPUT_2"):
        s = _shape(r, p)
        if s:
            out.append((p, s, r.get(f"{p}_DATATYPE", ""), r.get(f"{p}_MEMORY", "")))
    return out


def _variant(r) -> str:
    a = r.get("ATTRIBUTES", "")
    for key in ("binary_op_type", "op_type", "unary_op_type", "norm_type"):
        m = re.search(rf"'{key}': '([^']*)'", a)
        if m:
            return m.group(1).split("::")[-1]
    m = re.search(r"UnaryOpType::(\w+)", a)
    return m.group(1) if m else ""


def _fidelity(r) -> str:
    m = re.search(r"math_fidelity=(\w+)", r.get("ATTRIBUTES", ""))
    return m.group(1) if m else (r.get("MATH FIDELITY") or "")


def _nbytes(shape, dtype) -> float:
    n = 1
    for d in shape:
        n *= d
    return n * DTYPE_BYTES.get(dtype, 2.0)


def grid_cores_of(csv_path: Path, rows) -> int:
    """Compute cores of the grid the profile ran on: the profiler's per-op record, else device_info.json."""
    seen = {r.get("AVAILABLE WORKER CORE COUNT") or "" for r in rows} - {""}
    if len(seen) > 1:
        raise ValueError(f"{csv_path}: ops ran on different grids (AVAILABLE WORKER CORE COUNT {sorted(seen)})")
    if seen:
        return int(seen.pop())
    info = Path(csv_path).parent / "device_info.json"
    if info.exists():
        gx, gy = json.loads(info.read_text())["compute_grid"]
        return gx * gy
    raise ValueError(f"{csv_path}: no grid record (AVAILABLE WORKER CORE COUNT, device_info.json); pass --grid-cores")


def roofline(r, clock_hz: float, grid_cores: int) -> dict:
    code = r["OP CODE"]
    t = _tensors(r)
    if "Embeddings" in code or "Slice" in code:
        # gather / slice read only what they write: count the output twice (read + write) plus the indices
        outs = [x for x in t if x[0].startswith("OUTPUT")]
        idx = [x for x in t if x[0] == "INPUT_0"] if "Embeddings" in code else []
        moved = [(p, s, d, m) for p, s, d, m in outs] * 2 + idx
    elif "Rotary" in code and t:
        # cos / sin tables cover max_seq_len rows; the op reads only the rows of its input
        rows = t[0][1][2]
        moved = [(p, s[:2] + (min(s[2], rows),) + s[3:], d, m) for p, s, d, m in t]
    else:
        moved = t
    dram = sum(_nbytes(s, d) for _, s, d, m in moved if "DRAM" in m)
    l1 = sum(_nbytes(s, d) for _, s, d, m in moved if "L1" in m)
    mem_us = (dram / (DRAM_GBPS * 1e9) + l1 / (grid_cores * L1_BYTES_PER_CYCLE * clock_hz)) * 1e6
    flops, mnk = 0.0, ""
    fid = _fidelity(r)
    shapes = {p: s for p, s, _, _ in t}
    if "Matmul" in code and "INPUT_1" in shapes:
        a, b = shapes["INPUT_0"], shapes["INPUT_1"]
        batch = a[0] * a[1] if (b[0] * b[1] == 1) else max(a[0] * a[1], b[0] * b[1])
        M, K, N = a[2], a[3], b[3]
        flops = 2.0 * batch * M * K * N
        mnk = f"{batch}x{M}x{K}x{N}" if batch > 1 else f"{M}x{K}x{N}"
    elif "SDPA" in code and "INPUT_0" in shapes:
        q = shapes["INPUT_0"]
        B, H, S, D = q
        keys = shapes.get("INPUT_1", q)[2]
        need = min(keys, 2 * SLIDING_RADIUS + 1) if D == 256 else keys
        flops = 4.0 * B * H * S * need * D
        mnk = f"H{H} S{S} keys{need} D{D}"
    comp_us = flops / (grid_cores * FLOPS_PER_CYCLE.get(fid, 1024.0) * clock_hz) * 1e6 if flops else 0.0
    floor = max(comp_us, mem_us)
    bound = "compute" if comp_us >= mem_us and comp_us > 0 else ("DRAM" if dram >= l1 else "L1/NoC")
    return {
        "flops": flops,
        "mnk": mnk,
        "compute_us": comp_us,
        "memory_us": mem_us,
        "floor_us": floor,
        "bound_model": bound,
        "dram_bytes": dram,
        "l1_bytes": l1,
        "fidelity": fid,
    }


def load_forward(csv_path: Path, start="PERF_TEXT", end="PERF_TEXT_END"):
    rows = list(csv.DictReader(open(csv_path)))
    i0 = next(i for i, r in enumerate(rows) if r["OP CODE"] == start)
    i1 = next(i for i, r in enumerate(rows) if r["OP CODE"] == end and i > i0)
    return [r for r in rows[i0 + 1 : i1] if r["OP TYPE"] != "signpost"]


def table(csv_path: Path, bucket: int, clock_mhz: float = 1350.0, grid_cores: int | None = None) -> dict:
    clock = clock_mhz * 1e6
    rows = load_forward(csv_path)
    grid_cores = grid_cores or grid_cores_of(csv_path, rows)
    groups: OrderedDict = OrderedDict()
    kernel_ns = gap_ns = floor_us = 0.0
    first_gap_us = float(rows[0]["OP TO OP LATENCY [ns]"] or 0) / 1e3 if rows else 0.0
    gaps = []
    for i, r in enumerate(rows):
        dev = float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1e3
        # the first op's gap spans the signpost / profiler flush before the forward: reported separately
        gap = float(r["OP TO OP LATENCY [ns]"] or 0) / 1e3 if i > 0 else 0.0
        gaps.append(gap)
        kernel_ns += dev
        gap_ns += gap
        rf = roofline(r, clock, grid_cores)
        floor_us += rf["floor_us"]
        t = _tensors(r)
        sig = (
            r["OP CODE"],
            _variant(r),
            rf["fidelity"],
            tuple((p, s, d, m) for p, s, d, m in t),
        )
        g = groups.setdefault(
            sig,
            {
                "op": r["OP CODE"].replace("DeviceOperation", "").replace("Operation", ""),
                "variant": _variant(r),
                "count": 0,
                "device_us": 0.0,
                "gap_us": 0.0,
                "cores": set(),
                "mnk": rf["mnk"],
                "fidelity": rf["fidelity"],
                "dtypes": "/".join(d for _, _, d, _ in t),
                "memory": "/".join(
                    m.replace("DEV_1_", "")
                    .replace("DEV_0_", "")
                    .replace("_INTERLEAVED", "-i")
                    .replace("_SHARDED", "-s")
                    for _, _, _, m in t
                ),
                "out_shape": "x".join(str(x) for x in (t[-1][1] if t else ())),
                "compute_us": 0.0,
                "memory_us": 0.0,
                "floor_us": 0.0,
                "bound_model": rf["bound_model"],
            },
        )
        g["count"] += 1
        g["device_us"] += dev
        g["gap_us"] += gap
        g["cores"].add(int(r["CORE COUNT"] or 0))
        for k in ("compute_us", "memory_us", "floor_us"):
            g[k] += rf[k]
    out_rows = []
    for g in groups.values():
        g["cores"] = "-".join(str(c) for c in sorted({min(g["cores"]), max(g["cores"])}))
        g["share"] = g["device_us"] / kernel_ns if kernel_ns else 0.0
        g["gap_x"] = g["device_us"] / g["floor_us"] if g["floor_us"] else None
        g["bound"] = g["bound_model"] + (" (far above floor)" if g["gap_x"] and g["gap_x"] > 3 else "")
        out_rows.append(g)
    out_rows.sort(key=lambda g: -g["device_us"])
    return {
        "bucket": bucket,
        "csv": str(csv_path),
        "clock_mhz": clock_mhz,
        "grid_cores": grid_cores,
        "ops": len(rows),
        "device_kernel_us": kernel_ns,
        "op_to_op_gap_us_eager": gap_ns,
        "op_to_op_gap_us_eager_median": sorted(gaps[1:])[len(gaps[1:]) // 2] if len(gaps) > 1 else 0.0,
        "first_op_gap_us_excluded": first_gap_us,
        "floor_us": floor_us,
        "rows": out_rows,
    }


def to_markdown(t: dict, top: int | None = None) -> str:
    lines = [
        f"### b1_s{t['bucket']}: {t['ops']} ops, device kernel sum {t['device_kernel_us'] / 1e3:.3f} ms, "
        f"eager op-to-op gaps {t['op_to_op_gap_us_eager'] / 1e3:.3f} ms, roofline floor {t['floor_us'] / 1e3:.3f} ms "
        f"(clock {t['clock_mhz']:.0f} MHz, {t['grid_cores']} cores)",
        "",
        "| op | variant | count | device us | share | cores | M x K x N / SDPA | dtypes (in.../out) | fidelity | memory (in.../out) | compute us | memory us | floor us | device / floor | bound |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in t["rows"][: top or None]:
        gx = f"{g['gap_x']:.1f}" if g["gap_x"] else ""
        lines.append(
            f"| {g['op']} | {g['variant']} | {g['count']} | {g['device_us']:.1f} | {100 * g['share']:.1f}% | {g['cores']} | "
            f"{g['mnk']} | {g['dtypes']} | {g['fidelity']} | {g['memory']} | {g['compute_us']:.1f} | "
            f"{g['memory_us']:.1f} | {g['floor_us']:.1f} | {gx} | {g['bound']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--bucket", type=int, required=True)
    ap.add_argument("--clock-mhz", type=float, default=1350.0)
    ap.add_argument("--grid-cores", type=int, help="compute cores of the profiled grid (default: from the CSV)")
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--out-json", type=Path)
    a = ap.parse_args()
    t = table(a.csv, a.bucket, a.clock_mhz, a.grid_cores)
    md = to_markdown(t)
    if a.out_md:
        a.out_md.write_text(md)
    if a.out_json:
        a.out_json.write_text(json.dumps(t, indent=1) + "\n")
    print(md)


if __name__ == "__main__":
    main()
