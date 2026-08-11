#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Write ``doc/optimized_decoder/perf/accounting.json`` from the committed artifacts.

Reconciles the three numbers the `$optimize` skill requires from the same run - the
theoretical roofline, the device time from the signposted `tt-perf-report` window, and the
end-to-end wall clock - so none of them can be hand-typed.

The architecture constants are read out of `tt_perf_report`'s own `ArchitectureSpec` rather
than back-computed from percentages, so the roofline and the report's `DRAM %` / `FLOPs %`
columns are normalised against exactly the same peaks. `dram_bandwidth_gb_s` is chip-wide;
it is *not* scaled by the number of cores an op happened to use.

    python models/autoports/meta_models_muse_glimmer_30b/scripts/render_optimized_accounting.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC = MODEL_DIR / "doc" / "optimized_decoder"
OUT = DOC / "perf" / "accounting.json"

# One decoder layer's projection weights, in elements.
HIDDEN, INTERMEDIATE, HEADS, KV_HEADS, HEAD_DIM = 6656, 19968, 32, 2, 128
QKV_WIDTH = (HEADS + 2 * KV_HEADS) * HEAD_DIM
ATTN_GATE_WIDTH = HEADS * HEAD_DIM
WEIGHT_ELEMENTS = (
    HIDDEN * QKV_WIDTH  # QKV projection
    + HIDDEN * ATTN_GATE_WIDTH  # attention gate projection
    + ATTN_GATE_WIDTH * HIDDEN  # output projection
    + 3 * HIDDEN * INTERMEDIATE  # SwiGLU gate, up, down
)
# Bytes per element, including the block-float shared exponents (a 32x32 tile is 1088 B at
# BFP8_B and 576 B at BFP4_B, against 1024 and 512 B of mantissa).
BFP4_BYTES_PER_ELEMENT = 576 / 1024
BFP8_BYTES_PER_ELEMENT = 1088 / 1024
DECODE_CONTEXT = 4096
KV_ELEMENTS = 2 * KV_HEADS * HEAD_DIM * DECODE_CONTEXT  # K and V for one user

NAMED_LIMITATIONS = [
    "Decode device time is ~1.96x the chip DRAM roofline. Two counts must not be conflated: the "
    "DRAM bank count is 8 (mesh_device.dram_grid_size().x, the grid the weights are width-sharded "
    "over), while MatmulMultiCoreReuseMultiCastDRAMSharded picks its own 12-core DRAM-reader worker "
    "set via get_optimal_dram_bank_to_reader_assignment - that 12 is the Cores column of every "
    "dominant decode row. Those 12 workers sit simultaneously at ~53% of the chip DRAM bandwidth "
    "and ~53% of their own 12-core LoFi FLOP peak, so neither is saturated and the gap is "
    "read/compute overlap on a 12-core worker set. More compute cores IS expressible and loses: an "
    "explicitly configured MatmulMultiCoreReuseMultiCast1DProgramConfig candidate (16 cores, "
    "mcast_in0, L1 width-sharded in0, per-role in0_block_w, same BFP4/LoFi policy) measured "
    "1.2311 ms against 1.0644 ms, and the no-program-config interleaved baseline 1.4019 ms. 16 is "
    "that family's ceiling because num_cores must divide both the tiled K and the tiled N and "
    "gcd(208,144)=16, with no rectangle for 26/52/104 inside an 11x10 grid. Going further needs an "
    "op-level change, not a config; this is the largest remaining decode opportunity.",
    "Decode end-to-end is ~1% above device time, i.e. the traced decode loop carries no material "
    "host term and there is nothing left to remove there.",
    "Prefill matmuls run on 64 of 110 cores. grid_x is pinned to the 8 DRAM banks because a 2D "
    "multicast matmul with DRAM width-sharded weights returns NaN for any other width "
    "(scripts/repro_prefill_matmul_grid_x9.py), and grid_y above 8 measured 48% slower. Against the "
    "reachable 64-core LoFi peak the prefill matmul roofline is ~22 ms; the roofline_ms below is "
    "the unreachable 110-core figure.",
    "Prefill end-to-end is 4-9% above device time, down from ~20% in the functional stage. What "
    "remains is per-chunk host dispatch; prefill is not traced because the chunk count depends on "
    "the prompt length.",
    "Non-matmul prefill time is ~34% of the window: SDPA ~13%, elementwise ~9%, norms ~9%. Prefill "
    "activations are DRAM interleaved by design, so the norms are not sharded.",
    "In sliding_rope prefill at 8192 tokens the second of the two identical SwiGLU projections is "
    "7-13% slower than the first and degrades across the measured iterations, while the pair is "
    "symmetric at 4096 and on the full-attention path. Read as DRAM-allocator fragmentation after "
    "the non-chunked windowed SDPA's 8192-token Q/K/V; both candidate mitigations lose. It is "
    "inside the reported numbers, not excluded from them. See README.md 'Anomalies'.",
]


def _arch_spec():
    from tt_perf_report.perf_report import ArchitectureSpec

    spec = ArchitectureSpec.from_name("blackhole", 110)
    return spec.dram_bandwidth_gb_s * 1e9, spec.tflops_per_core("LoFi") * 1e12


def _iterations() -> dict:
    records = json.loads((DOC / "perf" / "perf_summary.json").read_text())["records"]
    out = {}
    for record in records:
        key = (
            "prefill" if record["measurement"].startswith("prefill") else "decode",
            record["kind"],
            str(record.get("seq_len") or record.get("batch")),
            bool(record.get("profiled")),
        )
        out[key] = record
    return out


def main() -> int:
    dram_bw, lofi_per_core = _arch_spec()
    records = _iterations()
    device_ms = {}
    for path in sorted(DOC.glob("tracy/*/*_perf_report.csv")):
        rows = list(csv.DictReader(path.open(newline="")))
        total = sum(float(r["Device Time"]) for r in rows if (r.get("Device Time") or "").strip())
        match = re.match(r"(prefill|decode)_(\d+)_perf_report\.csv", path.name)
        key = (match.group(1), path.parent.name, match.group(2))
        device_ms[key] = total / records[key + (True,)]["iterations"] / 1000.0

    decode_bytes = WEIGHT_ELEMENTS * BFP4_BYTES_PER_ELEMENT + KV_ELEMENTS * BFP8_BYTES_PER_ELEMENT
    workloads = []
    for kind in ("sliding_rope", "full_nope"):
        workloads.append(
            {
                "name": f"traced warmed decode, batch 1, context {DECODE_CONTEXT}, {kind}",
                "bound": "DRAM",
                "bytes_moved": decode_bytes,
                "roofline_ms": decode_bytes / dram_bw * 1e3,
                "device_ms": device_ms[("decode", kind, "1")],
                "e2e_ms": records[("decode", kind, "1", False)]["wall_clock_ms_per_iter"],
            }
        )
    for kind in ("sliding_rope", "full_nope"):
        flops = 2 * 8192 * WEIGHT_ELEMENTS
        workloads.append(
            {
                "name": f"warmed prefill, 8192 tokens, {kind}",
                "bound": "FLOPs",
                "flops": flops,
                "roofline_ms": flops / (lofi_per_core * 110) * 1e3,
                "device_ms": device_ms[("prefill", kind, "8192")],
                "e2e_ms": records[("prefill", kind, "8192", False)]["wall_clock_ms_per_iter"],
            }
        )

    payload = {
        "device": {
            "arch": "blackhole",
            "board": "p300c",
            "mesh_shape": [1, 1],
            "compute_cores": 110,
            "dram_banks": 8,
            "dram_bank_source": "mesh_device.dram_grid_size().x - the grid the weights are width-sharded over",
            "dram_sharded_matmul_reader_cores": 12,
            "dram_sharded_matmul_reader_source": (
                "chosen by the op's get_optimal_dram_bank_to_reader_assignment; the Cores column of "
                "every dominant decode matmul row"
            ),
            "dram_peak_bytes_per_second": dram_bw,
            "lofi_tflops_per_core": lofi_per_core / 1e12,
            "lofi_tflops_110_cores": lofi_per_core * 110 / 1e12,
            "peak_source": (
                "tt_perf_report.perf_report.ArchitectureSpec('blackhole'): dram_bandwidth_gb_s is "
                "chip-wide and is the same constant the reports' DRAM % column divides by; "
                "tflops_per_core('LoFi') is the per-core peak its FLOPs % column scales by the op's "
                "core count"
            ),
        },
        "workloads": workloads,
        "named_limitations": NAMED_LIMITATIONS,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")
    for entry in workloads:
        print(
            f"  {entry['name'][:52]:54s} roofline {entry['roofline_ms']:7.3f}  device {entry['device_ms']:7.3f}"
            f"  e2e {entry['e2e_ms']:7.3f}  host {entry['e2e_ms'] - entry['device_ms']:6.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
