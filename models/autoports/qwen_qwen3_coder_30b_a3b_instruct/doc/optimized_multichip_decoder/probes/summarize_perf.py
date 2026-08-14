# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Derive doc/optimized_multichip_decoder/perf_summary.json from the four perf CSVs.

Speedup and efficiency are computed here rather than typed into prose, so the
published numbers cannot drift from the artifacts. Run after
``pytest tests/test_perf.py -k optimized_multichip``:

    python summarize_perf.py

Efficiency is speedup / 4 -- the fraction of a perfect 4-die scaling that the
measured latency achieves.
"""
import csv
import json
from pathlib import Path

DOC = Path(__file__).resolve().parents[1]


def read(name, key):
    with (DOC / name).open() as fh:
        return {int(r[key]): r for r in csv.DictReader(fh)}


base_pf = read("perf_baseline_1x1_prefill.csv", "seq_len")
mc_pf = read("perf_prefill.csv", "seq_len")
base_dec = read("perf_baseline_1x1_decode.csv", "context_len")
mc_dec = read("perf_decode.csv", "context_len")

prefill = []
for s in sorted(set(base_pf) & set(mc_pf)):
    b, m = float(base_pf[s]["us_per_token"]), float(mc_pf[s]["us_per_token"])
    prefill.append(
        {
            "seq_len": s,
            "single_chip_us_per_token": b,
            "multichip_us_per_token": m,
            "speedup": round(b / m, 3),
            "efficiency": round(b / m / 4, 3),
        }
    )

decode = []
for c in sorted(set(base_dec) & set(mc_dec)):
    b, m = float(base_dec[c]["median_ms"]), float(mc_dec[c]["median_ms"])
    decode.append(
        {
            "context_len": c,
            "single_chip_ms": b,
            "multichip_ms": m,
            "speedup": round(b / m, 3),
            "efficiency": round(b / m / 4, 3),
        }
    )

# Stage 03's frozen CSVs, one directory over, so the before/after ratio is also
# computed from artifacts rather than typed.
S3 = DOC.parent / "multichip_decoder"


def read3(name, key):
    with (S3 / name).open() as fh:
        return {int(r[key]): r for r in csv.DictReader(fh)}


s3_pf, s3_dec = read3("perf_prefill.csv", "seq_len"), read3("perf_decode.csv", "context_len")
stage03_pf = {s: float(r["us_per_token"]) for s, r in s3_pf.items()}
stage03_dec = {c: float(r["median_ms"]) for c, r in s3_dec.items()}
pf_gain = [
    {
        "seq_len": s,
        "stage03": stage03_pf[s],
        "stage04": float(mc_pf[s]["us_per_token"]),
        "speedup": round(stage03_pf[s] / float(mc_pf[s]["us_per_token"]), 4),
    }
    for s in sorted(set(stage03_pf) & set(mc_pf))
]
dec_gain = [
    {
        "context_len": c,
        "stage03": stage03_dec[c],
        "stage04": float(mc_dec[c]["median_ms"]),
        "speedup": round(stage03_dec[c] / float(mc_dec[c]["median_ms"]), 4),
    }
    for c in sorted(set(stage03_dec) & set(mc_dec))
]

summary = {
    "devices": 4,
    "mesh": "1x4",
    "board": "p300 x2 (ClusterType.P300_X2), Blackhole",
    "topology": "Ring, 2 ethernet links per hop, FABRIC_1D_RING",
    "scope": "one decoder layer (layer 0), batch 1",
    "baseline": "tt/optimized_decoder.py on a 1x1 mesh, same harness, same tree",
    "stage03_sources": "../multichip_decoder/perf_prefill.csv, ../multichip_decoder/perf_decode.csv (frozen)",
    "sources": {
        "single_chip_prefill": "perf_baseline_1x1_prefill.csv",
        "single_chip_decode": "perf_baseline_1x1_decode.csv",
        "multichip_prefill": "perf_prefill.csv",
        "multichip_decode": "perf_decode.csv",
    },
    "prefill": prefill,
    "decode": decode,
    "stage03_prefill_us_per_token": stage03_pf,
    "stage03_decode_ms": stage03_dec,
    "stage04_vs_stage03_prefill": pf_gain,
    "stage04_vs_stage03_decode": dec_gain,
}
out = DOC / "perf_summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(f"\nwrote {out}")
