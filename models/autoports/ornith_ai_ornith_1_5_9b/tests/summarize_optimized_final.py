# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce final paired metrics and conservative bandwidth accounting."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "doc/optimized_decoder"


def read(name):
    return json.loads((ART / name).read_text())


def bench(name, key="benchmark"):
    return next(r for r in read(name)["records"] if r.get(key))


def csvwrite(name, rows):
    with (ART / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


rows, accuracy, accounting, geometry = [], [], {}, []
for kind in ("linear", "full"):
    for length in (128, 1023, 1024, 1025, 8192):
        for mode in ("baseline", "default"):
            file = f"{mode}_{kind}_s{length}.json"
            d = read(file)
            b = bench(file)
            rows.append(
                dict(
                    kind=kind,
                    mode=mode,
                    length=length,
                    chunk_size=1024,
                    chunks=(length + 1023) // 1024,
                    final_logical_chunk=(length - 1) % 1024 + 1,
                    prefill_ms=b["prefill_ms"],
                    decode_ms=b["decode_ms"],
                    artifact=file,
                )
            )
    for suffix in ("final", "watcher", "paired", "trajectory128", "long"):
        file = f"{kind}_{suffix}.json"
        d = read(file)
        scores = [v for r in d["records"] for k, v in r.items() if k in ("pcc", "contribution_pcc", "traced_pcc")]
        accuracy.append(dict(kind=kind, suite=suffix, min_pcc=min(scores), records=len(d["records"]), artifact=file))
    csvrows = list(csv.DictReader((ART / f"tracy_final/{kind}/decode_perf_report.csv").open()))
    ops = [r for r in csvrows if r["ID"].isdigit()]
    device = sum(float(r["Device Time [μs]"] or 0) for r in ops) / 1000
    gaps = sum(float(r["Op-to-Op Gap [μs]"] or 0) for r in ops) / 1000
    host = next(
        r["traced_host_wall_ms"] for r in read(f"{kind}_profile_final.json")["records"] if "traced_host_wall_ms" in r
    )
    weight_bytes = (438304768 if kind == "linear" else 419430400) * 17 // 32
    state_bytes = 2 * 32 * 128 * 128 * 4 if kind == "linear" else 2 * 4 * 1056 * 256 * 17 // 16
    floor = (weight_bytes + state_bytes) / 512e6
    b = bench(f"default_{kind}_s1024.json")
    batch = {
        mode: bench(f"{mode}_{kind}_batch32.json", "batch_benchmark")["decode_batch_ms"]
        for mode in ("baseline", "default")
    }
    accounting[kind] = dict(
        active_dram_projection_bytes=weight_bytes,
        state_or_kv_bytes=state_bytes,
        bandwidth_bytes_per_second=512e9,
        roofline_ms=floor,
        warmed_decode_ms=b["decode_ms"],
        roofline_fraction=floor / b["decode_ms"],
        profile_kernel_ms=device,
        profile_gaps_ms=gaps,
        profile_host_ms=host,
        profile_host_minus_kernels_and_gaps_ms=host - device - gaps,
        batch32_decode_ms=batch,
        batch32_roofline_fraction=(weight_bytes + (state_bytes if kind == "linear" else 0))
        / 512e6
        * 32
        / batch["default"],
        decode_ops=len(ops),
        profile_artifact=f"tracy_final/{kind}/decode_perf_report.csv",
    )
for file in sorted(ART.glob("roles_bfp*_l*.json")):
    if ".provenance" in file.name:
        continue
    for r in read(file.name):
        geometry.append(
            dict(source=file.name, **{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
        )
if geometry:
    keys = list(dict.fromkeys(k for r in geometry for k in r))
    with (ART / "matmul_geometry.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(geometry)
csvwrite("final_performance.csv", rows)
csvwrite("final_accuracy.csv", accuracy)
(ART / "perf_summary.json").write_text(
    json.dumps(
        dict(
            implementation_sha256=hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
            workload="Single-device decoder: real checkpoint, seeded BF16 hidden inputs, S1024 and ragged batch32.",
            roofline="Stored BFP8 tiles include 64 exponent bytes per 1024 values and DRAM shard padding. "
            "Linear includes FP32 recurrent read+write. Full includes K/V reads through padded position 1024. "
            "Ignores other activations, norms, convolution history, writes and metadata; conservative lower bound.",
            timing="Unprofiled medians: seven groups of 50 nonblocking trace replays; state restore outside timing. "
            "Profile host/kernel/gaps from the same single-replay synchronized signpost. "
            "Do not subtract instrumented kernel time from unprofiled grouped timing.",
            limitations=[
                "Decoder only; no model-level TTFT, token sampling or language-quality claim.",
                "Batch32 executes independent row states serially inside one trace; no dense batch speedup claimed.",
                "Profile header FLOPs/core utilization miscounts DRAM reader pipelines; raw durations used.",
            ],
            layers=accounting,
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(accounting, indent=2))
