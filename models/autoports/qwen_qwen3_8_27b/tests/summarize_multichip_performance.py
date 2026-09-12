# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize comparable final timings and per-device tt-perf-report CSV tables."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"
source_hash = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()
performance = []
profiler = []
for layer, kind in [(0, "Linear attention"), (3, "Full attention")]:
    name = f"final_benchmark_l{layer}"
    baseline = json.loads((DOC / (name + "_baseline.json")).read_text())
    multi = json.loads((DOC / (name + ".json")).read_text())
    assert multi["source_sha256"] == source_hash
    for phase in ("prefill", "decode"):
        base_ms, multi_ms = baseline[phase + "_ms"], multi[phase + "_ms"]
        performance.append(
            dict(
                layer_kind=kind,
                phase=phase,
                baseline_ms=base_ms,
                multichip_ms=multi_ms,
                speedup=base_ms / multi_ms,
                efficiency=base_ms / multi_ms / 4,
                baseline_artifact=name + "_baseline.json",
                multichip_artifact=name + ".json",
                source_sha256=source_hash,
            )
        )
    for is_baseline in (True, False):
        profile = f"final_profile_l{layer}" + ("_baseline" if is_baseline else "")
        profile_run = json.loads((DOC / (profile + ".json")).read_text())
        assert profile_run["source_sha256"] == (baseline["source_sha256"] if is_baseline else source_hash)
        folder = DOC / "tracy" / profile
        source = max(folder.glob("reports/*/ops_perf_results*.csv*"), key=lambda p: p.parent.name)
        for phase in ("prefill", "decode"):
            for device in [3] if is_baseline else range(4):
                table = folder / f"device{device}_{phase}_perf_report.csv"
                with table.open() as f:
                    rows = list(csv.DictReader(f))

                def number(value):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return 0

                times = {
                    key: sum(number(row["Device Time"]) for row in rows if test(row["OP Code"]))
                    for key, test in {
                        "matmul_us": lambda s: "Matmul" in s,
                        "ccl_us": lambda s: any(t in s for t in ("ReduceScatter", "AllGather", "AllReduce")),
                        "movement_us": lambda s: any(
                            t in s
                            for t in (
                                "Sharded",
                                "Reshard",
                                "Slice",
                                "Concat",
                                "Copy",
                                "FillPad",
                                "Reshape",
                                "Transpose",
                                "Untilize",
                                "Typecast",
                            )
                        ),
                    }.items()
                }
                profiler.append(
                    dict(
                        profile=profile,
                        phase=phase,
                        device=device,
                        source=str(source.relative_to(DOC)),
                        table=str(table.relative_to(DOC)),
                        kernel_us=sum(number(row["Device Time"]) for row in rows),
                        gap_us=sum(number(row["Op-to-Op Gap"]) for row in rows),
                        **times,
                    )
                )
for filename, value in [("performance_summary.json", performance), ("profiler_summary.json", profiler)]:
    (DOC / filename).write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(performance, indent=2))
