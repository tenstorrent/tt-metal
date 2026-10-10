# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce custom GDN stage attribution from the retained full-model CSV."""

import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURE = HERE.parent / "compact-profile-v1"


def run():
    provenance = json.loads((CAPTURE / "capture.json").read_text())["csv"]
    chunks = []
    for part in provenance["parts"]:
        raw = (CAPTURE / part["path"]).read_bytes()
        assert len(raw) == part["bytes"] and hashlib.sha256(raw).hexdigest() == part["sha256"]
        chunks.append(raw)
    compressed = b"".join(chunks)
    assert hashlib.sha256(compressed).hexdigest() == provenance["compressed_sha256"]
    raw = gzip.decompress(compressed)
    assert len(raw) == provenance["uncompressed_bytes"]
    assert hashlib.sha256(raw).hexdigest() == provenance["uncompressed_sha256"]
    profile = json.loads((CAPTURE / "receipts/export-recovery-v2/report/profile.json").read_text())
    inventory = json.loads((CAPTURE / "receipts/operator-profile-v1/report/analysis.json").read_text())
    mapping = json.loads((HERE / "source-mapping.json").read_text())["kernels"]
    header = '#define TRISC_MATH\n#include "defines_generated.h"\nvoid kernel_main();\n'
    for item in mapping:
        generated = gzip.decompress((HERE / item["retained_generated_source"]).read_bytes())
        assert hashlib.sha256(generated).hexdigest() == item["generated_sha256"]
        assert generated.decode().startswith(header)
        inline = generated[len(header.encode()) :]
        assert hashlib.sha256(inline).hexdigest() == item["inline_sha256"] and item["exact_source_match"]
    stages = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(io.StringIO(raw.decode())):
        if row.get("OP CODE") != "GenericOpDeviceOperation":
            continue
        if row.get("METAL TRACE ID") != str(profile["model_trace_id"]):
            continue
        if row.get("METAL TRACE REPLAY SESSION ID") in ("", "-", None):
            continue
        matches = [m for m in mapping if f'/{m["kernel_hash"]}/' in row["COMPUTE KERNEL HASH"]]
        assert len(matches) == 1, row["COMPUTE KERNEL HASH"]
        key = (int(row["DEVICE ID"]), int(row["METAL TRACE REPLAY SESSION ID"]))
        stages[matches[0]["stage"]][key].append(row)
    columns = {
        "kernel": "DEVICE KERNEL DURATION [ns]",
        "reader": "DEVICE BRISC KERNEL DURATION [ns]",
        "writer": "DEVICE NCRISC KERNEL DURATION [ns]",
        "compute": "DEVICE TRISC1 KERNEL DURATION [ns]",
    }
    summary, totals = [], defaultdict(float)
    for name, groups in stages.items():
        assert len(groups) == 12 and {d for d, _ in groups} == set(profile["device_ids"])
        assert all(len(rows) == 48 for rows in groups.values())
        stats = {k: [] for k in columns}
        per_rank = []
        for (device, replay), rows in sorted(groups.items()):
            sums = {}
            for metric, column in columns.items():
                values = [float(r[column]) for r in rows]
                assert all(math.isfinite(v) and v >= 0 for v in values)
                sums[metric + "_ms"] = sum(values) / 1e6
                stats[metric].append(sums[metric + "_ms"])
            totals[device, replay] += sums["kernel_ms"]
            per_rank.append(dict(device=device, replay_session=replay, calls=len(rows), **sums))
        summary.append(
            dict(
                stage=name,
                calls_per_rank_replay=48,
                **{"median_" + metric + "_ms": statistics.median(values) for metric, values in stats.items()},
                rank_replays=per_rank,
            )
        )
    generic = next(r for r in inventory["inventory"] if r["operation"] == "GenericOpDeviceOperation")
    assert math.isclose(statistics.median(totals.values()), generic["median_kernel_ms"], abs_tol=1e-9)
    result = dict(
        state="completed",
        csv_sha256=provenance["uncompressed_sha256"],
        scope="Qualified compact B16/32K TP4 full profile; 48 GDN layers, four ranks, three replays",
        whole_step_profiler_overhead_fraction=inventory["profiler_overhead_fraction"],
        physical_dram_counters=False,
        additive_wall_time=False,
        total_generic_kernel_median_ms=statistics.median(totals.values()),
        stages=sorted(summary, key=lambda row: -row["median_kernel_ms"]),
    )
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {**result, "stages": [{k: v for k, v in r.items() if k != "rank_replays"} for r in result["stages"]]},
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
