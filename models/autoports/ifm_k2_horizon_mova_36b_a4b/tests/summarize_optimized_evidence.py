# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Derive compact comparison tables from measured runner artifacts."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/optimized_decoder"
rows = []
for path in sorted((ROOT / "candidates").glob("*_layer*.json")):
    d = json.loads(path.read_text())
    if "traced_warmed_decode_wall_ms" not in d:
        continue
    rows.append(
        dict(
            candidate=path.stem,
            layer=d["layer"],
            batch=d["batch"],
            seq=d["prefill_length"],
            decode_position=d["decode_position"],
            activations=d.get("activation_source", "random-real-weights"),
            input_variant=d.get("input_variant", 0),
            prefill_pcc=d["prefill_pcc"],
            decode_pcc=d["traced_decode_pcc"],
            prefill_ms=d["warmed_prefill_wall_ms"],
            traced_decode_ms=d["traced_warmed_decode_wall_ms"],
            iterations=d["measured_decode_replays"],
            policy=json.dumps(d.get("effective_options", d.get("options", {})), sort_keys=True),
            source_sha256=d["source_sha256"],
            artifact=str(path.relative_to(ROOT)),
        )
    )
if rows:
    with (ROOT / "candidate_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "candidate_comparison.json").write_text(json.dumps(rows, indent=2))
for kind, layer in [("dense", 0), ("sparse", 3)]:
    paths = list((ROOT / "profile_raw" / f"micro_{kind}" / "reports").rglob("ops_perf_results*.csv"))
    if not paths:
        continue
    path = sorted(paths)[-1]
    with path.open() as f:
        raw = list(csv.DictReader(f))
    selected = []
    marker = None
    for row in raw:
        if row["OP TYPE"] == "signpost" and row["OP CODE"].startswith("MICRO_"):
            marker = None if row["OP CODE"].endswith("_END") else row["OP CODE"]
        elif marker and row["OP TYPE"] == "tt_dnn_device":
            selected.append({**row, "SIGNPOST": marker})
        elif marker and row["DEVICE KERNEL DURATION [ns]"]:
            selected.append({**row, "SIGNPOST": marker})
    if selected:
        with (ROOT / f"dram_microbench_device_layer{layer}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        summary = [
            dict(
                signpost=r["SIGNPOST"],
                op=r["OP CODE"],
                device_us=float(r["DEVICE KERNEL DURATION [ns]"]) / 1000,
                fidelity=r["MATH FIDELITY"],
                cores=r["CORE COUNT"],
                attributes=r["ATTRIBUTES"],
                dtype_fields={k: v for k, v in r.items() if "DATATYPE" in k and v},
            )
            for r in selected
        ]
        (ROOT / f"dram_microbench_device_layer{layer}.json").write_text(
            json.dumps(dict(source=str(path), rows=summary), indent=2)
        )
print("SUMMARY_ROWS", len(rows))
