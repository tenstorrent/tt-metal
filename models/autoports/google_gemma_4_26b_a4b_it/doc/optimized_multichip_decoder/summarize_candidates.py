# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import csv
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
rows = []
for path in sorted(root.glob("*.json")):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "timings" not in data or "4" not in data["timings"]:
        continue
    timings = data["timings"]["4"]
    pcc = data.get("pcc", [])
    rows.append(
        dict(
            candidate=path.stem,
            layer_type=data.get("layer_type"),
            length=data.get("length"),
            steps=data.get("steps"),
            prefill_host_us=median(timings["prefill_host_us"]) if timings.get("prefill_host_us") else None,
            decode_host_us=median(timings["decode_host_us"]) if timings.get("decode_host_us") else None,
            prefill_pcc=pcc[0] if pcc else None,
            min_decode_pcc=min(pcc[1:]) if len(pcc) > 1 else None,
            passed=data.get("passed"),
            runtime_sha256=data.get("runtime_sha256"),
            evidence=path.name,
        )
    )
(root / "candidate_summary.json").write_text(
    json.dumps({"timing_basis": "Host wall, warmed traced decode; not device time", "candidates": rows}, indent=2)
    + "\n"
)
with (root / "candidate_summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("Summarized", len(rows), "results")
