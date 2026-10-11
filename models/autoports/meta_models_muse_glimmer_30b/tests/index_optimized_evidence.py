# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Flatten retained experiment artifacts without converting failed rows to passes."""

import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "doc/optimized_decoder"
fields = [
    "artifact",
    "candidate",
    "layer",
    "batch",
    "length",
    "status",
    "prefill_hf_pcc",
    "decode_hf_pcc",
    "prefill_ms",
    "decode_ms",
    "error",
]
rows = []
for p in sorted(root.glob("*.json")):
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        continue
    for row in data.get("rows", []):
        if "candidate" in row:
            rows.append(dict(row, artifact=p.name, error=row.get("error", "").split("backtrace:")[0].strip()))
with (root / "candidate_summary.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print("Indexed candidate rows:", len(rows))
