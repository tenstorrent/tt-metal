# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact reproducible candidate inventory; failed jobs remain visible."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/optimized_decoder"
rows = []
for f in sorted(ROOT.glob("*.json")):
    j = json.loads(f.read_text())
    if not isinstance(j, dict) or "records" not in j:
        continue
    bs = [x for x in j["records"] if x.get("benchmark")]
    if not bs:
        continue
    b = bs[-1]
    scores = [v for r in j["records"] for k, v in r.items() if "pcc" in k]
    rows.append(
        dict(
            candidate=f.stem,
            layer=j["layer"],
            length=j["lengths"][-1],
            prefill_ms=b["prefill_ms"],
            decode_ms=b["decode_ms"],
            min_pcc=min(scores),
            gate_pass=min(scores) >= 0.995 and not j.get("diagnostic_only", False),
            policy=json.dumps(j.get("policy", {}), sort_keys=True),
            source_sha256=j["implementation_sha256"],
            artifact=f.name,
        )
    )
if rows:
    with (ROOT / "candidate_measurements.csv").open("w") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for layer in (0, 3):
        best = sorted(
            (r for r in rows if r["layer"] == layer and r["gate_pass"] and r["length"] == 1024),
            key=lambda r: r["decode_ms"],
        )[:8]
        print("Layer", layer)
        for row in best:
            print(row["candidate"], round(row["prefill_ms"], 4), round(row["decode_ms"], 5), round(row["min_pcc"], 6))
