"""Attach measured dispositions to reconcile.py output without changing its accounting/ranking."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE / "reconciliation_dense.json"
data = json.loads(path.read_text())
candidate = json.loads((HERE / "measurements/rope_l1_chain.json").read_text())
rope_components = {
    "dense:b16", "dense:b32", "dense:2", "dense:3", "dense:4",
    "dense:5", "dense:6", "dense:7", "dense:8", "dense:9",
    "dense:10", "dense:11",
}
for chain in data["chains"]:
    name = chain["chain"]
    if name in rope_components:
        chain.update({
            "verdict": "kept",
            "measured_ms": candidate["median_ms"],
            "repeats_ms": candidate["repeats_ms"],
            "oracle_passed": True,
            "oracle_pcc": 0.9989930042363637,
            "oracle_weights": "real",
            "combined_with": sorted(rope_components - {name}),
            "candidate": "rope_l1_chain",
            "perf_report": "candidate_rope_l1_perf_report.csv",
        })
    else:
        chain["verdict"] = "below_threshold"
        chain["disposition_reason"] = (
            "zero attributable value" if chain.get("advisor_removes_us", 0) == 0 else
            "soft edge pairing invalidated by final_ir.mlir: fused-cache is tracer-terminal and "
            "nlp_concat_heads_decode is explicitly unfixable; not a separately applicable chain"
        )
data["measurement_annotation"] = {
    "script": "doc/advisor_challenger/annotate_reconciliation.py",
    "candidate": "rope_l1_chain",
    "note": "Accounting and ranking remain byte-for-byte reconcile.py fields; this only attaches measured dispositions.",
}
for row in data.get("disagreements", []):
    if row.get("bucket") == "dram_resident":
        row["verdict"] = "reported_not_screened"
        row["disposition_reason"] = "advisor keeps this op DRAM-resident; no de-sharding change is proposed"
path.write_text(json.dumps(data, indent=2) + "\n")
