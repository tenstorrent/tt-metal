"""Attach measured decisions to reconcile.py outputs without changing its accounting or ranking."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text())


def save(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2) + "\n")


dense_measurement = load("measurements/advisor_dense_chain_exact.json")
dense_above_measurement = load("measurements/advisor_dense_chain_g110.json")
sliding_measurement = load("measurements/sliding_advice_aggregate_noop.json")

for name, mode in (
    ("reconciliation_dense_full_forced_rope.json", "dense"),
    ("reconciliation_sliding_rope_moe.json", "sliding"),
    ("reconciliation_full_no_rope_moe.json", "full"),
):
    data = load(name)
    attributable = [c["chain"] for c in data["chains"] if c.get("advisor_removes_us", 0) > 0]
    for chain in data["chains"]:
        if chain.get("advisor_removes_us", 0) <= 0:
            chain["verdict"] = "below_threshold"
            chain["decision_reason"] = "advisor-attributable value is zero"
        elif mode == "full":
            chain["verdict"] = "not_measurable"
            chain["decision_reason"] = "0.563 us total ceiling is below the 1.263 us incumbent spread"
        else:
            measurement = dense_measurement if mode == "dense" else sliding_measurement
            chain["verdict"] = "rejected"
            chain["measured_ms"] = measurement["median_ms"]
            chain["repeats_ms"] = measurement["repeats_ms"]
            chain["oracle_passed"] = None
            chain["combined_with"] = [x for x in attributable if x != chain["chain"]]
            if mode == "sliding" and not chain.get("ops"):
                chain["ops"] = ["IR-resolved boundary aggregate"]
            chain["decision_reason"] = (
                "advisor DS-down/L1 continuation candidate loses every incumbent repeat"
                if mode == "dense"
                else "aggregate is a topology no-op after IR resolves the ranked boundary pairs; fresh repeats overlap control"
            )
    for row in data.get("disagreements", []):
        if row.get("bucket") == "dram_resident" and not row.get("verdict"):
            row["verdict"] = "reported_not_screened"
            row["decision_reason"] = "advisor agrees with shipped DRAM residency or the sparse suffix is uncapturable"
    data["decision_attachment"] = {
        "script": "doc/advisor_challenger/record_decisions.py",
        "reconcile_accounting_unchanged": True,
        "dense_candidate": "advisor_dense_chain_exact",
        "dense_candidate_above_advised_core_count": {
            "candidate": "advisor_dense_chain_g110",
            "median_ms": dense_above_measurement["median_ms"],
            "repeats_ms": dense_above_measurement["repeats_ms"],
        },
        "dense_dram_sharded_attempts": {
            "verdict": "hard_failure",
            "candidates": ["advisor_dense_chain_ds_exact", "advisor_dense_chain_ds_down32"],
            "reason": "both legal down-projection geometries stalled before a timed repeat",
        },
        "sliding_candidate": "top three chains attempted together; mandatory layout constraints reject both rewrite isolates",
    }
    save(name, data)
