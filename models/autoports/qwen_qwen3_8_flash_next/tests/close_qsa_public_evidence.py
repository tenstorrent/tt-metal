# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact CPU closure of the literal-HF QSA public continuation v7 evidence."""

import argparse
import json
import math
from pathlib import Path

from .close_watcher_evidence import verify_watcher_run
from .summarize_functional_evidence import evaluate_gate
from .validate_trace_evidence import digest, require


def metrics(values):
    require(values and all(v["finite"] is True for v in values), "Missing/nonfinite metric group")
    rows = [row for v in values for row in v["per_row"]]
    result = {"cases": len(values), "rows": len(rows)}
    for key in ("pcc", "contribution_pcc"):
        result[f"minimum_aggregate_{key}"] = min(v[key] for v in values)
        result[f"minimum_row_{key}"] = min(v[key] for v in rows)
    require(
        all(math.isfinite(v) and v >= 0.995 for k, v in result.items() if k.startswith("minimum")), "PCC gate failed"
    )
    return result


def true_tree(value):
    return bool(value) and all(true_tree(v) for v in value) if isinstance(value, list) else value is True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "doc/functional_decoder/public_continuation_layer3_v7.json"
    report = json.loads(path.read_text())
    manifest = root / "doc/functional_decoder/final_gate_bindings.json"
    gates = [
        g
        for g in json.loads(manifest.read_text())["gates"]
        if g["id"] in ("layer3_public_continuation", "layer3_unseen_live_prefill")
    ]
    require(len(gates) == 2, "Missing declared public gates")
    evaluated = [evaluate_gate(root, gate) for gate in gates]
    require(all(v["status"] == "declared_checks_satisfied_review_required" for v in evaluated), "Declared gate failed")
    watcher = verify_watcher_run(
        root,
        {
            "layer": 3,
            "artifact": str(path.relative_to(root)),
            "provenance": gates[0]["provenance"],
            "required_sources": gates[0]["required_sources"],
        },
    )
    require(
        report["literal_hf_strict_passed"] is True and report["ple_backend_control_enabled"] is False,
        "Not literal strict evidence",
    )
    require(
        report["runtime_fallback_audit"] == "clean" and report["allocation_tracking"] is True,
        "Missing runtime/trace audit",
    )
    require(report["capture_count"] == 1 and report["replay_count"] == 13, "Unexpected trace coverage")
    require(
        report["capture_outputs_acknowledged"] == 2 and report["capture_output_readbacks_before_first_replay"] == 0,
        "Wrong capture output lifecycle",
    )
    for name in ("repeated_output_equal", "repeated_state_equal", "other_state_unchanged_by_decode"):
        require(true_tree(report[name]), f"Failed repeated/state evidence: {name}")
    for row in report["page_remap"]:
        require(
            all(
                row[key] is True
                for key in (
                    "same_trace_output_equal",
                    "logical_cache_equal",
                    "indexer_and_position_equal",
                    "stable_buffer_addresses",
                )
            ),
            "Page-remap evidence failed",
        )
    require(all(v["negative_control_detected"] is True for v in report["dropped_history"]), "History control failed")
    groups = {
        name: metrics(report[name])
        for name in ("fresh_prefix", "tuple_continuation", "decode", "live_other_prefill", "page_remap")
    }
    output = {
        "scope": "Closed QSA public continuation/live-unseen-input run only, not whole-stage acceptance",
        "all_declared_checks_satisfied": True,
        "stage_acceptance": "not_evaluated_review_required",
        "literal_hf_strict_passed": True,
        "artifact_sha256": digest(path),
        "gate_manifest_sha256": digest(manifest),
        "gates": [{"id": v["id"], "status": v["status"], "checks": len(v["checks"])} for v in evaluated],
        "metrics": groups,
        "capture_count": 1,
        "replay_count": 13,
        "capture_outputs_acknowledged": 2,
        "capture_output_readbacks_before_first_replay": 0,
        "repeated_output_and_state_exact": True,
        "other_state_unchanged": True,
        "page_remap_same_trace_and_state_exact": True,
        "runtime_fallback_audit": "clean",
        "allocation_tracking": True,
        "live_unseen_lengths": report["live_prefill_preparation"]["unseen_logical_lengths"],
        "tuple_positions_after_continuation": report["tuple_positions_after_continuation"],
        "dropped_history_aggregate_contribution_pcc": [v["contribution_pcc"] for v in report["dropped_history"]],
        "watcher": watcher,
        "validator_sha256": digest(Path(__file__)),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "all_declared_checks_satisfied": True,
                "metrics": groups,
                "watcher_lines": watcher["watcher_scan"]["lines"],
            }
        )
    )


if __name__ == "__main__":
    main()
