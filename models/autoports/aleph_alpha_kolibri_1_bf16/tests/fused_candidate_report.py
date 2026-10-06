# SPDX-License-Identifier: Apache-2.0
"""Collect all candidate metrics and gate final decode against distinct correct graphs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/fused_decoder"
SELECTED_GRAPH = {"final", "profiled", "constant_mask_rm128", "row_major_padding"}


def main():
    rows = []
    for path in sorted(ROOT.glob("*/profile_*.json")):
        data = json.loads(path.read_text())
        rows.append(
            dict(
                path=str(path.relative_to(ROOT)),
                layer=data["layer"],
                tokens=data.get("prefill_tokens", 128),
                repetitions=data["repetitions"],
                host_elapsed_ms=data["host_elapsed_ms"],
                pcc=data["pcc"],
                unfused_pcc=data.get("unfused_pcc"),
                flags=data.get("fusions", ""),
                provenance=data["provenance"],
            )
        )
    (ROOT / "candidate_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    comparisons = []
    for layer in (0, 4):
        final = json.loads((ROOT / f"final/profile_{layer}.json").read_text())
        candidates = [
            row
            for row in rows
            if row["layer"] == layer
            and row["tokens"] == 128
            and row["path"].split("/")[0] not in SELECTED_GRAPH
            and min(item["pcc"] for item in row["pcc"]) >= 0.995
            and (not row["unfused_pcc"] or min(row["unfused_pcc"].values()) >= 0.995)
        ]
        baseline = json.loads((ROOT / f"baseline/reference/profile_{layer}.json").read_text())
        candidates.append(
            dict(path=f"baseline/reference/profile_{layer}.json", host_elapsed_ms=baseline["host_elapsed_ms"])
        )
        best = min(candidates, key=lambda row: row["host_elapsed_ms"]["decode"])
        assert final["host_elapsed_ms"]["decode"] < best["host_elapsed_ms"]["decode"]
        comparisons.append(
            dict(
                layer=layer,
                final_ms=final["host_elapsed_ms"]["decode"],
                best_distinct_candidate=best["path"],
                candidate_ms=best["host_elapsed_ms"]["decode"],
                considered=len(candidates),
            )
        )
    result = dict(
        status="pass",
        workload="B1, 128-token prefill, traced decode at position128; unprofiled warmed timing",
        selected_graph_exclusions=sorted(SELECTED_GRAPH),
        exclusion_reason="final, constant_mask_rm128 and row_major_padding implement the selected decode graph; profiled has profiler overhead",
        limitations="Exploratory records use20 repetitions, final and last candidate controls use100; run variation remains.",
        comparisons=comparisons,
    )
    (ROOT / "candidate_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
