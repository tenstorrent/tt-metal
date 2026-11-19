# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Classify existing matched resume controls without generation or device access."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--starts", nargs="+", type=int, default=[68, 2528])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        command=sys.argv,
        status="completed_existing_evidence_classification",
        new_generation=False,
        device_access=False,
        gate_modified=False,
        waiver_inferred=False,
        classification_rule="Position-wise low PCC means below the unchanged minimum PCC threshold, not below the aggregate mean threshold",
        geometry_method=(
            "For centered unit-length logit vectors A=TT fresh, B=TT resumed, C=HF fresh, D=HF resumed: "
            "norm(B-A)^2=2(1-PCC(A,B)); norm(D-C)^2=2(1-PCC(C,D)); "
            "dot(B-A,D-C)=PCC(B,D)-PCC(B,C)-PCC(A,D)+PCC(A,C). "
            "Derived from all six recorded pairwise PCC values. This is descriptive output geometry, "
            "not a causal attribution test or a gate. A shared semantic boundary need not produce identical numerical error vectors."
        ),
        source_sha256={str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        cases=[],
        remaining_discriminating_checks=[
            "Run the eventual uniform policy through the unchanged full default gate; baseline controls cannot waive changed-runtime failures.",
            "For each remaining start, retain prefix-final recurrent/conv states and suffix-start states before and after the public cast; prove exact retention and the specified cast, with exact tokens, position/page indices, valid-row counts, and convolution history. The existing start2559 retention result does not establish starts68/2528 or full-attention KV equality.",
            "Localize the first additional TT hidden/state discrepancy at the failing absolute positions, with matched fresh/resumed operation boundaries. Distinguish an already-divergent incoming state from divergence introduced by the next operator.",
            "At that first operator, replay fresh/resumed branches with identical captured hidden/state/weight/router operands. Difference on identical operands isolates arithmetic or a kernel path; a mismatch in retained state, position, or history identifies a state/transport issue requiring repair.",
            "For routing-sensitive layers, capture actual TT FP32 router logits, IDs and scores and compare same-ID/same-score expert outputs across branches. CPU routing alone is not evidence that TT selected those exact experts.",
        ],
    )
    for start in args.starts:
        source = args.control_root / f"L2560_s{start}" / "result.json"
        data = json.loads(source.read_text())
        assert data["gate_modified"] is False and data["waiver_inferred"] is False
        report["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        cutoff = data["thresholds"]["pcc_min_min"]
        tt_low, hf_low = (set(data[key]) for key in ("tt_below_min_pcc_positions", "hf_below_min_pcc_positions"))
        tt_flips, hf_flips = (set(data[key]) for key in ("tt_mismatch_positions", "hf_mismatch_positions"))
        comparisons = data["comparisons"]
        assert len(comparisons) == 6
        case = dict(
            start=start,
            scored_position_count=len(data["positions"]),
            source=str(source),
            revision=data["revision"],
            thresholds=data["thresholds"],
            exact_six_pair_aggregate_comparisons=comparisons,
            tt_only_low_pcc_positions=sorted(tt_low - hf_low),
            shared_low_pcc_positions=sorted(tt_low & hf_low),
            hf_only_low_pcc_positions=sorted(hf_low - tt_low),
            tt_only_top1_flip_positions=sorted(tt_flips - hf_flips),
            shared_top1_flip_positions=sorted(tt_flips & hf_flips),
            hf_only_top1_flip_positions=sorted(hf_flips - tt_flips),
            tt_only_aggregate_mean_failure=(
                comparisons["tt_fresh_vs_tt_resumed"]["metrics"]["pcc_mean"]
                < data["thresholds"]["pcc_mean_min"]
                <= comparisons["hf_fresh_vs_hf_resumed"]["metrics"]["pcc_mean"]
            ),
            positions=[],
        )
        relevant = tt_low | hf_low | tt_flips | hf_flips
        for row in data["per_position"]:
            position = row["position"]
            if position not in relevant:
                continue
            pcc = row["pcc"]
            assert len(pcc) == 6
            tt_norm2 = max(0.0, 2 * (1 - pcc["tt_fresh_vs_tt_resumed"]))
            hf_norm2 = max(0.0, 2 * (1 - pcc["hf_fresh_vs_hf_resumed"]))
            dot = (
                pcc["hf_resumed_vs_tt_resumed"]
                - pcc["hf_fresh_vs_tt_resumed"]
                - pcc["hf_resumed_vs_tt_fresh"]
                + pcc["hf_fresh_vs_tt_fresh"]
            )
            cosine = dot / math.sqrt(tt_norm2 * hf_norm2) if tt_norm2 * hf_norm2 else None
            category = (
                "tt_only_below_minimum"
                if position in tt_low - hf_low
                else (
                    "both_below_minimum"
                    if position in tt_low & hf_low
                    else (
                        "hf_only_below_minimum"
                        if position in hf_low - tt_low
                        else "top1_flip_without_minimum_pcc_failure"
                    )
                )
            )
            entry = dict(
                **row,
                classification=category,
                tt_minimum_pass=pcc["tt_fresh_vs_tt_resumed"] >= cutoff,
                hf_minimum_pass=pcc["hf_fresh_vs_hf_resumed"] >= cutoff,
                all_four_top1_equal=len(set(row["top1"].values())) == 1,
                centered_unit_logit_split_geometry=dict(
                    tt_delta_norm=math.sqrt(tt_norm2),
                    hf_delta_norm=math.sqrt(hf_norm2),
                    delta_cosine=cosine,
                    tt_delta_projection_coefficient_on_hf_delta=dot / hf_norm2 if hf_norm2 else None,
                    tt_delta_orthogonal_fraction=math.sqrt(max(0.0, 1 - cosine**2)) if cosine is not None else None,
                ),
            )
            if category == "tt_only_below_minimum":
                entry[
                    "attribution_limit"
                ] = "HF passes the exact same-position minimum. A general HF split failure elsewhere does not account for this TT gate failure."
            elif category == "both_below_minimum":
                entry[
                    "attribution_limit"
                ] = "Shared threshold crossing establishes HF split sensitivity at this position, not a shared cause or permission to waive additional TT error."
            case["positions"].append(entry)
        report["cases"].append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": [
                    {
                        key: case[key]
                        for key in (
                            "start",
                            "tt_only_low_pcc_positions",
                            "shared_low_pcc_positions",
                            "hf_only_low_pcc_positions",
                            "tt_only_top1_flip_positions",
                            "shared_top1_flip_positions",
                            "tt_only_aggregate_mean_failure",
                        )
                    }
                    for case in report["cases"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
