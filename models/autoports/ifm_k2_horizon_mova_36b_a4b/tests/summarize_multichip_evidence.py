# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only compact indexes for measured candidates and accepted hardware gates."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    candidates = []
    for path in sorted(DOC.glob("*.json")):
        d = json.loads(path.read_text())
        if not isinstance(d, dict) or "decode_ms" not in d or "prefill_ms" not in d:
            continue
        candidates.append(
            dict(
                artifact=path.name,
                layer=d["layer"],
                batch=d["batch"],
                seq=d["seq"],
                baseline=d["baseline"],
                decode_ms=d["decode_ms"],
                prefill_ms=d["prefill_ms"],
                prefill_pcc=d.get("pcc", {}).get("prefill"),
                decode_pcc=d.get("pcc", {}).get("decode"),
                tracking=d["tracking"],
                iterations=d.get("iterations"),
                prefill_iterations=d.get("prefill_iterations"),
                source_sha256=d["source_sha256"],
                options=json.dumps(d["options"], sort_keys=True),
            )
        )
    write_csv(DOC / "candidate_measurements.csv", candidates)
    performance = []
    for seq in (32, 256, 2048, 2049):
        for layer in (0, 3):
            base = DOC / f"final_baseline_l{layer}_b1_s{seq}.json"
            multi = DOC / f"final_wall_l{layer}_b1_s{seq}.json"
            if not base.exists() or not multi.exists():
                continue
            b, m = json.loads(base.read_text()), json.loads(multi.read_text())
            assert b["context"] == m["context"] and b["tracking"] == m["tracking"] == "0"
            for phase in ("prefill", "decode"):
                speedup = b[phase + "_ms"] / m[phase + "_ms"]
                performance.append(
                    dict(
                        layer=layer,
                        seq=seq,
                        context=m["context"],
                        phase=phase,
                        baseline_ms=b[phase + "_ms"],
                        multichip_ms=m[phase + "_ms"],
                        speedup=speedup,
                        efficiency=speedup / 4,
                        baseline_artifact=base.name,
                        multichip_artifact=multi.name,
                    )
                )
    if performance:
        write_csv(DOC / "performance.csv", performance)
    validations = []
    patterns = [
        "final_boundaries_l0*.json",
        "final_contract_l0*.json",
        "rr_boundaries_l3*.json",
        "rr_contract_l3*.json",
        "watcher_os_fresh_l0*.json",
        "watcher_os_rr_l3*.json",
    ]
    for pattern in patterns:
        for path in sorted(DOC.glob(pattern)):
            d = json.loads(path.read_text())
            if "cases" not in d:
                continue
            validations.append(
                dict(
                    artifact=path.name,
                    source_sha256=d["source_sha256"],
                    cases=len(d["cases"]),
                    min_prefill_pcc=min(r["prefill_pcc"] for r in d["cases"]),
                    min_decode_pcc=min(r["decode_pcc"] for r in d["cases"]),
                    trace_requests=len(d["replays"]),
                    replays_per_request=3,
                    trace_replays=3 * len(d["replays"]),
                    stress_replays=d["stress_replays"],
                    tracking=d["tracking"],
                    skip_program_cache=d["skip_program_cache"],
                    cache_ownership=d["cache_ownership"],
                    capture_count=d["capture_count"],
                )
            )
    result = dict(
        source_sha256=hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
        validations=validations,
        dense_source_equivalence=[
            "dense_evidence_equivalence.json",
            "expert_placement_equivalence.json",
            "commit_source_equivalence.json",
        ],
        full_prefill=["full_prefill_tp4_layer0.json", "full_prefill_tp4_layer3.json"],
        branch_regression="local_expert_reduce_regression.json",
        stack=["rr_stack_l0_tp4.json", "rr_stack_l2_tp4.json"],
        note="Historical candidate files are not acceptance gates; pre-axis1 batched/one-tile expert results are superseded.",
    )
    (DOC / "validation_index.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "Indexed",
        len(candidates),
        "timings,",
        len(performance),
        "performance rows,",
        len(validations),
        "validation artifacts",
    )


if __name__ == "__main__":
    main()
