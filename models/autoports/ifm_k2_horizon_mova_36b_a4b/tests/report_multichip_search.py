# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Link final dominant matmul costs to measured whole-layer candidate searches."""

import csv
import json
import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"
ROLES = {
    0: ["attention", "output", "dense_gate_up", "dense_down"],
    3: ["attention", "value_experts", "output", "shared_gate_up", "shared_down", "expert_gate_up", "expert_down"],
}
PATTERNS = {
    "attention": "attn|attention|multicore|split|precision",
    "output": "output|multicore|fused|agmm|topology|precision",
    "dense_gate_up": "up_|gate_up|multicore|split|swiglu|dense|packed|precision",
    "dense_down": "down|multicore|dense|packed|precision",
    "shared_gate_up": "up_|shared|multicore|split|precision",
    "shared_down": "down|shared|multicore|precision",
    "value_experts": "value|sparse|precision|split",
    "expert_gate_up": "expert|gate_up|sparse|split|precision",
    "expert_down": "expert|down|sparse|split|precision",
}


def write_csv(name, rows):
    with (DOC / name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    candidates = list(csv.DictReader((DOC / "candidate_measurements.csv").open()))
    costs, searches = [], []
    for layer, kind in ((0, "dense"), (3, "mova_moe")):
        rows = list(csv.DictReader((DOC / "tracy/accepted" / kind / "decode_ops.csv").open()))
        start = next(i for i, r in enumerate(rows) if r["OP CODE"] == "PERF_DECODE")
        end = next(i for i, r in enumerate(rows) if r["OP CODE"] == "PERF_DECODE_END")
        rank = [r for r in rows[start + 1 : end] if r["DEVICE ID"] == "0"]
        kernel_sum = sum(float(r["DEVICE KERNEL DURATION [ns]"] or 0) for r in rank)
        matmuls = [r for r in rank if "matmul" in r["OP CODE"].lower()]
        assert len(matmuls) == len(ROLES[layer])
        for role, op in zip(ROLES[layer], matmuls):
            share = 100 * float(op["DEVICE KERNEL DURATION [ns]"]) / kernel_sum
            costs.append(
                dict(
                    layer=layer,
                    role=role,
                    rank=0,
                    op=op["OP CODE"],
                    kernel_us=float(op["DEVICE KERNEL DURATION [ns]"]) / 1000,
                    percent_rank0_decode_kernel_sum=share,
                    fidelity=op["MATH FIDELITY"],
                    attributes=op["ATTRIBUTES"],
                    weight_shape="x".join(op[f"INPUT_1_{axis}_PAD[LOGICAL]"] for axis in "WZYX"),
                )
            )
            matching = [
                r
                for r in candidates
                if int(r["layer"]) == layer
                and r["baseline"] == "False"
                and int(r["seq"]) <= 33
                and not int(r["profile_samples"])
                and re.search(PATTERNS[role], r["artifact"])
            ]
            assert matching, role
            for candidate in matching:
                searches.append(
                    dict(
                        layer=layer,
                        role=role,
                        percent_rank0_decode_kernel_sum=share,
                        artifact=candidate["artifact"],
                        configuration_options=candidate["options"],
                        whole_layer_traced_decode_ms=candidate["decode_ms"],
                        prefill_pcc=candidate["prefill_pcc"],
                        decode_pcc=candidate["decode_pcc"],
                        source_sha256=candidate["source_sha256"],
                        decision_reference="advice_ledger.md; candidate source snapshots give effective inherited configuration",
                    )
                )
    write_csv("dominant_matmul_costs.csv", costs)
    write_csv("matmul_search_index.csv", searches)
    (DOC / "matmul_search_index.json").write_text(
        json.dumps(
            dict(
                cost_basis="Rank0 kernel-duration accounting within one warmed decode profile. Excludes host/device gaps and is not cross-rank critical-path utilization.",
                scope="Search index links successful measured variants by affected projection family; combined variants affect multiple roles. Failed/adapted attempts are in execution logs and advice_ledger.md.",
                final_roles=len(costs),
                candidate_role_links=len(searches),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
