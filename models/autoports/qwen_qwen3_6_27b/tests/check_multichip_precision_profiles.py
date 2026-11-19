# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify lowered MLP dtypes/fidelity in the four stage-5 precision profiles.

CPU-only. Source-requested geometry is recorded by the ordinary runner; this
check independently reads actual signposted device-op metadata on all ranks.
"""

import csv
import json
from pathlib import Path
import re


def main():
    directory = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"
    observations = []
    for index in range(4):
        job = json.loads((directory / f"precision_geometry5_profile_{index}.json").read_text())[0]
        tag = f"precision_geometry5_profile_{index}_l{job['layer']}_b{job['batch']}_s{job['length']}"
        result = json.loads((directory / (tag + ".json")).read_text())
        assert result["pass"], tag
        expected_dtype = job["policy"]["mlp_dtype"].upper()
        expected_fidelity = job["policy"]["mlp_fidelity"]
        phase = None
        groups = {}
        path = directory / "perf" / tag / "ops.csv"
        with path.open() as stream:
            for row in csv.DictReader(stream):
                op = row["OP CODE"]
                if row["OP TYPE"] == "signpost":
                    phase = op if op in ("PERF_PREFILL", "PERF_DECODE") else None
                    continue
                if phase is None or "Matmul" not in op:
                    continue

                def logical(key):
                    value = row[key]
                    match = re.fullmatch(r"\d+\[(\d+)\]", value)
                    return int(match.group(1)) if match else int(value)

                k, n = logical("INPUT_1_Y_PAD[LOGICAL]"), logical("INPUT_1_X_PAD[LOGICAL]")
                role = (
                    "gate_up"
                    if k == 5120 and n in (8704, 9216)
                    else "down" if k == 4352 and n in (5120, 5632) else None
                )
                if role is None:
                    continue
                assert row["INPUT_1_DATATYPE"] == expected_dtype, (tag, role, row)
                assert row["INPUT_0_DATATYPE"] == "BFLOAT16", (tag, role, row)
                assert row["MATH FIDELITY"] == expected_fidelity, (tag, role, row)
                key = phase, role, int(row["DEVICE ID"])
                entry = groups.setdefault(
                    key,
                    dict(
                        profile=tag,
                        phase=phase,
                        role=role,
                        device_id=key[2],
                        count=0,
                        weight_dtype=row["INPUT_1_DATATYPE"],
                        activation_dtype=row["INPUT_0_DATATYPE"],
                        fidelity=row["MATH FIDELITY"],
                        weight_k=row["INPUT_1_Y_PAD[LOGICAL]"],
                        weight_n=row["INPUT_1_X_PAD[LOGICAL]"],
                        raw_ops=str(path),
                        operation=op,
                        attributes=row["ATTRIBUTES"],
                    ),
                )
                entry["count"] += 1
        assert set(groups) == {
            (phase, role, rank)
            for phase in ("PERF_PREFILL", "PERF_DECODE")
            for role in ("gate_up", "down")
            for rank in range(4)
        }, (tag, set(groups))
        observations.extend(groups.values())
    report = {
        "status": "pass",
        "profiles": 4,
        "observations": observations,
        "scope": "Actual signposted MLP input/weight dtype and fidelity on all four ranks, both phases",
        "limitations": "Precision candidate evidence; no final-default or speedup claim",
    }
    output = directory / "precision_geometry5_lowered_op_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "pass", "profiles": 4, "rows": len(observations), "output": str(output)}))


if __name__ == "__main__":
    main()
