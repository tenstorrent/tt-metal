"""Verify measured matmul policies and DRAM-reader counts from raw Tracy rows."""

import csv
import hashlib
import json
import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc/multichip_decoder"


def collect(tag, expected_readers=None):
    directory = DOC / "tracy" / tag
    source = directory / "ops.csv"
    provenance = json.loads((directory / "provenance.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance["csv_sha256"]
    rows = list(csv.DictReader(source.open()))
    policies = {}
    for phase in ("prefill", "decode"):
        start = next(i for i, r in enumerate(rows) if r["OP CODE"] == f"PERF_{phase.upper()}")
        end = next(i for i, r in enumerate(rows) if r["OP CODE"] == f"PERF_{phase.upper()}_END")
        found = {}
        for row in rows[start + 1 : end]:
            if "Matmul" not in row["OP CODE"] and "Expert" not in row["OP CODE"]:
                continue
            attributes = row["ATTRIBUTES"]
            reader = re.search(r"num_workers_per_dram_bank=(\d+)", attributes)
            entry = dict(
                operation=row["OP CODE"],
                input_shape=[row.get(f"INPUT_0_{a}_PAD[LOGICAL]") for a in "WZYX"],
                weight_shape=[row.get(f"INPUT_1_{a}_PAD[LOGICAL]") for a in "WZYX"],
                input_dtype=row["INPUT_0_DATATYPE"],
                weight_dtype=row["INPUT_1_DATATYPE"],
                output_dtype=row["OUTPUT_0_DATATYPE"],
                fidelity=row["MATH FIDELITY"],
                input_memory=re.sub(r"DEV_\d+_", "", row["INPUT_0_MEMORY"]),
                weight_memory=re.sub(r"DEV_\d+_", "", row["INPUT_1_MEMORY"]),
                readers=int(reader.group(1)) if reader else None,
                attributes=attributes,
            )
            found[json.dumps(entry, sort_keys=True)] = entry
        policies[phase] = list(found.values())
    if expected_readers is not None:
        actual = [r for r in policies["decode"] if r["readers"] is not None]
        assert actual and all(r["readers"] == expected_readers for r in actual), tag
        shapes = {tuple(r["weight_shape"][-2:]) for r in actual}
        if tag.startswith("column_proof"):
            assert shapes == {("6144[6144]", "640[640]")}, shapes
        elif tag.startswith(("dram_qsa", "selected_v5_qsa_proof")):
            assert shapes == {("2560[2560]", "4224[4224]")}, shapes
        else:
            assert shapes == {
                ("2560[2560]", "3200[3200]"),
                ("2560[2560]", "352[352]"),
                ("320[320]", "2560[2560]"),
                ("2560[2560]", "4160[4160]"),
                ("2560[2560]", "544[544]"),
                ("2560[2560]", "320[320]"),
                ("160[160]", "2560[2560]"),
            }, shapes
    else:
        wo = [r for r in policies["decode"] if r["weight_shape"][-2:] == ["6144[6144]", "640[640]"]]
        assert wo and all(r["input_dtype"] == r["weight_dtype"] == r["output_dtype"] == "BFLOAT8_B" for r in wo)
        assert all(r["fidelity"] == "LoFi" and "in0_block_w=48" in r["attributes"] for r in wo)
        hc = [r for r in policies["decode"] if r["weight_shape"][-2:] == ["2560[2560]", "352[352]"]]
        assert hc and all(r["output_dtype"] == "BFLOAT16" for r in hc)
        hc_prefill = [r for r in policies["prefill"] if r["weight_shape"][-2:] == ["2560[2560]", "352[352]"]]
        assert hc_prefill and all(r["output_dtype"] == "FLOAT32" for r in hc_prefill)
    if tag.startswith(("selected_v5", "selected_v6")):
        for phase in ("prefill", "decode"):
            qsa = [r for r in policies[phase] if r["weight_shape"][-2:] == ["2560[2560]", "4224[4224]"]]
            dtype, fidelity = (
                ("BFLOAT8_B", "HiFi2")
                if tag.startswith("selected_v6") and phase == "prefill"
                else ("BFLOAT4_B", "LoFi")
            )
            assert qsa and all(r["weight_dtype"] == dtype and r["fidelity"] == fidelity for r in qsa), (tag, phase)
    prefix, layer = tag.rsplit("_mesh_l", 1)
    timing_path = f"{prefix}_profile_mesh_l{layer}/timing.json"
    timing = json.loads((DOC / timing_path).read_text())
    return dict(
        tag=tag,
        timing=timing_path,
        source_sha256=timing["source_sha256"],
        raw_csv_sha256=provenance["csv_sha256"],
        provenance=str((directory / "provenance.json").relative_to(DOC)),
        policies=policies,
    )


def main():
    runs = [collect(f"selected_v4_mesh_l{layer}") for layer in (0, 1)]
    runs += [collect("selected_v6_mesh_l3")]
    runs += [collect(f"dram_proof_r{readers}_mesh_l1", readers) for readers in (1, 2, 3)]
    runs += [collect(f"selected_v5_qsa_proof_r{readers}_mesh_l3", readers) for readers in (1, 2, 3)]
    runs += [collect(f"column_proof_r{readers}_mesh_l0", readers) for readers in (1, 2, 3)]
    (DOC / "profile_policy_audit.json").write_text(json.dumps(dict(status="passed", runs=runs), indent=2) + "\n")
    lines = [
        "# Actual measured matmul policies",
        "",
        "Derived from raw Tracy rows within validated signpost windows; raw CSV hashes are checked against provenance. The selected_v4 GDN captures remain valid under the source_reuse.json equivalence proof; selected_v6 QSA captures use final-source BF8/HiFi2 prefill and converted BF4/LoFi decode. The selected_v5 QSA reader proofs retain identical BF4/LoFi decode shapes and configurations; only stored quantized values differ. The isolated reader-proof captures precede the HC decode-only BF16 refinement; the new precision-matched HC reader timings are selected_v4_hc_r1/r2/r3_l0. All other reader roles retain the same measured precision/configuration. Exact source hashes are recorded in the JSON audit.",
        "",
        "| Capture | Phase | Weight K×N (padded[logical]) | Input / weight / output dtype | Fidelity | DRAM readers |",
        "|---|---|---|---|---|---:|",
    ]
    for run in runs:
        for phase, policies in run["policies"].items():
            for row in policies:
                lines.append(
                    f"| [{run['tag']}]({run['provenance']}) | {phase} | {' × '.join(row['weight_shape'][-2:])} | {row['input_dtype']} / {row['weight_dtype']} / {row['output_dtype']} | {row['fidelity']} | {row['readers'] or 'interleaved/native'} |"
                )
    (DOC / "profile_policy_audit.md").write_text("\n".join(lines) + "\n")
    print("PROFILE_POLICY_AUDIT_PASS", len(runs))


if __name__ == "__main__":
    main()
