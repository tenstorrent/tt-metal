"""Check final measured precision, geometry, native arguments and cache format."""

import csv
import json
from pathlib import Path
from models.autoports.qwen_qwen3_8_flash_next.tests import audit_multichip_profiles as audit

DOC = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"


def main():
    audit.DOC = DOC
    runs = []
    for layer in (0, 1, 3):
        tag = f"s5_final4_mesh_l{layer}"
        run = audit.collect(tag)
        for phase, policies in run["policies"].items():
            width = 4224 if layer == 3 else 4160
            rows = [r for r in policies if r["weight_shape"][-2:] == ["2560[2560]", f"{width}[{width}]"]]
            assert rows, (tag, phase)
            expected = (
                ("BFLOAT4_B", "LoFi")
                if layer == 3 and phase == "decode"
                else ("BFLOAT8_B", "HiFi2" if layer == 3 else "LoFi")
            )
            assert all((r["weight_dtype"], r["fidelity"]) == expected for r in rows), (tag, phase, rows)
            if phase == "prefill":
                dense = [
                    r
                    for r in policies
                    if r["weight_shape"][-2:]
                    in [["2560[2560]", f"{n}[{n}]"] for n in (352, 320, 544, 3200, 4160, 4224)]
                ]
                assert dense and all(r["input_memory"] == "L1_INTERLEAVED" for r in dense), (tag, dense)
            if phase == "decode":
                assert all("in0_block_w=20" in r["attributes"] for r in rows)
                if layer == 3:
                    assert all(
                        "per_core_N=6" in r["attributes"] and "out_subblock_w=3" in r["attributes"] for r in rows
                    )
        timing = json.loads((DOC / run["timing"]).read_text())
        for argument in timing["native_expert_runtime_arguments"]:
            for role in ("gate", "up", "down"):
                assert argument[role]["count"] == 128 and argument[role]["dtypes"] == ["DataType.BFLOAT4_B"]
        if layer == 3:
            raw = list(csv.DictReader((DOC / "tracy" / tag / "ops.csv").open()))
            cache = [r for r in raw if "SparseSDPA" in r["OP CODE"] or "SparseSdpa" in r["OP CODE"]]
            assert cache, "Missing sparse SDPA runtime rows"
            run["kv_runtime_rows"] = [
                {k: v for k, v in r.items() if k in ["OP CODE", "INPUT_1_DATATYPE", "ATTRIBUTES"]} for r in cache
            ]
            assert all("FP8" in r["INPUT_1_DATATYPE"].upper() or "FP8" in r["ATTRIBUTES"].upper() for r in cache)
        runs.append(run)
    large = audit.collect("s5_final4_large_mesh_l0")
    small_groups = [
        r
        for r in large["policies"]["prefill"]
        if r["input_shape"][-2] == "128[128]"
        and r["weight_shape"][-2:] in [["2560[2560]", "544[544]"], ["2560[2560]", "320[320]"]]
    ]
    assert small_groups and all("in0_block_w=20" in r["attributes"] for r in small_groups), small_groups
    assert all(r["input_memory"] == "L1_INTERLEAVED" for r in small_groups)
    runs.append(large)
    (DOC / "profile_policy_audit.json").write_text(json.dumps(dict(status="passed", runs=runs), indent=2) + "\n")
    lines = [
        "# Final measured runtime policy",
        "",
        "Raw CSV hashes match the profiler provenance. Native expert runtime arguments are checked separately from generic-op labels.",
        "",
        "| Layer | Phase | Weight K×N | Input / weight / output dtype | Fidelity | Input memory |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        for phase, policies in run["policies"].items():
            for r in policies:
                lines.append(
                    f"| {run['tag']} | {phase} | {' × '.join(r['weight_shape'][-2:])} | {r['input_dtype']} / {r['weight_dtype']} / {r['output_dtype']} | {r['fidelity']} | {r['input_memory']} |"
                )
    (DOC / "profile_policy_audit.md").write_text("\n".join(lines) + "\n")
    print("FINAL_PROFILE_POLICY_PASS", len(runs))


if __name__ == "__main__":
    main()
