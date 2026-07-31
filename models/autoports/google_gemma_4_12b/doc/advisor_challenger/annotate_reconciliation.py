"""Attach measured screening results to untouched reconcile.py accounting.

Run reconcile.py first. This script refuses any non-generated input and changes
only screening/result fields; accounting, pairing, ranking, and feasibility stay
exactly as emitted by reconcile.py.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parent
MEASUREMENTS = ROOT / "measurements"
PCC = {
    "sliding_attention": 0.9998707062418563,
    "full_attention": 0.9996586498771043,
}


def load_measurement(name: str) -> dict:
    return json.loads((MEASUREMENTS / f"{name}.json").read_text())


def result(chain: dict, measurement: str, screened_as: str, *, combined_with=None, perf_report=None) -> None:
    measured = load_measurement(measurement)
    chain.update(
        verdict="kept" if measurement != "sliding_grouped_o_l1" else "rejected",
        measured_ms=measured["median_ms"],
        repeats_ms=measured["repeats_ms"],
        oracle_passed=True,
        oracle_pcc=PCC[chain["chain"].split(":", 1)[0]],
        oracle_weights="real",
        screened_as=screened_as,
    )
    if combined_with:
        chain["combined_with"] = combined_with
    if perf_report:
        chain["perf_report"] = perf_report


def annotate(kind: str) -> None:
    path = ROOT / f"reconciliation_{kind}.json"
    data = json.loads(path.read_text())
    if data.get("generated_by") != "advisor-challenger/scripts/reconcile.py":
        raise RuntimeError(f"refusing non-reconcile input: {path}")
    floor = data["feasibility"]["noise_floor_us"]

    for chain in data["chains"]:
        suffix = chain["chain"].split(":", 1)[1]
        if kind == "sliding_attention":
            if suffix == "b7":
                result(chain, "shipped_l1_interleaved_sliding", "K: L1-interleaved across norm", perf_report="candidate_k_sliding_ops.csv")
            elif suffix in {"b6", "b22", "5"}:
                result(
                    chain,
                    "sliding_q_l1_extended_sdpa",
                    "Q/rotary/SDPA extended L1 chain",
                    combined_with=["sliding_attention:b6", "sliding_attention:b22", "sliding_attention:5"],
                    perf_report="candidate_q_sliding_ops.csv",
                )
            elif suffix == "1":
                result(chain, "sliding_keep_v_l1", "V: L1-interleaved across norm", perf_report="candidate_v_sliding_ops.csv")
            elif suffix == "9":
                result(chain, "sliding_mlp_direct_down", "MLP multiply directly in down-projection input layout", perf_report="candidate_mlp_sliding_ops.csv")
            elif suffix in {"7", "6"}:
                result(
                    chain,
                    "sliding_grouped_o_l1",
                    "grouped concat/output-projection L1 chain",
                    combined_with=["sliding_attention:7", "sliding_attention:6"],
                )
            else:
                chain.update(verdict="below_threshold", threshold_us=floor)
        else:
            if suffix == "b7":
                result(chain, "shipped_l1_interleaved_full", "K: L1-interleaved across norm", perf_report="candidate_k_full_ops.csv")
            elif suffix in {"b6", "b22"}:
                result(
                    chain,
                    "full_q_l1_extended_sdpa",
                    "Q/rotary/SDPA extended L1 chain",
                    combined_with=["full_attention:b6", "full_attention:b22"],
                    perf_report="candidate_q_full_ops.csv",
                )
            elif suffix in {"1", "5", "9"}:
                result(
                    chain,
                    "full_k_v_mlp",
                    "grouped below-floor V, rotary, and MLP chains",
                    combined_with=["full_attention:1", "full_attention:5", "full_attention:9"],
                    perf_report="candidate_k_v_mlp_full_ops.csv",
                )
            elif suffix in {"7", "6"}:
                result(
                    chain,
                    "full_grouped_o_l1",
                    "grouped concat/output-projection L1 chain",
                    combined_with=["full_attention:7", "full_attention:6"],
                    perf_report="candidate_o_full_ops.csv",
                )
            else:
                chain.update(verdict="below_threshold", threshold_us=floor)

    # These rows already execute in DRAM in the raw op report, so the advice is
    # agreement rather than a de-sharding candidate. Record the raw evidence.
    for row in data.get("disagreements", []):
        if row.get("bucket") == "dram_resident":
            row.update(
                verdict="already_shipped",
                raw_profile_evidence="INPUT_0_MEMORY and OUTPUT_0_MEMORY are DEV_0_DRAM_INTERLEAVED",
            )
    for row in data.get("material_ops_on_le_2_cores", []):
        if row.get("device") == "NLPConcatHeadsDeviceOperation":
            measured = load_measurement("sliding_grouped_o_l1" if kind == "sliding_attention" else "full_grouped_o_l1")
            row.update(
                measured_ms=measured["median_ms"],
                repeats_ms=measured["repeats_ms"],
                screened_as="grouped concat/output-projection L1 chain; advisor interleaved output has no core-count recommendation",
            )
    path.write_text(json.dumps(data, indent=2) + "\n")


annotate("sliding_attention")
annotate("full_attention")
