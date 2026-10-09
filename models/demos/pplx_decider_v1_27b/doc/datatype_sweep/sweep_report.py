# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collect the stage-8 sweep results into sweep_results.json and sweep_results.csv.

Inputs per candidate (written with ``$PPLX_DECIDER_STAGE6_DIR`` = ``<artifacts>/<candidate id>``):
``e2e_decisions.json`` (tests/e2e/test_model.py), ``model_perf.json`` (tests/perf/test_model_perf.py)
and ``e2e.log`` (the ``model loaded ... DRAM`` line). Candidate policies: ``candidates/<id>.json``.
Candidates without results are listed with the reason from ``NOT_RUN``.

Run::

    python models/demos/pplx_decider_v1_27b/doc/datatype_sweep/sweep_report.py [artifacts dir]
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = Path(sys.argv[1] if len(sys.argv) > 1 else "/local/ttuser/gtobar/artifacts/pplx_decider/stage8")
BUCKETS = ("128", "1024", "2048", "4096", "8192")
MIN_AGREE, NEAR_TIE, MIN_LOGIT_PCC = 24, 0.05, 0.99
HARDWARE = "1x Blackhole p150a (single device), batch 1, eager"
ROLES = ("attention_qkvg", "attention_out", "delta_in", "delta_out", "mlp_gate_up", "mlp_down", "readout")
SHORT = {"bfloat8_b": "bfp8", "bfloat4_b": "bfp4", "bfloat16": "bf16"}
NOT_RUN = "not run - person selected C0 after C1 failed the logit-PCC gate (l01 0.98895 < 0.99)"


def gate(summary: dict, rows: list[dict]) -> tuple[bool, str]:
    reasons = []
    if summary["agree"] < MIN_AGREE:
        reasons.append(f"agree {summary['agree']}/{summary['rows']} < {MIN_AGREE}")
    for r in rows:
        if not r["agree"] and r["hf_top2_gap"] >= NEAR_TIE:
            reasons.append(f"miss {r['id']} on a non-tie (HF gap {r['hf_top2_gap']:.4f})")
        if not r["logit_pcc_valid"] >= MIN_LOGIT_PCC:
            reasons.append(f"{r['id']} logit PCC {r['logit_pcc_valid']:.5f} < {MIN_LOGIT_PCC}")
    return not reasons, "; ".join(reasons) or "pass"


def candidate(path: Path) -> dict:
    cid = path.stem
    policy = json.loads(path.read_text())["policy"]
    out = ART / cid
    entry = {
        "config_id": cid,
        "policy_name": policy["name"],
        "weight_dtypes": policy["weight_dtypes"],
        "fidelities": policy["fidelities"],
        "activation_dtype": policy["activation_dtype"],
        "embedding_dtype": policy["embedding_dtype"],
        "recurrent_dtype": policy["recurrent_dtype"],
        "hardware": HARDWARE,
    }
    if not (out / "e2e_decisions.json").exists():
        return {**entry, "status": NOT_RUN, "gate_pass": None}
    e2e = json.loads((out / "e2e_decisions.json").read_text())
    s = e2e["summary"]
    passed, why = gate(s, e2e["rows"])
    dram = re.findall(r"model loaded: .*'allocated_gib': ([0-9.]+)", (out / "e2e.log").read_text())
    perf_path = out / "model_perf.json"
    perf = json.loads(perf_path.read_text())["buckets"] if perf_path.exists() else {}
    return {
        **entry,
        "status": "measured",
        "agree": s["agree"],
        "rows": s["rows"],
        "misses": s["misses"],
        "logit_pcc_min": s["logit_pcc_valid_min"],
        "logit_pcc_median": s["logit_pcc_valid_median"],
        "final_hidden_pcc_min": s["final_hidden_pcc_min"],
        "final_hidden_pcc_median": s["final_hidden_pcc_median"],
        "max_abs_prob_diff": s["max_abs_prob_diff_max"],
        "gate_pass": passed,
        "gate_detail": why,
        "dram_allocated_gib_after_load": float(dram[-1]) if dram else None,
        "request_burst_ms": {b: perf[b]["request_burst"]["median_ms"] for b in BUCKETS if b in perf},
        "request_sustained_ms": {b: perf[b]["request_sustained"]["median_ms"] for b in BUCKETS if b in perf},
        "device_forward_burst_ms": {b: perf[b]["device_forward_burst"]["median_ms"] for b in BUCKETS if b in perf},
        "command": (
            f"PPLX_DECIDER_PRECISION_CONFIG=models/demos/pplx_decider_v1_27b/doc/datatype_sweep/candidates/{cid}.json "
            f"PPLX_DECIDER_STAGE6_DIR={out} pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_model.py -q -s; "
            "same env: pytest models/demos/pplx_decider_v1_27b/tests/perf/test_model_perf.py -k test_model_perf -q -s"
        ),
    }


def main():
    results = [candidate(p) for p in sorted((HERE / "candidates").glob("*.json"))]
    passing = [r for r in results if r["gate_pass"] and "2048" in r["request_burst_ms"]]
    fastest = min(passing, key=lambda r: (r["request_burst_ms"]["2048"], r["request_burst_ms"].get("8192", 0)))
    selected_id = json.loads((HERE / "selected_precision_config.json").read_text())["config_id"]
    for r in results:
        r["selected"] = r["config_id"].split("_")[0] == selected_id
    (HERE / "sweep_results.json").write_text(
        json.dumps(
            {
                "gate": {"min_agree": MIN_AGREE, "near_tie_gap": NEAR_TIE, "min_logit_pcc": MIN_LOGIT_PCC},
                "fastest_passing": fastest["config_id"],
                "selected": selected_id,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    cols = ["config_id", "status", *ROLES, "agree", "logit_pcc_min", "logit_pcc_median", "final_hidden_pcc_min"]
    cols += ["max_abs_prob_diff", "dram_gib", *[f"req_burst_{b}_ms" for b in BUCKETS]]
    cols += [f"req_sustained_{b}_ms" for b in BUCKETS] + ["gate_pass", "selected", "gate_detail", "command", "hardware"]
    with (HERE / "sweep_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {**r, **{g: f"{SHORT[r['weight_dtypes'][g]]}/{r['fidelities'][g]}" for g in ROLES}}
            if r["status"] == "measured":
                row["agree"] = f"{r['agree']}/{r['rows']}"
                row["dram_gib"] = r["dram_allocated_gib_after_load"]
                for b in BUCKETS:
                    row[f"req_burst_{b}_ms"] = round(r["request_burst_ms"].get(b, float("nan")), 1)
                    row[f"req_sustained_{b}_ms"] = round(r["request_sustained_ms"].get(b, float("nan")), 1)
            w.writerow(row)
    for r in results:
        if r["status"] != "measured":
            print(f"{r['config_id']:<22} {r['status']}")
            continue
        b = r["request_burst_ms"]
        print(
            f"{r['config_id']:<22} agree {r['agree']}/25 logitPCC min {r['logit_pcc_min']:.5f} med "
            f"{r['logit_pcc_median']:.5f} hid min {r['final_hidden_pcc_min']:.5f} dprob {r['max_abs_prob_diff']:.4f} "
            f"DRAM {r['dram_allocated_gib_after_load']:.2f} burst {[round(b.get(k, float('nan')), 1) for k in BUCKETS]} "
            f"gate {r['gate_detail']}"
        )
    print(f"fastest passing: {fastest['config_id']}; selected: {selected_id}")


if __name__ == "__main__":
    main()
