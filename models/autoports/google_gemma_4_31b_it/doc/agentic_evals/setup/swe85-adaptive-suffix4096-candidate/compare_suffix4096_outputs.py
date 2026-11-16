#!/usr/bin/env python3
"""Host-only compare matched suffix4096 chosen-token outputs; no model imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
OUT = D / "setup/swe85-adaptive-suffix4096-candidate"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    served_jobs = json.loads((OUT / "served-cases.json").read_text())
    full_jobs = json.loads((OUT / "full-cases.json").read_text())
    served_by_name = {j["name"]: j for j in served_jobs}
    full_by_name = {j["name"]: j for j in full_jobs}
    served_names = set(served_by_name)
    full_names = set(full_by_name)
    paired = sorted(served_names & full_names)
    full_only = sorted(full_names - served_names)
    assert len(served_jobs) == 12 and len(paired) == 12, "Expected exactly twelve matched suffix4096 cases"
    assert full_only == ["anomaly.pytest-dev__pytest-5787.call65.onset.cap4096.seed44"], full_only
    rows = []
    for name in paired:
        sj = served_by_name[name]
        fj = full_by_name[name]
        sf = json.loads((ROOT / sj["case"]).read_text())
        ff = json.loads((ROOT / fj["case"]).read_text())
        assert sf["prompt_token_ids"] == ff["prompt_token_ids"], name
        assert sf["continuations"]["chosen"] == ff["continuations"]["chosen"], name
        assert sf["metadata"]["sampling"] == ff["metadata"]["sampling"], name
        assert sf["metadata"]["seed"] == ff["metadata"]["seed"], name
        sp = OUT / "runs/served" / (name + ".json")
        fp = OUT / "runs/full" / (name + ".json")
        served = json.loads(sp.read_text())
        full = json.loads(fp.read_text())
        assert "diagnostic_total_wall_s" in served and "diagnostic_total_wall_s" in full
        assert served["case"] != full["case"] or served["case"], "Case provenance missing"
        assert set(served["continuations"]) == {"chosen"} and set(full["continuations"]) == {"chosen"}
        a = served["continuations"]["chosen"]
        b = full["continuations"]["chosen"]
        assert a["conditioning_token_ids"] == b["conditioning_token_ids"], name
        left = [(r["position"], r["forced_token"]) for r in a["positions"]]
        right = [(r["position"], r["forced_token"]) for r in b["positions"]]
        assert left == right, name
        n = len(left)
        rows.append(
            {
                "name": name,
                "tokens": n,
                "top1_agreement": sum(
                    x["top1_token"] == y["top1_token"] for x, y in zip(a["positions"], b["positions"])
                )
                / n,
                "served_sequence_log_probability": a["sequence_log_probability"],
                "full_sequence_log_probability": b["sequence_log_probability"],
                "served_result_sha256": sha(sp),
                "full_result_sha256": sha(fp),
            }
        )
    report = {
        "status": "passed",
        "scope": "Host-only comparison of matched forced-token chosen branches; no TTNN/device work.",
        "paired_case_count": len(rows),
        "paired_rows": rows,
        "full_only_cases": full_only,
        "full_only_policy": "Pytest cap4096 is a progressing free-run-only extension with no served counterpart; excluded from forced-token comparison.",
        "served_manifest_sha256": sha(OUT / "served-cases.json"),
        "full_manifest_sha256": sha(OUT / "full-cases.json"),
    }
    target = OUT / "comparison-summary.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    target.chmod(0o600)


if __name__ == "__main__":
    main()
