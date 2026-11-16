"""Bind isolated original-test evidence to supplemental controls, never frozen admission."""
import hashlib
import json
import subprocess
from pathlib import Path

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
B = D / "anomalies/swe85-below-band"
P = B / "correct-actions/pydata__xarray-3095"
O = P / "validated"
O.mkdir(exist_ok=True)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, j):
    p.write_text(json.dumps(j, indent=2) + "\n")
    p.chmod(0o600)


rows = []
for state, rel, manifest in [
    ("before", "pydata__xarray-3095-attempt2/before", P / "manifest.json"),
    (
        "chosen48",
        "pydata__xarray-3095-chosen48-attempt1",
        B / "correct-actions/pydata__xarray-3095-chosen48/manifest.json",
    ),
    ("correct", "pydata__xarray-3095-attempt2/after", P / "manifest.json"),
]:
    p = B / "correct-actions-semantic" / rel
    prov = p / "semantic-validation-provenance.json"
    report = p / "verifier/report.json"
    j = read(prov)
    r = read(report)["pydata__xarray-3095"]
    assert j["candidate_manifest_sha256"] == sha(manifest)
    assert j["source_and_test_hashes_preserved"]
    rows.append(
        {
            "state": state,
            "provenance": str(prov),
            "provenance_sha256": sha(prov),
            "candidate_manifest": str(manifest),
            "candidate_manifest_sha256": sha(manifest),
            "report": str(report),
            "report_sha256": sha(report),
            "resolved": r["resolved"],
            "counts": {k: {a: len(v) for a, v in group.items()} for k, group in r["tests_status"].items()},
            "image_id": j["image_id"],
            "test_hashes": j["test_hashes"],
            "recovery_wheels": j["recovery_wheels"],
            "source_files": j["source_files"],
        }
    )
assert [r["resolved"] for r in rows] == [False, False, True]
assert all(
    (r["image_id"], r["test_hashes"], r["recovery_wheels"])
    == (rows[0]["image_id"], rows[0]["test_hashes"], rows[0]["recovery_wheels"])
    for r in rows
)
original = P / "action_counterfactual.candidate.private.json"
case = read(original)
case["metadata"][
    "correct_semantic_validation"
] = "Original expected254 tests pass; before/chosen dtype target fails, adjacent253 pass. Diagnostic only."
case["metadata"]["native_message"] = 96
case["metadata"]["native_token_ids_available"] = False
out = O / "pydata__xarray-3095.call48.action.private.json"
write(out, case)
action = {"name": "pydata__xarray-3095.call48.validated-action", "case": str(out), "branches": ["chosen", "correct"]}
write(O / "cases.json", [action])
whole = read(B / "full-replies-xarray3095-semantic/cases.json")[0]
whole["name"] = "fullreply." + whole["name"]
combined = [whole, {**action, "name": "action." + action["name"]}]
write(O / "supplemental-cases.json", combined)
validation = []
for mode, cfg in [("served", "head_bfp8_build_kwargs.json"), ("full", "full_precision_build_kwargs.json")]:
    cmd = [
        "python_env/bin/python",
        "-m",
        "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
        "--cases",
        str(O / "supplemental-cases.json"),
        "--build-kwargs-file",
        str(D / "setup" / cfg),
        "--output-dir",
        "/tmp/xarray3095-supplement-validation-unused",
        "--validate-only",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0
    validation.append({"mode": mode, "exit_code": r.returncode, "stdout": r.stdout.strip()})
manifest = {
    "task": "pydata__xarray-3095",
    "call": 48,
    "native_message": 96,
    "response_id": case["metadata"]["response_id"],
    "status": "validated_supplement_pending_root_review_not_admitted",
    "candidate_manifest": str(P / "manifest.json"),
    "action_case": str(out),
    "action_case_sha256": sha(out),
    "original_pending_case": str(original),
    "original_pending_case_sha256": sha(original),
    "semantic_evidence": {"states": rows},
    "native_source_read_checks": str(P / "native-source-checks.json"),
    "native_source_read_checks_sha256": sha(P / "native-source-checks.json"),
    "whole_reply": whole,
    "CPU_validation": validation,
    "limits": "Raw48 first ineffective source edit; raw53 is independent malformed inspection tool and remains protocol-only. Reference-informed correct action conditioned on original reasoning, not a model answer; no native token equivalence, full filesystem/dependency replay, scored rescore or inference. Frozen161 admission unchanged. Original official test.sh/config unchanged; pinned original derived verifier image recovered by ID after tag removed. Attempt1 parser-absent base image failure preserved.",
}
write(O / "manifest.json", manifest)
print(
    json.dumps(
        {
            "supplement": str(O / "supplemental-cases.json"),
            "sha256": sha(O / "supplemental-cases.json"),
            "states": [{k: r[k] for k in ["state", "resolved", "counts"]} for r in rows],
        }
    )
)
