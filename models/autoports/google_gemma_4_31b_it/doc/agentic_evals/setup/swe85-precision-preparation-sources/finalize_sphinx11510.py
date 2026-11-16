import hashlib
import json
import pathlib
import subprocess

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
P = D / "anomalies/swe85-below-band/correct-actions/sphinx-doc__sphinx-11510"
V = P / "validated"
V.mkdir(exist_ok=False)
V.chmod(0o700)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, j):
    p.write_text(json.dumps(j, indent=2) + "\n")
    p.chmod(0o600)


evidence = {}
for state in ["before", "after"]:
    p = P / "semantic-offline-wheels" / state
    prov = json.loads((p / "semantic-validation-provenance.json").read_text())
    r = json.loads((p / "verifier/report.json").read_text())["sphinx-doc__sphinx-11510"]
    assert prov["source_and_test_hashes_preserved"]
    assert r["resolved"] == (state == "after")
    evidence[state] = {
        "report": str(p / "verifier/report.json"),
        "report_sha256": sha(p / "verifier/report.json"),
        "provenance": str(p / "semantic-validation-provenance.json"),
        "provenance_sha256": sha(p / "semantic-validation-provenance.json"),
        "resolved": r["resolved"],
        "tests_status": r["tests_status"],
    }
c = json.loads((P / "action_counterfactual.candidate.private.json").read_text())
c["metadata"][
    "correct_semantic_validation"
] = "Validated diagnostic source state: before fails2 source-read tests/passes7 regressions; after passes2+7, unchanged offline verifier, hash-verified build wheels, network none,1CPU4GiB. Shell action not executed by verifier; exact materialized patch after-state tested."
c["metadata"]["semantic_validation_evidence"] = evidence
c["metadata"][
    "source_status"
] = "Exact cached image checkout files; no earlier checkout source edits before raw24. Earlier native writes/installations affect installed site-packages or reproduction files. Relevant production source state only, not complete native environment replay."
p = V / "action_counterfactual.validated.private.json"
write(p, c)
jobs = [{"name": "sphinx-doc__sphinx-11510.validated-action", "case": str(p), "branches": ["chosen", "correct"]}]
write(V / "cases.json", jobs)
commands = {}
checks = []
for mode, config in [("served", "head_bfp8_build_kwargs.json"), ("full", "full_precision_build_kwargs.json")]:
    a = [
        "python_env/bin/python",
        "-m",
        "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
        "--cases",
        str(V / "cases.json"),
        "--build-kwargs-file",
        str(D / "setup" / config),
        "--output-dir",
        str(V / mode),
    ]
    if mode == "full":
        a += ["--compare-dir", str(V / "served")]
    r = subprocess.run(a + ["--validate-only"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    commands[mode] = a
    checks.append({"mode": mode, "exit_code": r.returncode, "stdout": r.stdout.strip()})
write(
    V / "manifest.json",
    {
        "task": "sphinx-doc__sphinx-11510",
        "call": 24,
        "native_message": 48,
        "response_id": "chatcmpl-80a175e4f4c12f83",
        "status": "semantic_target_validated_TT_not_executed",
        "action_case": str(p),
        "action_case_sha256": sha(p),
        "candidate_manifest": str(P / "manifest.json"),
        "candidate_manifest_sha256": sha(P / "manifest.json"),
        "version_evidence": str(P / "checkout-version-evidence.json"),
        "semantic_evidence": evidence,
        "initial_offline_failure_preserved": str(P / "semantic-offline"),
        "reference_informed_private_diagnostic_only": True,
        "native_score_unchanged": True,
        "source_and_test_hashes_preserved": True,
        "cpu_validation": checks,
        "commands_staged_only": commands,
        "limits": "Original reasoning prefix retained, may be faulty. Correct action conditions on that prefix; no corrected reasoning or model answer claim. Chosen full reply separately retained. Token IDs reconstructed with pinned HF template, native token identity unavailable.",
    },
)
print(json.dumps({"validated_manifest": str(V / "manifest.json"), "checks": checks}))
