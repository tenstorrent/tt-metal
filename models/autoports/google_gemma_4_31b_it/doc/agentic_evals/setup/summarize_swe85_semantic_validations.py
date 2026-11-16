"""Summarize diagnostic artifacts only, without editing shared scoring ledgers."""
import hashlib
import json
import pathlib

B = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals/anomalies/swe85-below-band")


def read(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, j):
    p.write_text(json.dumps(j, indent=2) + "\n")
    p.chmod(0o600)


fetch = []
for mp in sorted((B / "fetch-controls").glob("*/manifest.json")):
    m = read(mp)
    task = mp.parent.name
    attempts = []
    for p in sorted((B / "fetch-semantic-validation").glob(task + "-attempt*/semantic-validation.json")):
        j = read(p)
        if j:
            attempts.append((p, j))
    good = [
        (p, j)
        for p, j in attempts
        if j.get("valid_next_action") is True
        and j.get("cleanup_returncode") == 0
        and j.get("candidate_manifest_sha256") == sha(mp)
    ]
    selected = good[-1] if good else None
    fetch.append(
        {
            "task": task,
            "call": m["call"],
            "valid_next_action": bool(selected),
            "full_solution_claim": False,
            "evidence": str(selected[0]) if selected else None,
            "candidate_manifest_sha256": sha(mp),
            "action_case": m["action_case"],
            "action_case_sha256": sha(pathlib.Path(m["action_case"])),
            "scope": m["scope"],
            "expectation_correction": m.get("expectation_correction"),
            "attempts": [
                {
                    "evidence": str(p),
                    "status": j["status"],
                    "valid_next_action": j.get("valid_next_action"),
                    "error": j.get("error"),
                }
                for p, j in attempts
            ],
        }
    )
write(
    B / "fetch-semantic-validation/summary.json",
    {
        "status": "diagnostic_only_root_review_before_precision_admission",
        "n": len(fetch),
        "valid_next_actions": sum(r["valid_next_action"] for r in fetch),
        "tasks": fetch,
        "limitations": "Local pinned image plus recorded tracked edits/helpers, not whole native filesystem/dependency replay. No full solution or scoring claim. Attempt1 infrastructure capdrop/0600 issue preserved; driver adds DAC_OVERRIDE. Two expectations corrected with unchanged action/continuation (MPL26291 bug not reproduced, Xarray4687 baseline lacks keep_attrs).",
        "TT_executed": False,
    },
)
correct = []
for row in read(B / "correct-actions/all-candidate-action-cases.json"):
    cp = pathlib.Path(row["case"])
    candidate = cp.parent.name
    mp = cp.parent / "manifest.json"
    m = read(mp)
    pairs = []
    evidence_dirs = sorted((B / "correct-actions-semantic").glob(candidate + "-attempt*"))
    if m["task"] == "sympy__sympy-14248":
        sp = pathlib.Path(m["semantic_evidence"]["summary"])
        sv = read(sp)
        assert sha(sp) == m["semantic_evidence"]["sha256"] and sha(cp) == m["action_case_sha256"]
        assert sv["state_manifest"] == m["source_state_manifest"]
        assert (
            sha(pathlib.Path(sv["state_manifest"])) == sv["state_manifest_sha256"] == m["source_state_manifest_sha256"]
        )
        assert sv["official_test_script_unchanged"] and sv["network"] == "none"
        assert [r["state"] for r in sv["states"]] == ["before", "chosen50", "correct"]
        for state in sv["states"]:
            assert state["source_and_test_hashes_preserved"]
            assert sha(pathlib.Path(state["report"])) == state["report_sha256"]
            assert read(pathlib.Path(state["report"]))[m["task"]] == state["result"]
        assert [r["result"]["resolved"] for r in sv["states"]] == [False, False, True]
        pairs.append(
            {
                "attempt": str(sp.parent),
                "before_resolved": False,
                "after_resolved": True,
                "same_image_tests_wheels_manifest": True,
                "both_source_test_hashchecks": True,
                "candidate_target_tests_pass": True,
                "before_failure_to_after_pass": True,
                "network": "none",
                "summary_sha256": sha(sp),
                "scope": sv["scope"],
                "raw_baseline_ast_exceptions": 4,
            }
        )

    if m.get("candidate_manifest") and m.get("semantic_evidence"):
        mp = pathlib.Path(m["candidate_manifest"])
        candidate = m["task"]
        evidence_dirs = [pathlib.Path(m["semantic_evidence"]["after"]["provenance"]).parents[1]]
        m = read(mp)
    for p in evidence_dirs:
        bp = read(p / "before/semantic-validation-provenance.json")
        ap = read(p / "after/semantic-validation-provenance.json")
        br = read(p / "before/verifier/report.json")
        ar = read(p / "after/verifier/report.json")
        if not all([bp, ap, br, ar]):
            continue
        t = m["task"]
        same = (
            bp["image_id"] == ap["image_id"]
            and bp["test_hashes"] == ap["test_hashes"]
            and bp.get("recovery_wheels") == ap.get("recovery_wheels")
            and bp.get("environment_compatibility") == ap.get("environment_compatibility")
            and bp["candidate_manifest_sha256"] == ap["candidate_manifest_sha256"] == sha(mp)
        )
        verified = bp.get("source_and_test_hashes_preserved") and ap.get("source_and_test_hashes_preserved")
        before = br[t].get("resolved")
        after = ar[t].get("resolved")
        pairs.append(
            {
                "attempt": str(p),
                "before_resolved": before,
                "after_resolved": after,
                "same_image_tests_wheels_manifest": same,
                "both_source_test_hashchecks": verified,
                "candidate_target_tests_pass": bool(after and same and verified),
                "before_failure_to_after_pass": bool(not before and after and same and verified),
                "network": ap["network"],
                "environment_compatibility": ap.get("environment_compatibility"),
                "fresh_process_isolated_checks": ap.get("fresh_process_isolated_checks"),
            }
        )
    correct.append(
        {
            "candidate": candidate,
            "task": m["task"],
            "action_case": str(cp),
            "action_case_sha256": sha(cp),
            "pairs": pairs,
            "valid_candidate_target_tests": any(r["candidate_target_tests_pass"] for r in pairs),
        }
    )
for row in correct:
    comparison = B / "correct-actions-semantic" / (row["candidate"] + "-chosen-comparison.json")
    if comparison.exists():
        row["chosen_comparison"] = {"evidence": str(comparison), "status": read(comparison)["status"]}
    if row["task"] == "psf__requests-5414":
        local = B / "correct-actions/psf__requests-5414/semantic-pure-urls/summary.json"
        row["narrow_url_regression"] = {
            "evidence": str(local),
            "sha256": sha(local),
            "status": read(local)["status"],
            "full_suite_pass": False,
            "root_review_required": True,
        }
    if row["task"] == "psf__requests-1921":
        local = B / "correct-actions-semantic/psf__requests-1921-local-auth-attempt1/summary.json"
        if local.exists():
            row["narrow_auth_regression"] = {
                "evidence": str(local),
                "status": read(local)["status"],
                "full_suite_pass": False,
                "root_review_required": True,
            }
write(
    B / "correct-actions-semantic/summary.json",
    {
        "status": "current_roster_executed_diagnostic_only_root_review_required",
        "candidate_count": len(correct),
        "validated_candidates": sum(r["valid_candidate_target_tests"] for r in correct),
        "candidates": correct,
        "limits": "Before/after materialized patch-state validation using full original test.sh, not execution of candidate shell command. Native whole filesystem/dependency equivalence not claimed. No scores or shared ledgers changed.",
    },
)
print(
    "fetch",
    sum(r["valid_next_action"] for r in fetch),
    "/",
    len(fetch),
    "correct",
    sum(r["valid_candidate_target_tests"] for r in correct),
    "/",
    len(correct),
)
