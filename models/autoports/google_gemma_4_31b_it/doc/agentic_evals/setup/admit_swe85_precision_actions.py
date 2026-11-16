"""Build a hash-bound diagnostic manifest from reviewed local semantic evidence.

This prepares data only. It never opens a device or changes scored results.
Run again into a new snapshot directory after newly completed tasks are reviewed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = root / "setup/swe85-tt-execution-plan"
    evidence = root / "anomalies/swe85-below-band"
    correct_path = evidence / "correct-actions-semantic/summary.json"
    fetch_path = evidence / "fetch-semantic-validation/summary.json"
    correct, fetch = read(correct_path), read(fetch_path)
    indexed = {r["action_case"]: r for r in correct["candidates"]}
    indexed.update({r["action_case"]: r for r in fetch["tasks"]})
    jobs = read(plan / "combined-cases.json")
    admitted = []
    narrow_sources = {}
    for item in read(plan / "deferred-action-candidates.json"):
        case_path = item["case"]
        row = indexed[case_path]
        assert sha(case_path) == row["action_case_sha256"], case_path
        limitation = None
        if item["kind"] == "fetch":
            assert row["valid_next_action"] and not row["full_solution_claim"]
            limitation = "Valid local next action only; not a full task solution."
        elif row["valid_candidate_target_tests"]:
            assert any(
                p["candidate_target_tests_pass"]
                and p["same_image_tests_wheels_manifest"]
                and p["both_source_test_hashchecks"]
                for p in row["pairs"]
            )
            if item["task"] == "sympy__sympy-14248":
                limitation = "Official expected2+433 tests pass;4 unchanged baseline AST exceptions and1 XPASS retained. Eleven unrelated native test mutations excluded; production pre50 edits retained."
            if item["task"] == "sympy__sympy-12489":
                limitation = "Paired diagnostic Callable compatibility binding; not unchanged reference environment."
        elif item["task"] == "psf__requests-5414":
            manifest = read(Path(case_path).parent / "manifest.json")
            ev = manifest["semantic_evidence"]
            narrow_path = Path(ev["narrow_summary"])
            assert sha(narrow_path) == ev["narrow_summary_sha256"]
            assert any(r["case"] == case_path and r["sha256"] == sha(case_path) for r in manifest["cases"])
            narrow_sources[str(narrow_path)] = sha(narrow_path)
            narrow = read(narrow_path)
            states = narrow["states"]
            assert [r["counts"] for r in states] == [
                {"pass": 31, "fail": 1, "error": 0, "skip": 0},
                {"pass": 24, "fail": 8, "error": 0, "skip": 0},
                {"pass": 32, "fail": 0, "error": 0, "skip": 0},
            ]
            assert narrow["source_and_test_hashes_preserved"]
            assert all(r["tests_unchanged"] and sha(r["source"]) == r["source_sha256"] for r in states)
            assert all(
                r["tests"] == states[0]["tests"]
                and r["test_hashes_before_after"] == states[0]["test_hashes_before_after"]
                for r in states
            )
            limitation = "Original pure URL targets only: before31/32, chosen24/32, correct32/32. Full suite retains158 fixture errors and4 offline timeouts; no full-suite pass."
        else:
            assert item["task"] == "psf__requests-1921", case_path
            narrow_path = evidence / "correct-actions-semantic/psf__requests-1921-local-auth-attempt1/summary.json"
            narrow_sources[str(narrow_path)] = sha(narrow_path)
            narrow = read(narrow_path)
            assert [r["passed"] for r in narrow["rows"]] == [True, False, True]
            source = Path(case_path).parent / "candidate/requests/sessions.py"
            assert sha(source) == narrow["rows"][2]["source_sha256"]
            limitation = "Exact merge_setting auth path only; full suite remains externally limited."
        case = read(case_path)
        assert case["continuations"]["chosen"] and case["continuations"]["correct"]
        job = {"name": "action." + item["name"], "case": case_path, "branches": ["chosen", "correct"]}
        jobs.append(job)
        admitted.append(
            {**job, "sha256": sha(case_path), "task": item["task"], "kind": item["kind"], "limitation": limitation}
        )
    assert len({j["name"] for j in jobs}) == len(jobs)
    # A normal whole-reply case supplies the first long-prefix cost observation.
    pilot = max(
        (j for j in jobs if j["name"].startswith("fullreply.")), key=lambda j: len(read(j["case"])["prompt_token_ids"])
    )
    jobs = [pilot] + [j for j in jobs if j is not pilot]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = [
        correct_path,
        fetch_path,
        plan / "combined-cases.json",
        plan / "deferred-action-candidates.json",
        root / "review/swe85-fixtures-review.md",
    ]
    record = {
        "status": "host_admitted_not_executed",
        "jobs": len(jobs),
        "action_jobs": len(admitted),
        "pilot": pilot["name"],
        "sources": {**{str(p): sha(p) for p in sources}, **narrow_sources},
        "admitted": admitted,
        "limits": "Canonical reconstruction, conditional on original reasoning. No native token equivalence, task rescore or device-result claim. Require complete roster reconciliation and remaining-budget check before launch.",
    }
    for name, value in [("combined-cases.json", jobs), ("admission.json", record)]:
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"jobs": len(jobs), "actions": len(admitted), "pilot": pilot["name"]}))


if __name__ == "__main__":
    main()
