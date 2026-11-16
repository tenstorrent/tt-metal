"""Prepare, but never admit or execute, doubled-cap diagnostic continuations."""
import copy
import datetime
import hashlib
import json
from pathlib import Path

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
O = D / "setup/swe85-remainder-execution85"
OUT = D / "setup/swe85-cap-extensions"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value, private=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    if private:
        path.chmod(0o600)


def main():
    if (OUT / "admission.json").exists():
        raise RuntimeError("Refuse to overwrite admitted extensions")
    jobs = {j["name"]: j for j in read(D / "setup/swe85-admitted-controls85/combined-cases.json")}
    served = O / "served-semantic-coordinator-all36.json"
    sources = [served] + sorted(O.glob("full-semantic-coordinator-?.json"))
    by_mode = {"served": [], "full": []}
    evidence = []
    seen = set()
    for source in sources:
        mode = "served" if source == served else "full"
        for row in read(source)["rows"]:
            verdict = row.get("full_verdict", row.get("coordinator_verdict"))
            if verdict != "inconclusive":
                continue
            name, seed = row["job"], row["seed"]
            key = (mode, name, seed)
            if key in seen:
                raise ValueError(f"Duplicate adjudication: {key}")
            seen.add(key)
            job = jobs[name]
            result_path = D / "setup/swe85-pilot-split85/runs/rest" / mode / (name + ".json")
            result = read(result_path)
            assert "diagnostic_total_wall_s" in result
            sample = next(s for s in result["free_running"] if s["sampling"]["seed"] == seed)
            old_cap = job["free_run_tokens"]
            assert len(sample["token_ids"]) == old_cap, "Only cap-inconclusive samples qualify"
            case_path = Path(job["case"])
            case = copy.deepcopy(read(case_path))
            assert {k: v for k, v in sample["sampling"].items() if k != "seed"} == case["metadata"]["sampling"]
            case["metadata"]["seed"] = seed
            case["metadata"]["cap_extension"] = {
                "source_case": str(case_path),
                "source_case_sha256": sha(case_path),
                "source_result": str(result_path),
                "source_result_sha256": sha(result_path),
                "adjudication": str(source),
                "adjudication_sha256": sha(source),
                "mode": mode,
                "seed": seed,
                "previous_cap": old_cap,
                "new_cap": old_cap * 2,
                "reason": "Progressing at diagnostic cap; completion remains inconclusive.",
                "scope": "Same saved prompt IDs and sampling; extended diagnostic, no eval score change.",
            }
            new_name = name + f".cap{old_cap * 2}.seed{seed}"
            fixture = OUT / "fixtures" / mode / (new_name + ".private.json")
            write(fixture, case, private=True)
            by_mode[mode].append(
                {
                    "name": new_name,
                    "case": str(fixture),
                    "branches": [],
                    "free_run_tokens": old_cap * 2,
                    "free_run_trials": 1,
                }
            )
            evidence.append(
                {
                    "mode": mode,
                    "job": new_name,
                    "fixture_sha256": sha(fixture),
                    "source_job": name,
                    "source_result_sha256": sha(result_path),
                }
            )
    for mode, selected in by_mode.items():
        write(OUT / (mode + "-candidate-cases.json"), selected)
    write(
        OUT / "preparation.json",
        {
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "candidate_only_not_admitted_or_queued",
            "counts": {k: len(v) for k, v in by_mode.items()},
            "cases": evidence,
            "review_coverage": "Current coordinator-adjudicated cases only; refresh after remaining full review.",
            "gates": [
                "Finish active device queue; no overlap.",
                "Independent fixture review and cost gate before admission.",
                "Read extended outputs; any still-progressing cap remains inconclusive and needs further extension.",
            ],
        },
    )
    print(json.dumps({k: len(v) for k, v in by_mode.items()}))


if __name__ == "__main__":
    main()
