"""Validate saved cap-extension candidates and estimate their diagnostic cost, CPU only."""
import datetime
import hashlib
import json
from pathlib import Path

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
OUT = D / "setup/swe85-cap-extensions"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUT / "admission.json").exists():
        raise RuntimeError("Do not rewrite admitted evidence")
    validation, costs = [], []
    for mode in ("served", "full"):
        manifest = OUT / f"{mode}-candidate-cases.json"
        jobs = read(manifest)
        names, rows, prefills = set(), [], 0
        load = D / "setup/swe85-pilot-split85/runs/rest" / mode / "load.json"
        total = read(load)["wall_s"]
        for job in jobs:
            assert job["name"] not in names
            names.add(job["name"])
            fixture = Path(job["case"])
            case = read(fixture)
            meta = case["metadata"]["cap_extension"]
            source = Path(meta["source_case"])
            result = Path(meta["source_result"])
            adjudication = Path(meta["adjudication"])
            assert sha(source) == meta["source_case_sha256"]
            assert sha(result) == meta["source_result_sha256"]
            assert sha(adjudication) == meta["adjudication_sha256"]
            original = read(source)
            for key in ("prompt_token_ids", "continuations"):
                assert case[key] == original[key]
            assert case["metadata"]["sampling"] == original["metadata"]["sampling"]
            assert case["metadata"]["seed"] == meta["seed"]
            assert meta["mode"] == mode
            assert job["branches"] == [] and job["free_run_trials"] == 1
            assert job["free_run_tokens"] == meta["new_cap"] == 2 * meta["previous_cap"]
            sample = next(s for s in read(result)["free_running"] if s["sampling"]["seed"] == meta["seed"])
            assert len(sample["token_ids"]) == meta["previous_cap"]
            timing = sample["decode_timing"]
            assert timing["trace_counters_delta"]["captures"] == 0
            assert timing["trace_counters_delta"]["model_replays"] == timing["decode_tokens"]
            wall = sample["diagnostic_wall_s_including_prefill"]
            decode = timing["decode_wall_s"]
            assert wall >= decode > 0
            predicted = wall - decode + decode * (meta["new_cap"] - 1) / timing["decode_tokens"]
            total += predicted
            prefills += len(case["prompt_token_ids"])
            rows.append(
                {
                    "job": job["name"],
                    "fixture_sha256": sha(fixture),
                    "source_result_sha256": sha(result),
                    "predicted_seconds_at_new_cap": predicted,
                }
            )
        validation.append(
            {
                "mode": mode,
                "jobs": len(jobs),
                "manifest_sha256": sha(manifest),
                "total_prefill_tokens": prefills,
                "prompt_ids_continuations_sampling_unchanged": True,
                "source_hashes_and_traced_replays_verified": True,
            }
        )
        costs.append(
            {
                "mode": mode,
                "jobs": len(jobs),
                "manifest_sha256": sha(manifest),
                "model_load_source": str(load),
                "model_load_source_sha256": sha(load),
                "model_load_seconds_basis": read(load)["wall_s"],
                "predicted_hours": total / 3600,
                "predicted_interval_hours": [total / 3600 * 0.7, total / 3600 * 1.5],
                "rows": rows,
            }
        )
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for filename, rows in [("host-validation.json", validation), ("cost-estimate.json", costs)]:
        value = {
            "created_utc": now,
            "status": "candidate_snapshot_not_admitted",
            "rows": rows,
            "basis": "Same-case serial traced diagnostic wall: preserve prefill/host overhead, scale decode to new cap minus one, add observed model load. Planning interval 0.7–1.5x; not serving performance.",
            "limitation": "Refresh after remaining full review and candidate regeneration; independent fixture review required before admission.",
        }
        (OUT / filename).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps([{k: r[k] for k in ("mode", "jobs", "predicted_hours")} for r in costs]))


if __name__ == "__main__":
    main()
