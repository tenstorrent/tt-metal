"""Audit completed diagnostic artifacts without importing TTNN or opening devices."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[7]
assert (ROOT / "bringup/run_contract.json").is_file()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = HERE.parent / "proposal-with-cap4096-pending-v-closure/candidate-cases.json"
jobs = read(manifest)
expected = read(HERE / "head_bf16_hifi4_precision.json")
rows = []
missing = []
for job in jobs:
    path = HERE / "runs/head_bf16_hifi4" / (job["name"] + ".json")
    if not path.exists():
        missing.append(job["name"])
        continue
    result = read(path)
    fixture = ROOT / job["case"]
    case = read(fixture)
    assert result["case"] == job["case"]
    assert result["runtime_precision"] == expected
    assert result["continuations"] == {} and job["branches"] == []
    assert len(result["free_running"]) == job["free_run_trials"] == 1
    trial = result["free_running"][0]
    assert trial["sampling"] == {**case["metadata"]["sampling"], "seed": job["seed"]}
    count = len(trial["token_ids"])
    assert 0 < count <= job["free_run_tokens"]
    counters = trial["decode_timing"]["trace_counters_delta"]
    assert all(counters[k] == count - 1 for k in ("model_replays", "sample_replays", "readbacks"))
    assert all(counters[k] == 0 for k in ("captures", "token_refreshes", "position_refreshes", "page_refreshes"))
    assert trial["decode_timing"]["decode_tokens"] == count - 1
    rows.append(
        {
            "name": job["name"],
            "result_sha256": sha(path),
            "fixture_sha256": sha(fixture),
            "prompt_tokens": len(case["prompt_token_ids"]),
            "generated_tokens": count,
            "cap": job["free_run_tokens"],
            "seed": job["seed"],
        }
    )
out = {
    "status": "passed" if not missing else "partial_pass",
    "expected": len(jobs),
    "checked": len(rows),
    "missing": missing,
    "manifest_sha256": sha(manifest),
    "precision_sha256": sha(HERE / "head_bf16_hifi4_precision.json"),
    "rows": rows,
    "scope": "Runtime precision, fixture, seed, caps and trace counters; semantic review is separate.",
}
(HERE / "runtime-integrity-audit.json").write_text(json.dumps(out, indent=2) + "\n")
print(out["status"], len(rows), "/", len(jobs))
