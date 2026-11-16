"""Prepare unadmitted suffix controls using token IDs only; never load a model/tokenizer."""
import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=D / "setup/swe85-tt-execution-plan/anomaly-cases.json")
parser.add_argument("--output-dir", type=Path, default=D / "anomalies/swe85-optional-suffix512")
parser.add_argument("--expected-count", type=int, default=21)
args = parser.parse_args()
SOURCE = args.source
OUT = args.output_dir
OUT.mkdir(exist_ok=True)
OUT.chmod(0o700)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    encoded = (json.dumps(data, indent=2) + "\n").encode()
    if path.exists():
        assert path.read_bytes() == encoded, f"Refusing to overwrite existing artifact: {path}"
    else:
        path.write_bytes(encoded)
        path.chmod(0o600)


source_hash = sha(SOURCE)
source_jobs = json.loads(SOURCE.read_text())
assert (
    len(source_jobs) == args.expected_count
), "Review source-roster changes explicitly before generating another snapshot."
jobs, rows, skipped = [], [], []
source_hashes = {}
for job in source_jobs:
    source = Path(job["case"])
    original = json.loads(source.read_text())
    source_hashes[str(source)] = sha(source)
    ids = original["prompt_token_ids"]
    chosen = original["continuations"]["chosen"]
    assert 0 < len(chosen) <= 64
    assert (
        original["metadata"].get("seed") == 42
    ), "Missing or changed diagnostic seed: review explicitly; never invent it."
    if len(ids) <= 512:
        skipped.append(
            {
                "source_job": job["name"],
                "source_case": str(source),
                "source_sha256": sha(source),
                "reason": "Already at most512 prompt IDs; no duplicate suffix case created.",
            }
        )
        continue
    start, end = len(ids) - 512, len(ids)
    new = copy.deepcopy(original)
    new["prompt_token_ids"] = ids[start:end]
    new["metadata"]["optional_suffix_control"] = {
        "status": "candidate_only_not_admitted_not_executed",
        "source_case": str(source),
        "source_case_sha256": sha(source),
        "source_job": job["name"],
        "source_roster": str(SOURCE),
        "source_roster_sha256": source_hash,
        "source_prompt_tokens": len(ids),
        "suffix_start_inclusive": start,
        "suffix_end_exclusive": end,
        "suffix_tokens": 512,
        "source_metadata_preserved": True,
        "continuation_token_ids_unchanged": True,
        "native_equivalence_claim": False,
        "conditioning_limit": "Drops original context and potentially starts inside a turn/tokenized protocol. No new BOS/template text inserted; exactly the final512 source conditioning IDs.",
        "admission_gate": "Reviewer and root approval before any execution. A shortened HF comparison is admissible only after this identical suffix reproduces the relevant anomaly on matched TT; full-TT anomaly persistence is required before conditional HF escalation.",
        "reproduction_judgment": "Evaluate mechanical degeneration and the source-specific semantic/protocol motif. Mere continuation difference is not reproduction. A non-reproducing suffix cannot explain the full-prefix anomaly.",
        "free_run": {"token_cap": 512, "trials": 3, "seeds": [42, 43, 44]},
    }
    target = OUT / (job["name"] + ".suffix512.private.json")
    save(target, new)
    assert new["prompt_token_ids"] == ids[-512:]
    assert new["continuations"] == original["continuations"]
    assert {k: v for k, v in new["metadata"].items() if k != "optional_suffix_control"} == original["metadata"]
    newjob = {
        "name": job["name"] + ".optional-suffix512",
        "case": str(target),
        "branches": ["chosen"],
        "free_run_tokens": 512,
        "free_run_trials": 3,
    }
    jobs.append(newjob)
    rows.append(
        {
            "source_job": job["name"],
            "source_case": str(source),
            "source_sha256": sha(source),
            "case": str(target),
            "case_sha256": sha(target),
            "source_prompt_tokens": len(ids),
            "source_suffix_range": {"start_inclusive": start, "end_exclusive": end},
            "prompt_tokens": 512,
            "teacher_forcing_positions": len(chosen),
            "free_run_token_cap": 1536,
            "seeds": [42, 43, 44],
            "sampling": original["metadata"].get("sampling"),
            "all_other_source_metadata_and_continuations_unchanged": True,
        }
    )
save(OUT / "candidate-cases.json", jobs)
validation = []
for mode, config in [("served", "head_bfp8_build_kwargs.json"), ("full", "full_precision_build_kwargs.json")]:
    cfg = D / "setup" / config
    argv = [
        "python_env/bin/python",
        "-m",
        "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
        "--cases",
        str(OUT / "candidate-cases.json"),
        "--build-kwargs-file",
        str(cfg),
        "--output-dir",
        str(OUT / ("UNADMITTED-" + mode)),
        "--validate-only",
    ]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    validation.append(
        {
            "mode": mode,
            "config": str(cfg),
            "config_sha256": sha(cfg),
            "command": argv,
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
        }
    )
assert sha(SOURCE) == source_hash
assert all(sha(Path(p)) == h for p, h in source_hashes.items())
tf = sum(r["teacher_forcing_positions"] for r in rows)
free = sum(r["free_run_token_cap"] for r in rows)
prefill = sum(r["prompt_tokens"] * 4 for r in rows)  # one chosen TF + three free-run trials
fullproxy = sum(r["source_prompt_tokens"] * 4 for r in rows)
manifest = {
    "status": "optional_candidate_only_requires_reviewer_and_root_admission",
    "source_roster": str(SOURCE),
    "source_roster_sha256": source_hash,
    "source_count": len(source_jobs),
    "candidate_count": len(jobs),
    "deduplicated_existing_short": skipped,
    "cases": rows,
    "cpu_schema_validation": validation,
    "inference_executed": False,
    "admitted_manifests_modified": False,
    "original_source_hashes_unchanged": True,
    "added_work_per_precision_if_admitted": {
        "teacher_forcing_positions": tf,
        "free_run_token_cap": free,
        "prefill_input_tokens_proxy": prefill,
        "prefill_calls_proxy": 4 * len(rows),
        "same_jobs_with_original_prefix_prefill_input_tokens_proxy": fullproxy,
        "ratio_suffix_to_original_prefill_tokens": prefill / fullproxy if fullproxy else None,
    },
    "added_work_two_TT_precisions_if_both_admitted": {
        "teacher_forcing_positions": 2 * tf,
        "free_run_token_cap": 2 * free,
        "prefill_input_tokens_proxy": 2 * prefill,
    },
    "estimate_limits": "Token-count proxy only, not wall-time or measured throughput. Free-run values are caps and may stop early. No HF execution plan or host weight load authorized. Full-prefix originals remain authoritative; no effect on admitted manifests.",
    "gate": "Matched TT suffix must reproduce source anomaly before conditional HF suffix comparison; negative suffix result is inconclusive for full prefix.",
}
save(OUT / "candidate-manifest.json", manifest)
print(
    json.dumps(
        {
            "candidate_count": len(jobs),
            "already_short": len(skipped),
            "added_work_per_precision": manifest["added_work_per_precision_if_admitted"],
        }
    )
)
