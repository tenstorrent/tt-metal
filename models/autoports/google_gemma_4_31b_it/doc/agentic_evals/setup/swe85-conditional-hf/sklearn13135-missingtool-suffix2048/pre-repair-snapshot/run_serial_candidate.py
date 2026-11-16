#!/usr/bin/env python3
"""Inert-by-default one-mode runner for the Sklearn suffix2048 diagnostic."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd().resolve()
HERE = (
    ROOT
    / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup/swe85-conditional-hf/sklearn13135-missingtool-suffix2048"
)
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
STOP = HERE / "STOP_BEFORE_LAUNCH"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def check_hash(path, expected, label):
    path = Path(path)
    require(path.is_file(), f"missing {label}: {path}")
    require(sha(path) == expected, f"{label} hash mismatch: {path}")


def check_source_and_artifacts():
    inventory = read(HERE / "source-hashes.json")
    files = inventory.get("files")
    require(isinstance(files, dict) and files, "source inventory missing files")
    for rel, digest in files.items():
        check_hash(ROOT / rel, digest, f"source {rel}")
    artifacts = read(HERE / "artifact-hashes.json").get("files", {})
    require(artifacts, "candidate artifact hash inventory is empty")
    for rel, digest in artifacts.items():
        check_hash(HERE / rel, digest, f"candidate artifact {rel}")
    return inventory, artifacts


def check_candidate_and_policy():
    require((ROOT / "bringup/run_contract.json").is_file(), "run from repository root")
    source_inventory, artifacts = check_source_and_artifacts()
    commands = read(HERE / "commands.json")
    provenance = read(HERE / "prefix-provenance.json")
    require(
        provenance["suffix_tokens"] == 2048 and provenance["chosen_tokens"] == 64,
        "candidate provenance dimensions changed",
    )
    require(
        provenance["seeds"] == [42, 43, 44] and provenance["free_run_cap_tokens"] == 1024,
        "candidate seed/cap list changed",
    )
    for mode in ("served", "full"):
        spec = commands[mode]
        manifest_path = ROOT / spec["manifest"]
        check_hash(manifest_path, spec["manifest_sha256"], f"{mode} manifest")
        check_hash(ROOT / spec["runner"], spec["runner_sha256"], f"{mode} runner")
        check_hash(ROOT / spec["build_kwargs_file"], spec["build_kwargs_sha256"], f"{mode} build kwargs")
        check_hash(spec["precision_config"], spec["precision_config_sha256"], f"{mode} precision config")
        cfg = read(spec["precision_config"])
        kwargs = read(ROOT / spec["build_kwargs_file"])
        require(
            (ROOT / kwargs.get("precision_config", "")).resolve() == Path(spec["precision_config"]).resolve(),
            f"{mode} kwargs point elsewhere",
        )
        require(cfg.get("config_id") == spec["expected_config_id"], f"{mode} loaded config_id changed")
        require(spec["execution_command"][0:3] == ["timeout", "1200", "env"], f"{mode} timeout/env changed")
        require(
            any(x.startswith("PYTHONPATH=") and str(ROOT) in x for x in spec["execution_command"]),
            f"{mode} missing explicit repository PYTHONPATH",
        )
        require(
            "READINESS_TRACE_REGION_SIZE=134217728" in spec["execution_command"], f"{mode} trace region setting changed"
        )
        require("--build-kwargs-file" in spec["execution_command"], f"{mode} command missing policy arg")
        pos = spec["execution_command"].index("--build-kwargs-file")
        require(
            spec["execution_command"][pos + 1] == str(ROOT / spec["build_kwargs_file"]),
            f"{mode} command policy path differs from loaded kwargs",
        )
        jobs = read(manifest_path)
        require(len(jobs) == 3, f"{mode} must contain exactly three trials")
        seeds = []
        for job in jobs:
            require(job["free_run_tokens"] == 1024 and job["free_run_trials"] == 1, f"{mode} cap/trial changed")
            fixture_path = ROOT / job["case"]
            fixture = read(fixture_path)
            md = fixture["metadata"]
            seeds.append(md["seed"])
            require(len(fixture["prompt_token_ids"]) == 2048, f"{mode} fixture context length changed")
            require(
                json_sha(fixture["prompt_token_ids"]) == provenance["suffix_token_ids_sha256"],
                f"{mode} fixture is not the exact pinned source suffix",
            )
            require(len(fixture["continuations"]["chosen"]) == 64, f"{mode} chosen branch length changed")
            require(
                json_sha(fixture["continuations"]["chosen"]) == provenance["chosen_token_ids_sha256"],
                f"{mode} chosen branch changed",
            )
            require(
                md["prompt_suffix_start_index"] == 15347 and md["prompt_suffix_end_index_exclusive"] == 17395,
                f"{mode} suffix offsets changed",
            )
            require(
                md["prompt_suffix_token_ids_sha256"] == provenance["suffix_token_ids_sha256"],
                f"{mode} fixture suffix metadata changed",
            )
            require(
                md["diagnostic_mode"] == mode and md["diagnostic_seed"] == md["seed"], f"{mode} fixture labels mismatch"
            )
        require(sorted(seeds) == [42, 43, 44], f"{mode} seed coverage changed")
    return source_inventory, artifacts, commands


def verify_admission():
    admission_path = HERE / "admission.json"
    require(admission_path.is_file(), "root admission.json is required")
    admission = read(admission_path)
    require(admission.get("status") == "admitted", "candidate is not root-admitted")
    for field, path in (
        ("proposal_sha256", HERE / "proposal.json"),
        ("driver_sha256", Path(__file__).resolve()),
        ("source_inventory_sha256", HERE / "source-hashes.json"),
        ("commands_sha256", HERE / "commands.json"),
        ("forecast_sha256", HERE / "forecast.json"),
        ("artifact_hashes_sha256", HERE / "artifact-hashes.json"),
    ):
        require(admission.get(field) == sha(path), f"admission pin mismatch: {field}")
    return admission


def check_predecessors():
    p2 = D / "setup/swe85-precision-bisect/head-bf16-hifi4-fixed108-proposal"
    # Use the exact immutable pins captured at preparation time.
    proposal = read(HERE / "proposal.json")["predecessors"]
    check_hash(p2 / "admission.json", proposal["P2_admission_sha256"], "P2 admission")
    check_hash(p2 / "completion.json", proposal["P2_completion_sha256"], "P2 completion")
    check_hash(p2 / "root-forecast-review.json", proposal["P2_root_forecast_review_sha256"], "P2 forecast review")
    p2c = read(p2 / "completion.json")
    require(p2c.get("exit_code") == 0, "P2 is not terminal-success")
    sk = HERE.parent / "sklearn13135-missingtool-suffix1024"
    for file, key in (
        ("admission.json", "SK1024_admission_sha256"),
        ("completion-summary.json", "SK1024_completion_summary_sha256"),
        ("root-served-review.json", "SK1024_root_served_review_sha256"),
        ("root-full-review.json", "SK1024_root_full_review_sha256"),
    ):
        check_hash(sk / file, proposal[key], f"suffix1024 {file}")
    summary = read(sk / "completion-summary.json")
    require(
        summary.get("original_full_missing_tool_reproduced") is False,
        "suffix1024 predecessor no longer supports this diagnostic",
    )
    require(
        summary.get("status") == "completed_root_and_independent_triage_reconciled",
        "suffix1024 is not fully reconciled",
    )
    # Root writes this clearance only after the serial HF queue has completed.
    clearance_path = HERE / "predecessors/serial-hf-clearance.json"
    require(clearance_path.is_file(), "root serial HF clearance is required")
    clearance = read(clearance_path)
    require(
        clearance.get("reviewer") == "root" and clearance.get("accepted") is True,
        "root has not cleared the serial HF queue",
    )
    for key in ("primary_hf", "sklearn_secondary_hf"):
        item = clearance.get(key, {})
        require(item.get("status") == "completed_root_reviewed", f"{key} is not root-reviewed complete")
        check_hash(ROOT / item["completion_path"], item["completion_sha256"], f"{key} completion")
    return clearance


def no_active_hf():
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        require("precision_repro_hf.py" not in args, f"active HF process {proc.name}; refusing overlap")


def no_device_owner():
    sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
    import benchmark_server

    owners = benchmark_server.device_owners()
    require(not owners, f"device owners present: {owners}")
    return owners


def verify_mode_outputs(mode, commands):
    spec = commands[mode]
    output = ROOT / spec["output_dir"]
    load = output / "load.json"
    require(load.is_file(), f"{mode} load record missing")
    cfg = read(spec["precision_config"])
    loaded_cfg = read(load).get("build_kwargs", {}).get("precision_config", "")
    require(
        (ROOT / loaded_cfg).resolve() == Path(spec["precision_config"]).resolve(),
        f"{mode} load record used a different config",
    )
    expected = sorted([42, 43, 44])
    actual = []
    manifest = read(ROOT / spec["manifest"])
    for job in manifest:
        matches = list(output.glob(Path(job["case"]).name.replace(".private.json", ".json")))
        require(len(matches) == 1, f"{mode} expected exactly one output for {job['name']}")
        row = read(matches[0])
        actual_seed = row["free_running"][0]["sampling"]["seed"]
        actual.append(actual_seed)
        require(row.get("runtime_precision") == cfg, f"{mode} runtime precision mismatch for seed {actual_seed}")
        require(len(row["free_running"][0]["token_ids"]) <= 1024, f"{mode} generated tokens exceed cap")
        require(row.get("metadata", {}).get("diagnostic_mode") == mode, f"{mode} output mode metadata mismatch")
    require(sorted(actual) == expected, f"{mode} output seed coverage mismatch: {actual}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("served", "full"), required=True)
    ap.add_argument(
        "--validate-only", action="store_true", help="host-only pins/fixture/command checks; launches nothing"
    )
    ap.add_argument("--execute", action="store_true", help="run one root-admitted mode after all gates")
    args = ap.parse_args()
    require(args.validate_only != args.execute, "choose exactly one of --validate-only or --execute")
    source_inventory, artifacts, commands = check_candidate_and_policy()
    if args.validate_only:
        require(STOP.is_file(), "STOP_BEFORE_LAUNCH must remain for inert validation")
        print(
            json.dumps(
                {
                    "status": "host_static_validation_passed_no_launch",
                    "mode": args.mode,
                    "source_files": len(source_inventory["files"]),
                    "candidate_files": len(artifacts),
                },
                indent=2,
            )
        )
        return 0
    verify_admission()
    require(not STOP.exists(), "STOP_BEFORE_LAUNCH is present")
    clearance = check_predecessors()
    no_active_hf()
    owners_before = no_device_owner()
    if args.mode == "full":
        served = HERE / "completion-served.json"
        review = HERE / "predecessors/served-forecast-review.json"
        require(
            served.is_file() and review.is_file(), "served completion and root forecast review required before full"
        )
        sc = read(served)
        sr = read(review)
        require(
            sc.get("status") == "completed" and sc.get("exit_code") == 0, "served mode did not complete successfully"
        )
        ranges = read(HERE / "forecast.json")["mode_estimates"]["served"]
        allowed_ranges = [
            ranges["early_stop_scenario"]["range_seconds"],
            ranges["full_1024_cap_scenario"]["range_seconds"],
        ]
        selected_range = sr.get("forecast_range_seconds")
        require(
            sr.get("reviewer") == "root"
            and sr.get("accepted") is True
            and sr.get("completion_sha256") == sha(served)
            and sr.get("admission_sha256") == sha(HERE / "admission.json")
            and selected_range in allowed_ranges
            and sr.get("reviewed_elapsed_seconds") == sc.get("elapsed_seconds"),
            "root served forecast review does not bind current completion/admission/forecast range",
        )
        lo, hi = selected_range
        expected_status = "in_range" if lo <= sc["elapsed_seconds"] <= hi else "accepted_miss"
        require(
            sr.get("forecast_status") == expected_status,
            "root served review status does not match its selected scenario range",
        )
    spec = commands[args.mode]
    launch_path = HERE / f"launch-{args.mode}.json"
    completion_path = HERE / f"completion-{args.mode}.json"
    require(not launch_path.exists() and not completion_path.exists(), "mode output record already exists")
    started = time.time()
    launch = {
        "status": "running",
        "mode": args.mode,
        "started_unix": started,
        "admission_sha256": sha(HERE / "admission.json"),
        "command": spec["execution_command"],
        "serial_hf_clearance_sha256": sha(HERE / "predecessors/serial-hf-clearance.json"),
        "device_owners_before": owners_before,
    }
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")
    result = subprocess.run(spec["execution_command"], cwd=ROOT, check=False)
    elapsed = time.time() - started
    completion = {
        "status": "completed" if result.returncode == 0 else "failed",
        "mode": args.mode,
        "exit_code": result.returncode,
        "elapsed_seconds": elapsed,
        "admission_sha256": sha(HERE / "admission.json"),
        "launch_sha256": sha(launch_path),
        "forecast_scenarios": read(HERE / "forecast.json")["mode_estimates"][args.mode],
    }
    if result.returncode == 0:
        verify_mode_outputs(args.mode, commands)
        completion["output_status"] = "all_expected_outputs_and_runtime_precision_verified"
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"candidate gate failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
