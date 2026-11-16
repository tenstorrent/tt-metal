#!/usr/bin/env python3
"""Run exactly one admitted Sklearn secondary-loop HF group; inert unless --execute --group N."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[8]
assert (ROOT / "bringup/run_contract.json").is_file(), "repository root resolution failed"
OUT = (
    ROOT
    / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup/swe85-conditional-hf/sklearn13135-secondary-loops-suffix512"
)
ADMISSION = OUT / "admission.json"
PROPOSAL = OUT / "proposal.json"
COMMANDS = OUT / "commands.json"
INTEGRITY = OUT / "source-integrity.json"
FORECAST = OUT / "forecast.json"
STOP = OUT / "STOP_BEFORE_LAUNCH"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def verify(path, expected, label):
    path = Path(path)
    assert path.is_file(), f"missing {label}: {path}"
    actual = sha(path)
    assert actual == expected, f"{label} hash changed: expected {expected}, got {actual}"
    return actual


def verify_inventory():
    proposal, commands, integrity, forecast = map(read, (PROPOSAL, COMMANDS, INTEGRITY, FORECAST))
    assert proposal["status"] == "candidate_only_not_admitted_not_run"
    assert commands["status"] == "candidate_only_not_admitted_no_launch"
    assert integrity["hf_revision"] == "842da3794eaa0b77d5f08bae87a17459d91ff475"
    assert len(commands["groups"]) == 2 and len(forecast["groups"]) == 2
    assert commands["source_integrity"] == {"path": "source-integrity.json", "sha256": sha(INTEGRITY)}
    assert proposal["source_integrity"] == {"path": "source-integrity.json", "sha256": sha(INTEGRITY)}
    assert proposal["candidate_hashes"]["commands_sha256"] == sha(COMMANDS)
    assert proposal["candidate_hashes"]["forecast_sha256"] == sha(FORECAST)
    assert proposal["candidate_hashes"]["source_integrity_sha256"] == sha(INTEGRITY)
    assert proposal["candidate_hashes"]["runner_sha256"] == integrity["runner"]["sha256"]
    driver = integrity["execution_driver"]
    verify(ROOT / driver["path"], driver["sha256"], "execution driver")
    assert driver["path"] == Path(__file__).resolve().relative_to(ROOT).as_posix()
    for rel, digest in integrity["external_files"].items():
        path = Path(rel) if Path(rel).is_absolute() else ROOT / rel
        verify(path, digest, "pinned source " + rel)
    corrected = integrity["corrected_served_arm"]
    corrected_dir = ROOT / corrected["directory"]
    assert corrected["config_id"] == "stage12_head_bfp8_lofi"
    assert read(corrected_dir / "admission.json").get("status") == "root_admitted"
    assert read(corrected_dir / "root-review.json").get("status") == "completed_root_reviewed"
    assert sha(corrected_dir / "admission.json") == corrected["admission_sha256"]
    assert sha(corrected_dir / "root-review.json") == corrected["root_review_sha256"]
    for seed in (42, 43, 44):
        name = f"anomaly.scikit-learn__scikit-learn-13135.call29.onset.suffix512.seed{seed}.json"
        results_dir = Path(corrected["results_dir"])
        if not results_dir.is_absolute():
            results_dir = ROOT / results_dir
        result = read(results_dir / name)
        assert result.get("runtime_precision", {}).get("config_id") == "stage12_head_bfp8_lofi"
    for rel, digest in integrity["candidate_manifests"].items():
        verify(OUT / rel, digest, "candidate manifest " + rel)
    assert set(proposal["candidate_hashes"]["manifest_sha256s"]) == set(integrity["candidate_manifests"])
    assert proposal["candidate_hashes"]["manifest_sha256s"] == integrity["candidate_manifests"]
    seen = []
    for group, forecast_row in zip(commands["groups"], forecast["groups"]):
        order = group["serial_order"]
        seen.append(order)
        assert forecast_row["serial_order"] == order and forecast_row["seed"] == group["seed_group"]
        assert group["timeout_seconds"] == forecast_row["timeout_seconds"] == 3600
        assert group["manifest_sha256"] == sha(OUT / group["manifest"])
        jobs = read(OUT / group["manifest"])
        assert len(jobs) == 1 and jobs[0]["name"] == group["case_names"][0]
        assert (
            jobs[0]["branches"] == ["chosen"] and jobs[0]["free_run_tokens"] == 512 and jobs[0]["free_run_trials"] == 1
        )
        case = read(ROOT / jobs[0]["case"])
        assert len(case["prompt_token_ids"]) == 512
        assert len(case["continuations"]["chosen"]) == 64
        argv = group["command"]
        assert argv[0] == "timeout" and argv[1] == "3600"
        assert "HF_HUB_OFFLINE=1" in argv and "HF_DATASETS_OFFLINE=1" in argv
        for env in ("OMP_NUM_THREADS=16", "MKL_NUM_THREADS=16", "OPENBLAS_NUM_THREADS=16"):
            assert env in argv
        assert argv[argv.index("--cases") + 1] == str(OUT / group["manifest"])
        assert argv[argv.index("--output") + 1] == str(ROOT / group["output_dir"])
        assert argv[argv.index("--tokens") + 1] == "512"
        assert argv[argv.index("--prefill-chunk") + 1] == "512"
        assert argv[argv.index("--seeds") + 1] == str(group["seed_group"])
        assert argv.count("--compare-dir") == 2
        assert "--validate-only" not in argv, "saved generation command must not be used as validation"
        assert Path(group["compare_dirs"][0], jobs[0]["name"] + ".json").is_file()
        assert Path(group["compare_dirs"][1], jobs[0]["name"] + ".json").is_file()
        assert forecast_row["range_seconds"] == [980, 1780]
    assert seen == [1, 2]
    return proposal, commands, integrity, forecast


def verify_admission():
    admission = read(ADMISSION)
    assert admission.get("status") == "admitted", "root admission for this exact candidate is required"
    proposal, commands, integrity, forecast = verify_inventory()
    pins = {
        "proposal_sha256": (PROPOSAL, sha(PROPOSAL)),
        "commands_sha256": (COMMANDS, sha(COMMANDS)),
        "source_integrity_sha256": (INTEGRITY, sha(INTEGRITY)),
        "forecast_sha256": (FORECAST, sha(FORECAST)),
        "runner_sha256": (ROOT / integrity["runner"]["path"], integrity["runner"]["sha256"]),
    }
    for key, (path, digest) in pins.items():
        assert admission.get(key) == digest, f"admission {key} mismatch"
        verify(path, digest, key)
    assert admission.get("manifest_sha256s") == proposal["candidate_hashes"]["manifest_sha256s"]
    corrected = integrity["corrected_served_arm"]
    assert admission.get("corrected_served_arm_admission_sha256") == corrected["admission_sha256"]
    assert admission.get("corrected_served_arm_root_review_sha256") == corrected["root_review_sha256"]
    return admission, proposal, commands, integrity, forecast, sha(ADMISSION)


def proc_table():
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            if not cmd:
                continue
            rows.append((int(entry.name), cmd, entry))
        except OSError:
            continue
    return rows


def preflight():
    pid = os.getpid()
    hf = [{"pid": p, "cmdline": cmd[:400]} for p, cmd, _ in proc_table() if p != pid and "precision_repro_hf.py" in cmd]
    assert not hf, f"HF reference process is active: {hf}"
    owners = []
    for p, cmd, entry in proc_table():
        if p == pid:
            continue
        for fd in (entry / "fd").glob("*"):
            try:
                if os.readlink(fd).startswith("/dev/tenstorrent/"):
                    owners.append({"pid": p, "cmdline": cmd[:400]})
                    break
            except OSError:
                continue
    assert not owners, f"device owners are active: {owners}"
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    raw = fields.get("MemAvailable", "")
    assert raw.endswith("kB"), "cannot read MemAvailable"
    mem = int(raw[:-2].strip()) * 1024
    assert mem >= 100 * 1024**3, f"MemAvailable below100GiB: {mem} bytes"
    return {
        "active_hf_processes": hf,
        "device_owners": owners,
        "mem_available_bytes": mem,
        "required_mem_available_bytes": 100 * 1024**3,
    }


def verify_prior_group_one(admission_sha, commands, forecast):
    launch_path = OUT / "launch-group-01.json"
    completion_path = OUT / "completion-group-01.json"
    assert launch_path.is_file() and completion_path.is_file(), "group1 launch/completion missing"
    launch, completion = read(launch_path), read(completion_path)
    assert launch.get("candidate_admission_sha256") == admission_sha
    assert completion.get("candidate_admission_sha256") == admission_sha
    assert launch.get("status") == "launched" and completion.get("status") == "completed"
    assert completion.get("exit_code") == 0
    group = commands["groups"][0]
    assert launch.get("command") == group["command"] and completion.get("command") == group["command"]
    assert completion.get("launch_sha256") == sha(launch_path)
    for rel, digest in completion.get("result_sha256s", {}).items():
        verify(ROOT / rel, digest, "group1 raw HF result")
    assert completion.get("result_sha256s"), "group1 raw result hashes missing"
    frow = forecast["groups"][0]
    assert completion.get("forecast_range_seconds") == frow["range_seconds"]
    in_range = frow["range_seconds"][0] <= completion["elapsed_seconds"] <= frow["range_seconds"][1]
    acceptance = None
    if not in_range:
        acceptance_path = OUT / "forecast-miss-accepted-group-01.json"
        assert acceptance_path.is_file(), "group1 forecast miss requires root hash-bound acceptance"
        acceptance = read(acceptance_path)
        for key, value in {
            "status": "root_accepted_forecast_miss",
            "reviewer": "root",
            "accepted": True,
            "completion_sha256": sha(completion_path),
            "measured_seconds": completion["elapsed_seconds"],
            "forecast_range_seconds": frow["range_seconds"],
            "admission_sha256": admission_sha,
            "serial_order": 1,
        }.items():
            assert acceptance.get(key) == value, f"group1 forecast acceptance mismatch: {key}"
    review_path = OUT / "root-review-group-01.json"
    assert review_path.is_file(), "group1 raw-output/forecast review required before group2"
    review = read(review_path)
    expected = {
        "status": "root_reviewed_and_authorized_next_group",
        "reviewer": "root",
        "candidate_admission_sha256": admission_sha,
        "launch_sha256": sha(launch_path),
        "completion_sha256": sha(completion_path),
        "forecast_gate": "hit" if in_range else "root_accepted_miss",
    }
    for key, value in expected.items():
        assert review.get(key) == value, f"group1 root review mismatch: {key}"
    assert review.get("result_sha256s") == completion["result_sha256s"]
    if acceptance:
        assert review.get("forecast_miss_acceptance_sha256") == sha(OUT / "forecast-miss-accepted-group-01.json")
    return {
        "launch_sha256": sha(launch_path),
        "completion_sha256": sha(completion_path),
        "review_sha256": sha(review_path),
        "forecast_gate": "hit" if in_range else "root_accepted_miss",
    }


def verify_outputs(group):
    output = ROOT / group["output_dir"]
    jobs = read(OUT / group["manifest"])
    results = {}
    for job in jobs:
        path = output / (job["name"] + ".hf.private.json")
        assert path.is_file(), f"missing HF result: {path}"
        result = read(path)
        assert result.get("status") == "complete"
        assert result.get("revision") == "842da3794eaa0b77d5f08bae87a17459d91ff475"
        assert result.get("prompt_tokens") == 512
        assert len(result.get("teacher_forced_chosen", {}).get("positions", [])) == 64
        trials = result.get("results", [])
        assert len(trials) == 1 and trials[0].get("seed") == group["seed_group"]
        assert trials[0].get("requested_tokens") == 512
        assert set(result.get("tt_comparisons", {})) == set(group["compare_dirs"])
        results[path.relative_to(ROOT).as_posix()] = sha(path)
    return results


def complete_validation(completion, group):
    """Keep child status separate from validation status and select CLI exit code."""
    if completion["exit_code"] != 0:
        completion["status"] = "failed"
        completion["output_validation_status"] = "not_run_child_failed"
        completion["driver_exit_code"] = completion["exit_code"] or 1
        return completion
    try:
        completion["result_sha256s"] = verify_outputs(group)
    except Exception as exc:
        completion["status"] = "output_validation_failed"
        completion["output_validation_status"] = "failed"
        completion["output_validation_error"] = f"{type(exc).__name__}: {exc}"
        completion["driver_exit_code"] = 2
        return completion
    completion["status"] = "completed"
    completion["output_validation_status"] = "passed"
    completion["driver_exit_code"] = 0
    return completion


def execute_one(group_index):
    admission, proposal, commands, integrity, forecast, admission_sha = verify_admission()
    assert not STOP.exists(), "STOP_BEFORE_LAUNCH is set"
    group = next((g for g in commands["groups"] if g["serial_order"] == group_index), None)
    assert group is not None
    assert not (OUT / f"launch-group-{group_index:02d}.json").exists()
    assert not (OUT / f"completion-group-{group_index:02d}.json").exists()
    prior = None
    if group_index == 2:
        prior = verify_prior_group_one(admission_sha, commands, forecast)
    checks = preflight()
    output = ROOT / group["output_dir"]
    assert not output.exists() or not any(output.iterdir()), f"refusing non-empty output directory {output}"
    launch_path = OUT / f"launch-group-{group_index:02d}.json"
    completion_path = OUT / f"completion-group-{group_index:02d}.json"
    launch = {
        "status": "launched",
        "serial_order": group_index,
        "seed": group["seed_group"],
        "candidate_admission_sha256": admission_sha,
        "proposal_sha256": sha(PROPOSAL),
        "commands_sha256": sha(COMMANDS),
        "source_integrity_sha256": sha(INTEGRITY),
        "forecast_sha256": sha(FORECAST),
        "manifest_sha256": group["manifest_sha256"],
        "command": group["command"],
        "preflight": checks,
        "prior_group_review": prior,
        "start_unix": time.time(),
        "timeout_seconds": group["timeout_seconds"],
    }
    write_exclusive(launch_path, launch)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_path = OUT / f"group-{group_index:02d}.log"
    try:
        with log_path.open("x") as log:
            result = subprocess.run(
                group["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=group["timeout_seconds"] + 30
            )
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    end = time.time()
    completion = {
        **launch,
        "end_unix": end,
        "elapsed_seconds": end - launch["start_unix"],
        "exit_code": exit_code,
        "log_sha256": sha(log_path),
        "launch_sha256": sha(launch_path),
    }
    complete_validation(completion, group)
    row = forecast["groups"][group_index - 1]
    completion["forecast_range_seconds"] = row["range_seconds"]
    measured = completion["elapsed_seconds"]
    completion["forecast_gate"] = (
        "hit" if row["range_seconds"][0] <= measured <= row["range_seconds"][1] else "root_review_required"
    )
    write_exclusive(completion_path, completion)
    return completion


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--group", type=int, choices=(1, 2))
    args = parser.parse_args()
    verify_inventory()
    if args.validate_only:
        assert args.group is None
        print("candidate static checks passed; no command execution, model load, or device query")
        return
    if not args.execute or args.group is None:
        raise SystemExit("inert by default; use --validate-only or root-admitted --execute --group N")
    completion = execute_one(args.group)
    if completion["driver_exit_code"] != 0:
        raise SystemExit(completion["driver_exit_code"])


if __name__ == "__main__":
    main()
