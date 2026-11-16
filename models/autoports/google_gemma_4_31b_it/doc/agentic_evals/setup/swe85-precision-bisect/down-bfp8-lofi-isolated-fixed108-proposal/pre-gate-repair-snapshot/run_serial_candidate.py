#!/usr/bin/env python3
"""Inert-until-admitted serial gate for the isolated down BFP8 LoFi panel."""
import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd().resolve()
assert (ROOT / "bringup/run_contract.json").is_file(), "run from repository root"
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
P1 = D / "setup/swe85-precision-bisect/head-bfp8-hifi2-first-panel-after-w"
P2 = D / "setup/swe85-precision-bisect/head-bf16-hifi4-fixed108-proposal"
SK = D / "setup/swe85-conditional-hf/sklearn13135-missingtool-suffix512"
P2_ADMISSION_SHA = "74bef162c97e23e882612eb85dae365f32c9da1f13ba4b9b8e1e1f8f582ad1be"
STOP = HERE / "STOP_BEFORE_LAUNCH"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_review(review, expected_range, elapsed, *, contention=False):
    require(review.get("reviewer") == "root" and review.get("accepted") is True, "root review missing or not accepted")
    require(review.get("forecast_range_seconds") == list(expected_range), "review forecast range mismatch")
    require(review.get("reviewed_elapsed_seconds") == elapsed, "review elapsed mismatch")
    in_range = expected_range[0] <= elapsed <= expected_range[1]
    require(
        review.get("forecast_status") == ("in_range" if in_range else "accepted_miss"),
        "review forecast status does not match measured elapsed",
    )
    if contention:
        require(
            review.get("contention_disclosed") is True and review.get("speed_ranking_used") is False,
            "P2 contention must be disclosed and excluded from policy speed ranking",
        )
    return in_range


def source_inventory(path):
    inv = read(path)
    rows = inv.get("files", {})
    require(isinstance(rows, dict) and rows, f"empty/malformed source inventory: {path}")
    for label, item in rows.items():
        require(
            isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str),
            f"bad inventory row {label}",
        )
        source = ROOT / item["path"]
        require(source.is_file() and sha(source) == item["sha256"], f"source pin mismatch: {item['path']}")
    return inv


def candidate_gate(admission_required=False):
    inventory = source_inventory(HERE / "source-hashes.json")
    admission = None
    if admission_required:
        ap = HERE / "admission.json"
        require(ap.is_file(), "explicit root admission required")
        admission = read(ap)
        require(admission.get("status") == "admitted", "candidate admission is not admitted")
        for field, path in (
            ("proposal_sha256", HERE / "proposal.json"),
            ("driver_sha256", Path(__file__)),
            ("source_inventory_sha256", HERE / "source-hashes.json"),
            ("command_sha256", HERE / "command.json"),
            ("forecast_sha256", HERE / "forecast.json"),
            ("precision_sha256", HERE / "down_bfp8_lofi_isolated_precision.json"),
            ("build_kwargs_sha256", HERE / "down_bfp8_lofi_isolated_build_kwargs.json"),
            ("manifest_sha256", HERE / "candidate-cases.json"),
        ):
            require(sha(path) == admission.get(field), f"candidate admission pin mismatch: {field}")
    return admission, inventory


def validate_static():
    inventory = source_inventory(HERE / "source-hashes.json")
    manifest_path = HERE / "candidate-cases.json"
    jobs = read(manifest_path)
    require(len(jobs) == 108, "fixed108 manifest must contain 108 jobs")
    seeds = {42, 43, 44}
    pairs = [(j["source_onset"], j["seed"]) for j in jobs]
    require(
        len(set(pairs)) == 108 and len({j["source_onset"] for j in jobs}) == 36,
        "manifest must cover 36 unique cases x three unique seeds",
    )
    require({j["seed"] for j in jobs} == seeds, "manifest seed set mismatch")
    require(
        all(j.get("branches") == [] and j.get("free_run_trials") == 1 for j in jobs),
        "panel must be free-run-only with one trial per job",
    )
    require(
        all(isinstance(j.get("free_run_tokens"), int) and j["free_run_tokens"] > 0 for j in jobs),
        "manifest has invalid free-run caps",
    )
    # This panel intentionally preserves per-case caps, including longer extension caps.
    baseline = read(D / "setup/head_bfp8_precision_config.json")
    new = read(HERE / "down_bfp8_lofi_isolated_precision.json")
    for key in baseline:
        if key == "config_id":
            continue
        if key in ("weight_groups", "compute_fidelities"):
            for group in baseline[key]:
                if group == "down":
                    expected = "bfp8_lofi" if key == "weight_groups" else "LoFi"
                    require(new[key][group] == expected, f"down-group {key} delta mismatch")
                else:
                    require(new[key][group] == baseline[key][group], f"unintended {group} {key} change")
        else:
            require(new[key] == baseline[key], f"unintended policy field changed: {key}")
    require(
        baseline["weight_groups"]["down"] == "bfp4"
        and baseline["weight_groups"]["lm_head"] == new["weight_groups"]["lm_head"] == "bfp8_lofi"
        and baseline["compute_fidelities"]["lm_head"] == new["compute_fidelities"]["lm_head"] == "LoFi",
        "candidate must isolate down from the served BFP8 LoFi lm_head baseline",
    )
    require(new["max_seq_len"] == 262144, "context contract changed")
    precision_module = runpy.run_path(str(ROOT / "models/autoports/google_gemma_4_31b_it/tt/precision.py"))
    precision_module["load_precision"](new)
    require("ttnn" not in sys.modules, "static policy validation imported TTNN")
    spec = read(HERE / "command.json")
    argv = spec["argv"]
    require(spec["case_count"] == 108 and spec["mode"] == "full", "frozen command scope mismatch")
    require(argv[:3] == ["timeout", "9000", "env"], "frozen timeout changed")
    require(any(x.startswith("PYTHONPATH=") and str(ROOT) in x for x in argv), "explicit PYTHONPATH missing")
    require("READINESS_TRACE_REGION_SIZE=134217728" in argv, "trace region setting missing")
    require(spec["source_panel_sha256"] == sha(manifest_path), "command manifest pin mismatch")
    return {
        "status": "host_static_validation_passed_no_ttnn_import",
        "jobs": len(jobs),
        "unique_cases": len({j["source_onset"] for j in jobs}),
        "seeds": sorted(seeds),
        "caps": {
            str(cap): sum(j["free_run_tokens"] == cap for j in jobs)
            for cap in sorted({j["free_run_tokens"] for j in jobs})
        },
        "source_count": len(inventory["files"]),
        "stop_present": STOP.is_file(),
        "admission_present": (HERE / "admission.json").is_file(),
    }


def check_p2_terminal():
    admission_path = P2 / "admission.json"
    require(admission_path.is_file() and sha(admission_path) == P2_ADMISSION_SHA, "P2 actual admission bytes mismatch")
    admission = read(admission_path)
    require(admission.get("status") == "admitted", "P2 not admitted")
    for field, path in (
        ("proposal_sha256", P2 / "proposal.json"),
        ("driver_sha256", P2 / "run_after_p1_x.py"),
        ("source_inventory_sha256", P2 / "source-hashes.json"),
    ):
        require(path.is_file() and sha(path) == admission.get(field), f"P2 admission pin mismatch: {field}")
    source_inventory(P2 / "source-hashes.json")
    frozen_command = read(P2 / "command.json").get("argv")
    require(isinstance(frozen_command, list) and frozen_command, "P2 frozen command malformed")
    queue_path, launch_path, completion_path = P2 / "queue.json", P2 / "launch.json", P2 / "completion.json"
    require(
        queue_path.is_file() and launch_path.is_file() and completion_path.is_file(),
        "P2 queue/launch/completion not terminal",
    )
    queue, launch, completion = read(queue_path), read(launch_path), read(completion_path)
    require(
        queue.get("status") == "completed" and queue.get("exit_code") == 0, "P2 queue did not complete successfully"
    )
    require(queue.get("candidate_admission_sha256") == sha(admission_path), "P2 queue admission mismatch")
    require(
        launch.get("status") == "running" and launch.get("candidate_admission_sha256") == sha(admission_path),
        "P2 launch admission/status mismatch",
    )
    require(launch.get("command") == frozen_command, "P2 launch command differs from frozen argv")
    require(
        completion.get("status") == "completed" and completion.get("exit_code") == 0, "P2 completion did not succeed"
    )
    require(
        completion.get("candidate_admission_sha256") == sha(admission_path)
        and completion.get("command") == frozen_command,
        "P2 completion admission/command mismatch",
    )
    elapsed = completion.get("elapsed_seconds")
    bounds_obj = launch.get("forecast", {}).get("forecast_seconds", {})
    bounds = (bounds_obj.get("low"), bounds_obj.get("high"))
    require(
        elapsed is not None and all(isinstance(x, (int, float)) for x in bounds) and bounds[0] <= bounds[1],
        "P2 elapsed or forecast bounds missing",
    )
    review_path = HERE / "predecessors/p2-forecast-review.json"
    require(review_path.is_file(), "root P2 forecast review missing")
    review = read(review_path)
    require(
        review.get("completion_sha256") == sha(completion_path)
        and review.get("admission_sha256") == sha(admission_path),
        "P2 root review evidence hash mismatch",
    )
    in_range = validate_review(review, bounds, elapsed, contention=True)
    if not in_range:
        miss_path = P2 / "forecast-miss-accepted.json"
        require(miss_path.is_file(), "P2 forecast miss has no root acceptance")
        miss = read(miss_path)
        require(
            miss.get("reviewer") == "root"
            and miss.get("accepted") is True
            and miss.get("status") == "root_accepted_forecast_miss"
            and miss.get("completion_sha256") == sha(completion_path)
            and miss.get("admission_sha256") == sha(admission_path)
            and miss.get("measured_seconds") == elapsed
            and miss.get("forecast_range_seconds") == list(bounds)
            and miss.get("contention_disclosed") is True
            and miss.get("speed_ranking_used") is False,
            "P2 forecast-miss acceptance is not bound to actual completion/range/contention",
        )
    return {
        "admission_sha256": sha(admission_path),
        "queue_sha256": sha(queue_path),
        "launch_sha256": sha(launch_path),
        "completion_sha256": sha(completion_path),
        "elapsed_seconds": elapsed,
        "forecast_range_seconds": list(bounds),
        "forecast_review_sha256": sha(review_path),
    }


def check_sk_serial():
    admission_path = SK / "admission.json"
    require(admission_path.is_file(), "SK candidate needs a fresh root admission after mutable-pin correction")
    sk_admission = read(admission_path)
    require(sk_admission.get("status") == "admitted", "SK candidate is not freshly admitted")
    for field, path in (
        ("proposal_sha256", SK / "proposal.json"),
        ("driver_sha256", SK / "run_serial_candidate.py"),
        ("source_inventory_sha256", SK / "source-hashes.json"),
    ):
        require(path.is_file() and sha(path) == sk_admission.get(field), f"SK admission pin mismatch: {field}")
    source_inventory(SK / "source-hashes.json")
    review_path = HERE / "predecessors/sklearn-serial-review.json"
    require(review_path.is_file(), "root matched-Sklearn serialization review missing")
    review = read(review_path)
    require(
        review.get("reviewer") == "root" and review.get("accepted") is True,
        "root has not accepted matched-Sklearn serialization review",
    )
    require(
        review.get("matched_tt_status") in ("completed", "not_required"), "matched suffix512 TT review is not terminal"
    )
    require(
        review.get("hf_followup") in ("completed", "not_required"), "conditional HF follow-up remains pending/deferred"
    )
    require(
        review.get("sklearn_candidate_admission_sha256") == sha(admission_path),
        "Sklearn candidate admission pin mismatch",
    )
    for mode in ("served", "full"):
        cp = SK / f"completion-{mode}.json"
        require(
            cp.is_file() and review.get(f"{mode}_completion_sha256") == sha(cp),
            f"Sklearn {mode} completion is not bound to root review",
        )
        comp = read(cp)
        require(
            comp.get("status") == "completed" and comp.get("exit_code") == 0,
            f"Sklearn {mode} suffix512 run did not complete",
        )
    return {
        "admission_sha256": sha(admission_path),
        "review_sha256": sha(review_path),
        "matched_tt_status": review["matched_tt_status"],
        "hf_followup": review["hf_followup"],
    }


def active_hf_processes():
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if "precision_repro_hf.py" in args:
            found.append({"pid": int(proc.name), "cmdline": args[:300]})
    return found


def device_owners():
    sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
    import benchmark_server

    return benchmark_server.device_owners()


def self_test():
    # Pure in-memory mock checks: no subprocess, device query, or TTNN import.
    assert validate_review(
        {
            "reviewer": "root",
            "accepted": True,
            "forecast_range_seconds": [10, 20],
            "reviewed_elapsed_seconds": 15,
            "forecast_status": "in_range",
        },
        (10, 20),
        15,
    )
    assert not validate_review(
        {
            "reviewer": "root",
            "accepted": True,
            "forecast_range_seconds": [10, 20],
            "reviewed_elapsed_seconds": 25,
            "forecast_status": "accepted_miss",
            "contention_disclosed": True,
            "speed_ranking_used": False,
        },
        (10, 20),
        25,
        contention=True,
    )
    for bad in (
        {
            "reviewer": "root",
            "accepted": True,
            "forecast_range_seconds": [11, 20],
            "reviewed_elapsed_seconds": 15,
            "forecast_status": "in_range",
        },
        {
            "reviewer": "not-root",
            "accepted": True,
            "forecast_range_seconds": [10, 20],
            "reviewed_elapsed_seconds": 15,
            "forecast_status": "in_range",
        },
    ):
        try:
            validate_review(bad, (10, 20), 15)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid root forecast review accepted")
    print("down candidate host-gate self-test passed; no TTNN import or device query")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    static = validate_static()
    if args.validate_only:
        print(json.dumps(static, sort_keys=True))
        return
    require(args.execute, "explicit --execute required")
    admission, _ = candidate_gate(admission_required=True)
    require(not STOP.exists(), "STOP_BEFORE_LAUNCH is present")
    p2 = check_p2_terminal()
    sklearn = check_sk_serial()
    review_path = HERE / "predecessors/forecast-review.json"
    require(review_path.is_file(), "root candidate forecast review missing")
    review = read(review_path)
    forecast = read(HERE / "forecast.json")
    expected = (forecast["forecast_seconds"]["low"], forecast["forecast_seconds"]["high"])
    require(
        review.get("forecast_sha256") == sha(HERE / "forecast.json")
        and review.get("forecast_range_seconds") == list(expected)
        and review.get("reviewer") == "root"
        and review.get("accepted") is True
        and review.get("uncertainty_acknowledged") is True
        and review.get("p2_contention_excluded_from_speed_calibration") is True,
        "root candidate forecast review missing, stale, or incomplete",
    )
    hf = active_hf_processes()
    require(not hf, f"active HF process(es): {hf}")
    owners = device_owners()
    require(not owners, f"device owner(s): {owners}")
    launch_path = HERE / "launch.json"
    completion_path = HERE / "completion.json"
    require(not launch_path.exists() and not completion_path.exists(), "duplicate launch refused")
    outdir = ROOT / read(HERE / "command.json")["argv"][read(HERE / "command.json")["argv"].index("--output-dir") + 1]
    require(not outdir.exists() or not any(outdir.iterdir()), "nonempty output directory refused")
    outdir.mkdir(parents=True, exist_ok=True)
    argv = read(HERE / "command.json")["argv"]
    launch = {
        "status": "running",
        "command": argv,
        "start_unix": time.time(),
        "timeout_seconds": 9000,
        "candidate_admission_sha256": sha(HERE / "admission.json"),
        "candidate_proposal_sha256": sha(HERE / "proposal.json"),
        "candidate_source_inventory_sha256": sha(HERE / "source-hashes.json"),
        "forecast": forecast,
        "p2_terminal": p2,
        "sklearn_serial": sklearn,
        "forecast_review_sha256": sha(review_path),
        "active_hf_processes": hf,
        "device_owner_preflight": owners,
    }
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")
    with (HERE / "panel.log").open("w") as log:
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=9010, check=False)
    outputs = []
    if result.returncode == 0:
        for job in read(HERE / "candidate-cases.json"):
            path = outdir / (job["name"] + ".json")
            require(path.is_file() and path.stat().st_size > 0, f"missing/empty output: {path}")
            row = read(path)
            free = row.get("free_running", [])
            require(
                row.get("case") == job["case"] and row.get("metadata", {}).get("seed") == job["seed"],
                f'output fixture/seed mismatch: {job["name"]}',
            )
            require(
                len(free) == 1 and 0 < len(free[0].get("token_ids", [])) <= job["free_run_tokens"],
                f'free-run output/cap mismatch: {job["name"]}',
            )
            require(free[0].get("sampling", {}).get("seed") == job["seed"], f'seed mismatch: {job["name"]}')
            outputs.append(
                {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "tokens": len(free[0]["token_ids"])}
            )
    completion = {
        **launch,
        "status": "completed" if result.returncode == 0 else "failed",
        "end_unix": time.time(),
        "elapsed_seconds": time.time() - launch["start_unix"],
        "exit_code": result.returncode,
        "outputs": outputs,
    }
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"panel failed with {result.returncode}; no retry")
    print(
        json.dumps(
            {"status": completion["status"], "elapsed_seconds": completion["elapsed_seconds"], "outputs": len(outputs)}
        )
    )


if __name__ == "__main__":
    main()
