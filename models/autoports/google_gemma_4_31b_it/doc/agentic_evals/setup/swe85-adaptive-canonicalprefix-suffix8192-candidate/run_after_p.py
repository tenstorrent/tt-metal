#!/usr/bin/env python3
"""Inert-until-admitted serial suffix8192 runner gated on P terminal success and cost review."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
assert (ROOT / "bringup/run_contract.json").is_file(), "repository root resolution failed"
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
OUT = D / "setup/swe85-adaptive-canonicalprefix-suffix8192-candidate"
P = D / "setup/swe85-precision-bisect/head-bfp8-hifi2-first-panel-after-w"
EXPECTED_P_ADMISSION_SHA256 = "91dc764e6a3a6b1e8a1be51e6eee5fc958c19d1dfcfa3d057b989aa325618cdd"
STOP = OUT / "STOP_BEFORE_LAUNCH"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def candidate_admission():
    admission = read(OUT / "admission.json")
    assert admission.get("status") == "admitted", "root admission required; candidate remains inert"
    proposal = OUT / "proposal.json"
    driver = Path(__file__)
    assert sha(proposal) == admission.get("proposal_sha256"), "candidate proposal hash mismatch"
    assert sha(driver) == admission.get("driver_sha256"), "candidate driver hash mismatch"
    assert admission.get("expected_p_admission_sha256") == EXPECTED_P_ADMISSION_SHA256
    return admission, read(proposal)


def check_candidate_sources(proposal):
    for name, expected in proposal["candidate_hashes"].items():
        path = OUT / name
        assert path.is_file() and sha(path) == expected, f"candidate source changed: {name}"
    provenance = read(OUT / "provenance.json")
    for row in provenance["fixture_records"]:
        path = ROOT / row["path"]
        assert path.is_file() and sha(path) == row["sha256"], f"candidate fixture changed: {row['path']}"


def p_immutable_pins():
    admission_path = P / "admission.json"
    assert admission_path.is_file() and sha(admission_path) == EXPECTED_P_ADMISSION_SHA256, "P admission pin mismatch"
    admission = read(admission_path)
    assert admission.get("status") == "admitted", "P is not admitted"
    assert sha(P / "proposal.json") == admission.get("proposal_sha256"), "P proposal changed"
    assert sha(P / "run_panel_after_w.py") == admission.get("driver_sha256"), "P driver changed"
    source_hashes_path = P / "source-hashes.json"
    inventory = read(source_hashes_path)
    for item in inventory["files"].values():
        path = ROOT / item["path"]
        assert path.is_file() and sha(path) == item["sha256"], f"P source pin changed: {item['path']}"
    return {
        "admission_sha256": sha(admission_path),
        "proposal_sha256": sha(P / "proposal.json"),
        "driver_sha256": sha(P / "run_panel_after_w.py"),
        "source_hashes_sha256": sha(source_hashes_path),
    }


def evaluate_forecast_gate(
    measured_seconds, forecast_range, accepted_record=None, completion_sha256=None, admission_sha256=None
):
    low, high = forecast_range
    if low <= measured_seconds <= high:
        return {
            "status": "forecast_range_hit",
            "measured_seconds": measured_seconds,
            "forecast_range_seconds": [low, high],
        }
    if not accepted_record:
        raise RuntimeError("P forecast miss; root acceptance record is required before suffix8192 execution")
    required = {
        "status": "root_accepted_forecast_miss",
        "reviewer": "root",
        "accepted": True,
        "completion_sha256": completion_sha256,
        "measured_seconds": measured_seconds,
        "forecast_range_seconds": [low, high],
        "admission_sha256": admission_sha256,
    }
    for key, value in required.items():
        assert accepted_record.get(key) == value, f"forecast-miss acceptance mismatch: {key}"
    return {
        "status": "root_accepted_forecast_miss",
        "measured_seconds": measured_seconds,
        "forecast_range_seconds": [low, high],
        "acceptance_record_sha256": accepted_record.get("_sha256"),
    }


def wait_for_p_terminal():
    deadline = time.monotonic() + 12 * 3600
    qpath, cpath = P / "queue.json", P / "completion.json"
    while True:
        if STOP.exists():
            raise RuntimeError("stop file set while waiting for P")
        if qpath.is_file() and cpath.is_file():
            current_queue = read(qpath)
            if current_queue.get("status") in {"completed", "failed"}:
                break
        if time.monotonic() >= deadline:
            raise TimeoutError("P did not reach terminal queue/completion within 12 hours")
        time.sleep(10)
    q = read(qpath)
    completion = read(cpath)
    launch = read(P / "launch.json")
    assert q.get("status") == "completed" and q.get("exit_code") == 0, "P queue is not terminal-success"
    assert completion.get("exit_code") == 0, "P completion is not successful"
    p_proposal = read(P / "proposal.json")
    assert (
        completion.get("command") == p_proposal["execution"]["command"]
    ), "P completion command does not match frozen proposal"
    assert q.get("admission_sha256") == EXPECTED_P_ADMISSION_SHA256
    assert completion.get("admission_sha256") == EXPECTED_P_ADMISSION_SHA256
    assert launch.get("admission_sha256") == EXPECTED_P_ADMISSION_SHA256
    assert completion.get("elapsed_seconds") is not None, "P completion lacks measured elapsed_seconds"
    forecast = launch.get("forecast", {})
    forecast_range = forecast.get("forecast_range_seconds")
    if forecast_range is None:
        # Existing P launch records use this exact field; do not guess a range from a stale source.
        raise RuntimeError("P launch lacks frozen forecast_range_seconds")
    assert len(forecast_range) == 2 and forecast_range[0] <= forecast_range[1]
    estimate_path = ROOT / forecast["forecast_path"]
    assert sha(estimate_path) == forecast.get("forecast_sha256"), "P forecast source hash mismatch"
    completion_hash = sha(cpath)
    acceptance_path = P / "forecast-miss-accepted.json"
    acceptance = None
    measured = completion["elapsed_seconds"]
    low, high = forecast_range
    if not low <= measured <= high:
        if not acceptance_path.is_file():
            raise RuntimeError("P elapsed outside forecast range; missing root forecast-miss-accepted.json")
        acceptance = read(acceptance_path)
        acceptance["_sha256"] = sha(acceptance_path)
    gate = evaluate_forecast_gate(measured, forecast_range, acceptance, completion_hash, EXPECTED_P_ADMISSION_SHA256)
    pins = p_immutable_pins()
    terminal_hashes = {
        "queue_sha256": sha(qpath),
        "completion_sha256": completion_hash,
        "launch_sha256": sha(P / "launch.json"),
        **pins,
    }
    return gate, terminal_hashes


def active_hf_processes():
    hits = []
    for ent in Path("/proc").iterdir():
        if not ent.name.isdigit() or int(ent.name) == os.getpid():
            continue
        try:
            args = (ent / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if "precision_repro_hf.py" in args:
            hits.append({"pid": int(ent.name), "cmdline": args[:400]})
    return hits


def device_owners():
    sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
    import benchmark_server

    return benchmark_server.device_owners()


def recheck_predecessor(expected_terminal):
    pins = p_immutable_pins()
    for key, expected in expected_terminal.items():
        if key in pins:
            assert pins[key] == expected, f"P immutable pin changed before mode: {key}"
        else:
            name = {
                "queue_sha256": "queue.json",
                "completion_sha256": "completion.json",
                "launch_sha256": "launch.json",
            }[key]
            assert sha(P / name) == expected, f"P terminal artifact changed before mode: {name}"


def run_mode(mode, proposal, admission, p_gate, p_terminal):
    if STOP.exists():
        raise RuntimeError("stop file set before mode launch")
    check_candidate_sources(proposal)
    current_admission = read(OUT / "admission.json")
    assert current_admission.get("status") == "admitted", "root admission required before each mode"
    assert current_admission == admission, "candidate admission changed between serial modes"
    assert sha(OUT / "proposal.json") == current_admission.get("proposal_sha256")
    assert sha(Path(__file__)) == current_admission.get("driver_sha256")
    recheck_predecessor(p_terminal)
    hits = active_hf_processes()
    assert not hits, f"active HF reference process overlaps: {hits}"
    owners = device_owners()
    assert not owners, f"device owners exist: {owners}"
    recheck_predecessor(p_terminal)
    launch_path = OUT / f"launch-{mode}.json"
    completion_path = OUT / f"completion-{mode}.json"
    assert not launch_path.exists() and not completion_path.exists(), f"refusing duplicate {mode} execution"
    command_config = read(OUT / "command.json")["modes"][mode]
    command = command_config["command"]
    outdir = ROOT / command_config["output_dir"]
    assert not outdir.exists() or not any(outdir.iterdir()), f"refusing non-empty output directory: {outdir}"
    outdir.mkdir(parents=True, exist_ok=True)
    launch = {
        "mode": mode,
        "command": command,
        "start_unix": time.time(),
        "candidate_admission_sha256": sha(OUT / "admission.json"),
        "candidate_proposal_sha256": sha(OUT / "proposal.json"),
        "p_gate": p_gate,
        "p_terminal_hashes": p_terminal,
        "device_owner_preflight": owners,
        "active_hf_processes": hits,
        "timeout_seconds": 1800,
    }
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")
    with (OUT / f"{mode}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1810)
    completion = {
        **launch,
        "end_unix": time.time(),
        "elapsed_seconds": time.time() - launch["start_unix"],
        "exit_code": result.returncode,
    }
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    if result.returncode != 0:
        raise RuntimeError(f"{mode} suffix8192 command failed; no automatic retry")
    jobs = read(OUT / f"{mode}-cases.json")
    for job in jobs:
        result_path = outdir / f"{job['name']}.json"
        assert result_path.is_file(), f"missing {mode} result: {result_path}"
        result_data = read(result_path)
        assert len(result_data.get("continuations", {}).get("chosen", {}).get("positions", [])) == 64
        assert len(result_data.get("free_running", [])) == 1
    return completion


def compare_modes():
    served_jobs = {j["name"]: j for j in read(OUT / "served-cases.json")}
    full_jobs = {j["name"]: j for j in read(OUT / "full-cases.json")}
    assert set(served_jobs) == set(full_jobs) and len(served_jobs) == 3
    rows = []
    for name in sorted(served_jobs):
        a = read(ROOT / served_jobs[name]["case"])
        b = read(ROOT / full_jobs[name]["case"])
        assert a["prompt_token_ids"] == b["prompt_token_ids"]
        assert a["continuations"]["chosen"] == b["continuations"]["chosen"]
        assert a["metadata"]["sampling"] == b["metadata"]["sampling"]
        assert a["metadata"]["seed"] == b["metadata"]["seed"]
        sr = read(OUT / "runs/served" / f"{name}.json")
        fr = read(OUT / "runs/full" / f"{name}.json")
        s = sr["continuations"]["chosen"]
        f = fr["continuations"]["chosen"]
        assert [(x["position"], x["forced_token"]) for x in s["positions"]] == [
            (x["position"], x["forced_token"]) for x in f["positions"]
        ]
        n = len(s["positions"])
        rows.append(
            {
                "name": name,
                "chosen_tokens": n,
                "top1_agreement": sum(
                    x["top1_token"] == y["top1_token"] for x, y in zip(s["positions"], f["positions"])
                )
                / n,
                "served_result_sha256": sha(OUT / "runs/served" / f"{name}.json"),
                "full_result_sha256": sha(OUT / "runs/full" / f"{name}.json"),
            }
        )
    report = {
        "status": "passed",
        "scope": "Host-only matched chosen-token comparison; no score/admission decision.",
        "rows": rows,
        "served_manifest_sha256": sha(OUT / "served-cases.json"),
        "full_manifest_sha256": sha(OUT / "full-cases.json"),
    }
    (OUT / "comparison-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def self_test():
    assert evaluate_forecast_gate(5.0, [4.0, 6.0])["status"] == "forecast_range_hit"
    try:
        evaluate_forecast_gate(7.0, [4.0, 6.0])
    except RuntimeError:
        pass
    else:
        raise AssertionError("forecast miss without root record must block")
    accepted = {
        "status": "root_accepted_forecast_miss",
        "reviewer": "root",
        "accepted": True,
        "completion_sha256": "abc",
        "measured_seconds": 7.0,
        "forecast_range_seconds": [4.0, 6.0],
        "admission_sha256": "def",
    }
    assert evaluate_forecast_gate(7.0, [4.0, 6.0], accepted, "abc", "def")["status"] == "root_accepted_forecast_miss"
    wrong = dict(accepted, completion_sha256="other")
    try:
        evaluate_forecast_gate(7.0, [4.0, 6.0], wrong, "abc", "def")
    except AssertionError:
        pass
    else:
        raise AssertionError("mismatched root miss record must block")
    print("forecast gate self-test passed; no TTNN, device, or predecessor access")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    admission, proposal = candidate_admission()
    check_candidate_sources(proposal)
    if STOP.exists():
        raise RuntimeError("stop file present")
    qstate = {
        "status": "waiting_for_P_terminal_success_and_forecast_gate",
        "candidate_admission_sha256": sha(OUT / "admission.json"),
        "p_admission_sha256": EXPECTED_P_ADMISSION_SHA256,
        "stop_file": str(STOP),
        "queued_unix": time.time(),
    }
    (OUT / "queue.json").write_text(json.dumps(qstate, indent=2) + "\n")
    gate, terminal = wait_for_p_terminal()
    qstate.update(status="P_terminal_success_forecast_gate_passed", p_gate=gate, p_terminal_hashes=terminal)
    (OUT / "queue.json").write_text(json.dumps(qstate, indent=2) + "\n")
    for mode in ["served", "full"]:
        if STOP.exists():
            qstate.update(status="stopped_before_" + mode)
            (OUT / "queue.json").write_text(json.dumps(qstate, indent=2) + "\n")
            return
        run_mode(mode, proposal, admission, gate, terminal)
        qstate.update(
            status=mode + "_completed",
            last_completed_mode=mode,
            last_completion_sha256=sha(OUT / f"completion-{mode}.json"),
        )
        (OUT / "queue.json").write_text(json.dumps(qstate, indent=2) + "\n")
    report = compare_modes()
    qstate.update(
        status="all_suffix8192_modes_completed",
        exit_code=0,
        comparison_sha256=sha(OUT / "comparison-summary.json"),
        paired_cases=report["rows"],
    )
    (OUT / "queue.json").write_text(json.dumps(qstate, indent=2) + "\n")


if __name__ == "__main__":
    main()
