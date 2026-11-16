#!/usr/bin/env python3
"""Inert-until-admitted P2 driver gated on P1 and X terminal success."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[8]
assert (ROOT / "bringup/run_contract.json").is_file(), "repository root resolution failed"
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
OUT = D / "setup/swe85-precision-bisect/head-bf16-hifi4-fixed108-proposal"
P1 = D / "setup/swe85-precision-bisect/head-bfp8-hifi2-first-panel-after-w"
X = D / "setup/swe85-adaptive-canonicalprefix-suffix8192-p1pinfix-candidate"
EXPECTED_P1_ADMISSION = "91dc764e6a3a6b1e8a1be51e6eee5fc958c19d1dfcfa3d057b989aa325618cdd"
EXPECTED_X_ADMISSION = "300d69403a28112bc5f2a7225df5327bdb9e577ad9ca96cf754ab304f3538205"
STOP = OUT / "STOP_BEFORE_LAUNCH"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require_fields(actual, expected, label):
    for key, value in expected.items():
        assert actual.get(key) == value, f"{label} field mismatch: {key}"


def require_terminal(record, status, label):
    require_fields(record, {"status": status, "exit_code": 0}, label)


def check_inventory(path, inventory):
    for item in inventory["files"].values():
        f = ROOT / item["path"]
        assert f.is_file() and sha(f) == item["sha256"], f"pinned source changed: {item['path']}"


def candidate_gate():
    admission_path = OUT / "admission.json"
    assert admission_path.is_file(), "root admission required; candidate remains inert"
    admission = read(admission_path)
    assert admission.get("status") == "admitted", "root admission required; candidate remains inert"
    proposal_path = OUT / "proposal.json"
    assert sha(proposal_path) == admission.get("proposal_sha256"), "candidate proposal hash mismatch"
    assert sha(Path(__file__)) == admission.get("driver_sha256"), "candidate driver hash mismatch"
    proposal = read(proposal_path)
    assert sha(OUT / "source-hashes.json") == proposal["source_inventory_sha256"], "candidate inventory hash mismatch"
    check_inventory(OUT / "source-hashes.json", read(OUT / "source-hashes.json"))
    return admission, proposal


def p1_immutable_pins():
    apath = P1 / "admission.json"
    ppath = P1 / "proposal.json"
    dpath = P1 / "run_panel_after_w.py"
    assert apath.is_file() and sha(apath) == EXPECTED_P1_ADMISSION, "P1 actual admission bytes differ from pinned SHA"
    adm = read(apath)
    assert adm.get("status") == "admitted", "P1 is not admitted"
    assert sha(ppath) == adm.get("proposal_sha256"), "P1 actual proposal does not match its admission"
    assert sha(dpath) == adm.get("driver_sha256"), "P1 actual driver does not match its admission"
    pp = read(ppath)
    for item in pp["source_hashes"]:
        f = ROOT / item["path"]
        assert f.is_file() and sha(f) == item["sha256"], f"P1 source pin changed: {item['path']}"
    inventory = read(P1 / "source-hashes.json")
    stale_placeholder = "428bea328fae8fa6e4f6fe21802f3763985d6b5ee819f5beac6c6df3ff1f2b97"
    for item in inventory["files"].values():
        f = ROOT / item["path"]
        if item["path"] == str((P1 / "admission.json").relative_to(ROOT)):
            assert item["sha256"] == stale_placeholder, "unexpected P1 stale-admission placeholder lineage"
            continue
        assert f.is_file() and sha(f) == item["sha256"], f"P1 inventory source changed: {item['path']}"
    return {
        "admission_sha256": sha(apath),
        "proposal_sha256": sha(ppath),
        "driver_sha256": sha(dpath),
        "source_inventory_sha256": sha(P1 / "source-hashes.json"),
    }


def forecast_gate(measured, bounds, accepted, completion_sha, admission_sha):
    lo, hi = bounds
    if lo <= measured <= hi:
        return {"status": "forecast_range_hit", "measured_seconds": measured, "forecast_range_seconds": bounds}
    assert accepted is not None, "forecast miss requires root-authored acceptance"
    expect = {
        "status": "root_accepted_forecast_miss",
        "reviewer": "root",
        "accepted": True,
        "completion_sha256": completion_sha,
        "measured_seconds": measured,
        "forecast_range_seconds": bounds,
        "admission_sha256": admission_sha,
    }
    for k, v in expect.items():
        assert accepted.get(k) == v, f"forecast-miss acceptance mismatch: {k}"
    return {"status": "root_accepted_forecast_miss", "measured_seconds": measured, "forecast_range_seconds": bounds}


def wait_terminal(predecessor, statuses, label):
    deadline = time.monotonic() + 24 * 3600
    qpath = predecessor / "queue.json"
    while True:
        if STOP.exists():
            raise RuntimeError("STOP_BEFORE_LAUNCH set while waiting")
        if qpath.is_file() and read(qpath).get("status") in statuses:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{label} failed to reach terminal state within24h")
        time.sleep(10)
    return read(qpath)


def check_p1():
    q = wait_terminal(P1, {"completed", "failed"}, "P1")
    cpath = P1 / "completion.json"
    lpath = P1 / "launch.json"
    require_terminal(q, "completed", "P1 queue")
    assert cpath.is_file() and lpath.is_file(), "P1 completion/launch missing"
    c = read(cpath)
    launch = read(lpath)
    pp = read(P1 / "proposal.json")
    assert (
        c.get("exit_code") == 0 and c.get("command") == pp["execution"]["command"]
    ), "P1 completion does not match frozen successful command"
    pins = p1_immutable_pins()
    assert c.get("admission_sha256") == pins["admission_sha256"] == EXPECTED_P1_ADMISSION
    assert launch.get("admission_sha256") == EXPECTED_P1_ADMISSION
    forecast = launch.get("forecast", {})
    bounds = forecast.get("forecast_range_seconds")
    assert bounds and len(bounds) == 2 and bounds[0] <= bounds[1], "P1 frozen forecast range missing"
    fpath = ROOT / forecast["forecast_path"]
    assert fpath.is_file() and sha(fpath) == forecast.get("forecast_sha256"), "P1 forecast source mismatch"
    measured = c.get("elapsed_seconds")
    assert measured is not None
    csha = sha(cpath)
    acceptance_path = P1 / "forecast-miss-accepted.json"
    accepted = read(acceptance_path) if acceptance_path.is_file() else None
    gate = forecast_gate(measured, bounds, accepted, csha, EXPECTED_P1_ADMISSION)
    return q, c, launch, gate, pins


def x_immutable_pins():
    apath = X / "admission.json"
    ppath = X / "proposal.json"
    dpath = X / "run_after_p.py"
    assert apath.is_file() and sha(apath) == EXPECTED_X_ADMISSION, "X actual admission bytes differ from pinned SHA"
    adm = read(apath)
    proposal = read(ppath)
    assert adm.get("status") == "admitted" and sha(ppath) == adm.get(
        "proposal_sha256"
    ), "X actual proposal/admission mismatch"
    assert sha(dpath) == adm.get("driver_sha256"), "X actual driver/admission mismatch"
    for name, expected in proposal["candidate_hashes"].items():
        f = X / name
        assert f.is_file() and sha(f) == expected, f"X candidate changed: {name}"
    prov = read(X / "provenance.json")
    for row in prov["fixture_records"]:
        f = ROOT / row["path"]
        assert f.is_file() and sha(f) == row["sha256"], f"X fixture changed: {row['path']}"
    return {"admission_sha256": sha(apath), "proposal_sha256": sha(ppath), "driver_sha256": sha(dpath)}


def check_x(p1_completion_sha):
    q = wait_terminal(X, {"all_suffix8192_modes_completed", "failed"}, "X")
    require_terminal(q, "all_suffix8192_modes_completed", "X queue")
    pins = x_immutable_pins()
    proposal = read(X / "proposal.json")
    assert proposal["predecessor_gate"]["admission_sha256"] == EXPECTED_P1_ADMISSION
    assert q.get("candidate_admission_sha256") == EXPECTED_X_ADMISSION, "X queue candidate admission pin mismatch"
    assert (
        q.get("p_terminal_hashes", {}).get("completion_sha256") == p1_completion_sha
    ), "X did not use the exact P1 completion"
    forecast = read(X / "forecast.json")
    command = read(X / "command.json")
    mode_gates = {}
    mode_hashes = {}
    for mode in ("served", "full"):
        cpath = X / f"completion-{mode}.json"
        lpath = X / f"launch-{mode}.json"
        assert cpath.is_file() and lpath.is_file(), f"X {mode} terminal evidence missing"
        c = read(cpath)
        l = read(lpath)
        spec = command["modes"][mode]
        assert (
            c.get("exit_code") == 0 and c.get("mode") == mode and c.get("command") == spec["command"]
        ), f"X {mode} completion command/status mismatch"
        assert l.get("command") == spec["command"] and l.get("mode") == mode, f"X {mode} launch command mismatch"
        bounds = forecast["modes"][mode]["range_s"]
        measured = c.get("elapsed_seconds")
        assert measured is not None
        acceptance_path = X / f"forecast-miss-accepted-{mode}.json"
        accepted = read(acceptance_path) if acceptance_path.is_file() else None
        mode_gates[mode] = forecast_gate(measured, bounds, accepted, sha(cpath), EXPECTED_X_ADMISSION)
        mode_hashes[mode] = {"completion_sha256": sha(cpath), "launch_sha256": sha(lpath)}
    comparison = X / "comparison-summary.json"
    assert comparison.is_file() and read(comparison).get("status") == "passed", "X paired comparison absent"
    return (
        q,
        {
            **pins,
            "queue_sha256": sha(X / "queue.json"),
            "forecast_sha256": sha(X / "forecast.json"),
            "served": mode_hashes["served"],
            "full": mode_hashes["full"],
            "comparison_sha256": sha(comparison),
        },
        mode_gates,
    )


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


def self_test():
    assert forecast_gate(5, [4, 6], None, "c", "a")["status"] == "forecast_range_hit"
    try:
        forecast_gate(7, [4, 6], None, "c", "a")
    except AssertionError:
        pass
    else:
        raise AssertionError("forecast miss without acceptance did not stop")
    ok = {
        "status": "root_accepted_forecast_miss",
        "reviewer": "root",
        "accepted": True,
        "completion_sha256": "c",
        "measured_seconds": 7,
        "forecast_range_seconds": [4, 6],
        "admission_sha256": "a",
    }
    assert forecast_gate(7, [4, 6], ok, "c", "a")["status"] == "root_accepted_forecast_miss"
    for altered in (dict(ok, completion_sha256="x"), dict(ok, admission_sha256="x"), dict(ok, measured_seconds=8)):
        try:
            forecast_gate(7, [4, 6], altered, "c", "a")
        except AssertionError:
            pass
        else:
            raise AssertionError("mismatched miss acceptance was accepted")
    try:
        check_inventory("unused", {"files": {"bad": {"path": "bringup/run_contract.json", "sha256": "bad"}}})
    except AssertionError:
        pass
    else:
        raise AssertionError("tampered source pin was accepted")
    require_terminal({"status": "completed", "exit_code": 0}, "completed", "mock predecessor")
    try:
        require_fields({"admission_sha256": "old"}, {"admission_sha256": "expected"}, "mock predecessor pin")
    except AssertionError:
        pass
    else:
        raise AssertionError("mismatched predecessor admission was accepted")
    for bad in (
        {"status": "failed", "exit_code": 0},
        {"status": "completed", "exit_code": 1},
        {"status": "completed", "exit_code": 0, "admission_sha256": "wrong"},
    ):
        try:
            require_terminal(bad, "completed", "mock predecessor")
            require_fields(bad, {"admission_sha256": "expected"}, "mock predecessor pin")
        except AssertionError:
            pass
        else:
            raise AssertionError("invalid predecessor terminal/pin fields were accepted")
    print("P2 host gate tests passed; no TTNN import, device query, or predecessor mutation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    admission, proposal = candidate_gate()
    assert not STOP.exists(), "STOP_BEFORE_LAUNCH present"
    qstate = {
        "status": "waiting_for_P1_and_X_terminal_success",
        "candidate_admission_sha256": sha(OUT / "admission.json"),
        "stop_file": str(STOP),
        "queued_unix": time.time(),
    }
    (OUT / "queue.json").write_text(json.dumps(qstate, indent=2) + "\n")
    p1q, p1c, p1l, p1gate, p1pins = check_p1()
    xq, xpins, xgates = check_x(sha(P1 / "completion.json"))
    qstate.update(
        status="P1_X_terminal_success_forecasts_accepted",
        p1_forecast_gate=p1gate,
        x_mode_forecast_gates=xgates,
        p1_hashes={
            "queue": sha(P1 / "queue.json"),
            "completion": sha(P1 / "completion.json"),
            "launch": sha(P1 / "launch.json"),
            **p1pins,
        },
        x_hashes=xpins,
    )
    (OUT / "queue.json").write_text(json.dumps(qstate, indent=2) + "\n")
    if STOP.exists():
        qstate.update(status="stopped_before_launch")
        (OUT / "queue.json").write_text(json.dumps(qstate, indent=2) + "\n")
        return
    admission2, proposal2 = candidate_gate()
    assert admission2 == admission and proposal2 == proposal
    assert (
        sha(P1 / "queue.json") == qstate["p1_hashes"]["queue"]
        and sha(P1 / "completion.json") == qstate["p1_hashes"]["completion"]
        and sha(P1 / "launch.json") == qstate["p1_hashes"]["launch"]
    )
    assert (
        sha(X / "queue.json") == xpins["queue_sha256"]
        and sha(X / "completion-served.json") == xpins["served"]["completion_sha256"]
        and sha(X / "completion-full.json") == xpins["full"]["completion_sha256"]
    )
    hits = active_hf_processes()
    assert not hits, f"HF reference process overlaps: {hits}"
    owners = device_owners()
    assert not owners, f"device owners exist: {owners}"
    assert not (OUT / "launch.json").exists(), "refusing duplicate P2 launch"
    if STOP.exists():
        raise RuntimeError("STOP_BEFORE_LAUNCH set after preflight")
    command = read(OUT / "command.json")["argv"]
    assert command[:2] == ["timeout", "7200"], "P2 timeout must be7200s"
    output_dir = Path(command[command.index("--output-dir") + 1])
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    assert not output_dir.exists() or not any(output_dir.iterdir()), f"refusing nonempty output directory: {output_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)
    launch = {
        "status": "running",
        "command": command,
        "start_unix": time.time(),
        "timeout_seconds": 7200,
        "candidate_admission_sha256": sha(OUT / "admission.json"),
        "candidate_proposal_sha256": sha(OUT / "proposal.json"),
        "forecast": read(OUT / "runtime-estimate.json"),
        "p1_forecast_gate": p1gate,
        "x_forecast_gates": xgates,
        "p1_terminal_hashes": qstate["p1_hashes"],
        "x_terminal_hashes": xpins,
        "active_hf_processes": hits,
        "device_owner_preflight": owners,
    }
    (OUT / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    with (OUT / "panel.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=7210)
    completion = {
        **launch,
        "status": "completed" if result.returncode == 0 else "failed",
        "end_unix": time.time(),
        "elapsed_seconds": time.time() - launch["start_unix"],
        "exit_code": result.returncode,
    }
    (OUT / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    qstate.update(status=completion["status"], exit_code=result.returncode, end_unix=completion["end_unix"])
    (OUT / "queue.json").write_text(json.dumps(qstate, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError("P2 failed; no automatic retry")


if __name__ == "__main__":
    main()
