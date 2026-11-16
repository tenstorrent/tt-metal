#!/usr/bin/env python3
"""Host-gated, one-mode-at-a-time runner for the Sklearn suffix512 candidate."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd().resolve()
assert (ROOT / "bringup/run_contract.json").is_file(), "run from repository root"
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
P2 = D / "setup/swe85-precision-bisect/head-bf16-hifi4-fixed108-proposal"
P2_ADMISSION_SHA = "74bef162c97e23e882612eb85dae365f32c9da1f13ba4b9b8e1e1f8f582ad1be"
STOP = HERE / "STOP_BEFORE_LAUNCH"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_hf_followup(review):
    require(
        review.get("hf_followup") in ("completed", "not_required", "deferred_until_matched_TT"),
        "HF follow-up status must be completed, not required, or deferred until matched TT evidence",
    )


def validate_stop_state(stop_present, executing):
    if executing:
        require(not stop_present, "STOP_BEFORE_LAUNCH is present")


def validate_forecast_review(review, bounds, elapsed):
    in_range = bounds[0] <= elapsed <= bounds[1]
    require(review.get("reviewer") == "root" and review.get("accepted") is True, "root forecast review is not accepted")
    require(
        review.get("forecast_range_seconds") == list(bounds),
        "root forecast review does not bind the actual forecast range",
    )
    require(
        review.get("reviewed_elapsed_seconds") == elapsed,
        "root forecast review does not bind measured elapsed wall time",
    )
    require(
        review.get("contention_disclosed") is True and review.get("speed_ranking_used") is False,
        "P2 concurrent-HF overlap must be disclosed and excluded from policy speed ranking",
    )
    require(
        review.get("forecast_status") == ("in_range" if in_range else "accepted_miss"),
        "root forecast status must match the measured range",
    )


def candidate_pins(admission_required):
    inventory_path = HERE / "source-hashes.json"
    if admission_required:
        ap = HERE / "admission.json"
        require(ap.is_file(), "root admission.json is required")
        adm = read(ap)
        require(adm.get("status") == "admitted", "candidate is not admitted")
        for field, path in (
            ("proposal_sha256", HERE / "proposal.json"),
            ("driver_sha256", Path(__file__)),
            ("source_inventory_sha256", inventory_path),
            ("commands_sha256", HERE / "commands.json"),
            ("forecast_sha256", HERE / "forecast.json"),
        ):
            require(sha(path) == adm.get(field), f"admission pin mismatch: {field}")
    else:
        adm = None
    inventory = read(inventory_path)
    for rel, digest in inventory["files"].items():
        path = ROOT / rel
        require(path.is_file() and sha(path) == digest, f"source pin mismatch: {rel}")
    return adm, inventory


def terminal_p2():
    admission = P2 / "admission.json"
    require(admission.is_file(), "P2 actual admission is missing")
    actual_admission_sha = sha(admission)
    if P2_ADMISSION_SHA:
        require(actual_admission_sha == P2_ADMISSION_SHA, "P2 admission bytes changed")
    p2adm = read(admission)
    require(p2adm.get("status") == "admitted", "P2 lacks admitted status")
    for field, path in (
        ("proposal_sha256", P2 / "proposal.json"),
        ("driver_sha256", P2 / "run_after_p1_x.py"),
        ("source_inventory_sha256", P2 / "source-hashes.json"),
    ):
        require(path.is_file() and sha(path) == p2adm.get(field), f"P2 actual admission pin mismatch: {field}")
    p2inventory = read(P2 / "source-hashes.json")
    sources = p2inventory.get("files", {})
    require(isinstance(sources, dict) and sources, "P2 source inventory is empty or malformed")
    for name, item in sources.items():
        require(
            isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str),
            f"P2 source inventory row malformed: {name}",
        )
        rel, digest = item["path"], item["sha256"]
        path = ROOT / rel
        require(path.is_file() and sha(path) == digest, f"P2 source pin mismatch: {rel}")
    p2command_path = P2 / "command.json"
    require(p2command_path.is_file(), "P2 frozen command is missing")
    p2command_doc = read(p2command_path)
    p2command = p2command_doc.get("argv")
    require(isinstance(p2command, list) and p2command, "P2 frozen command is malformed")
    completion_path, queue_path = P2 / "completion.json", P2 / "queue.json"
    require(completion_path.is_file() and queue_path.is_file(), "P2 completion/queue not terminal yet")
    completion, queue = read(completion_path), read(queue_path)
    require(
        queue.get("candidate_admission_sha256") == actual_admission_sha, "P2 queue is bound to a different admission"
    )
    require(
        queue.get("status") == "completed" and queue.get("exit_code") == 0,
        "P2 must finish successfully before this candidate",
    )
    require(
        completion.get("status") == "completed" and completion.get("exit_code") == 0, "P2 completion is not successful"
    )
    launch = read(P2 / "launch.json")
    require(launch.get("candidate_admission_sha256") == actual_admission_sha, "P2 launch used a different admission")
    require(
        completion.get("candidate_admission_sha256") == actual_admission_sha, "P2 completion used a different admission"
    )
    require(
        launch.get("command") == p2command and completion.get("command") == p2command,
        "P2 launch/completion command differs from its frozen command",
    )
    review_path = HERE / "predecessors/p2-forecast-review.json"
    require(review_path.is_file(), "root P2 forecast reconciliation is required")
    review = read(review_path)
    require(
        review.get("reviewer") == "root" and review.get("accepted") is True,
        "root has not accepted P2 forecast reconciliation",
    )
    require(
        review.get("completion_sha256") == sha(completion_path)
        and review.get("admission_sha256") == actual_admission_sha,
        "P2 forecast review is not bound to the terminal P2 run",
    )
    elapsed = completion.get("elapsed_seconds")
    bounds = launch.get("forecast", {}).get("forecast_seconds")
    require(elapsed is not None and isinstance(bounds, dict), "P2 elapsed/forecast missing")
    lo, hi = bounds.get("low"), bounds.get("high")
    require(lo is not None and hi is not None and lo <= hi, "P2 forecast bounds invalid")
    in_range = lo <= elapsed <= hi
    validate_forecast_review(review, (lo, hi), elapsed)
    if not in_range:
        miss = P2 / "forecast-miss-accepted.json"
        require(miss.is_file(), "P2 forecast miss needs the P2 root acceptance record")
        m = read(miss)
        require(
            m.get("reviewer") == "root"
            and m.get("accepted") is True
            and m.get("forecast_status") == "accepted_miss"
            and m.get("forecast_range_seconds") == [lo, hi]
            and m.get("contention_disclosed") is True
            and m.get("speed_ranking_used") is False
            and m.get("completion_sha256") == sha(completion_path)
            and m.get("admission_sha256") == actual_admission_sha
            and m.get("measured_seconds") == elapsed,
            "P2 forecast miss acceptance does not match terminal bytes",
        )
        require(review.get("forecast_status") == "accepted_miss", "root review does not accept P2 forecast miss")
    return {
        "admission_sha256": actual_admission_sha,
        "completion_sha256": sha(completion_path),
        "queue_sha256": sha(queue_path),
        "forecast_status": review["forecast_status"],
        "forecast_review_sha256": sha(review_path),
        "elapsed_seconds": elapsed,
        "forecast_bounds_seconds": [lo, hi],
    }


def sklearn_serial_gate():
    review_path = HERE / "predecessors/sklearn-serial-review.json"
    require(review_path.is_file(), "root Sklearn/HF serialization review is required")
    review = read(review_path)
    require(
        review.get("reviewer") == "root" and review.get("accepted") is True,
        "root Sklearn/HF serialization review is not accepted",
    )
    validate_hf_followup(review)
    return {"review_sha256": sha(review_path), "status": review.get("status"), "hf_followup": review["hf_followup"]}


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


def forecast_for(mode):
    forecast = read(HERE / "forecast.json")
    return forecast["mode_estimates"][mode]


def check_mode_predecessors(mode):
    p2 = terminal_p2()
    sklearn = sklearn_serial_gate()
    earlier = None
    if mode == "full":
        served = HERE / "completion-served.json"
        launch = HERE / "launch-served.json"
        require(served.is_file() and launch.is_file(), "served mode must finish before full mode")
        scomp, sl = read(served), read(launch)
        require(
            scomp.get("status") == "completed" and scomp.get("exit_code") == 0,
            "served mode did not complete successfully",
        )
        review_path = HERE / "predecessors/served-forecast-review.json"
        require(review_path.is_file(), "root served forecast review is required before full mode")
        sr = read(review_path)
        require(
            sr.get("reviewer") == "root"
            and sr.get("accepted") is True
            and sr.get("completion_sha256") == sha(served)
            and sr.get("forecast_status") in ("in_range", "accepted_miss"),
            "served forecast review must accept exact completion and reconcile forecast",
        )
        earlier = {
            "served_completion_sha256": sha(served),
            "served_launch_sha256": sha(launch),
            "served_review_sha256": sha(review_path),
        }
    return {"p2": p2, "sklearn": sklearn, "earlier_mode": earlier}


def validate_static():
    _, inventory = candidate_pins(False)
    commands = read(HERE / "commands.json")
    loaded = {}
    for mode in ("served", "full"):
        spec = commands[mode]
        require(spec["manifest_sha256"] == sha(ROOT / spec["manifest"]), f"{mode} manifest pin mismatch")
        require(spec["runner_sha256"] == sha(ROOT / spec["runner"]), f"{mode} runner pin mismatch")
        require(spec["timeout_seconds"] == 600, f"{mode} timeout changed")
        require(spec["execution_command"][0:3] == ["timeout", "600", "env"], f"{mode} timeout/env mismatch")
        require(
            any(x.startswith("PYTHONPATH=") and str(ROOT) in x for x in spec["execution_command"]),
            f"{mode} missing explicit PYTHONPATH",
        )
        require(
            "READINESS_TRACE_REGION_SIZE=134217728" in spec["execution_command"], f"{mode} missing trace region setting"
        )
        loaded[mode] = [read(ROOT / job["case"]) for job in read(ROOT / spec["manifest"])]
    for index, seed in enumerate((42, 43, 44)):
        a, b = loaded["served"][index], loaded["full"][index]
        require(a["metadata"]["seed"] == b["metadata"]["seed"] == seed, "paired mode seed mismatch")
        require(
            a["prompt_token_ids"] == b["prompt_token_ids"]
            and a["continuations"]["chosen"] == b["continuations"]["chosen"],
            "served/full input or chosen branch mismatch",
        )
        require(
            len(a["prompt_token_ids"]) == 512 and len(a["continuations"]["chosen"]) == 64,
            "candidate input lengths changed",
        )
    return {
        "source_count": len(inventory["files"]),
        "served_jobs": 3,
        "full_jobs": 3,
        "stop_present": STOP.is_file(),
        "admission_present": (HERE / "admission.json").exists(),
    }


def self_test():
    global HERE, ROOT, P2, P2_ADMISSION_SHA
    assert Path.cwd().resolve() == ROOT
    try:
        require(False, "mock")
    except RuntimeError:
        pass
    else:
        raise AssertionError("gate helper failed")
    # Exercise the root-review state transitions without touching filesystem or devices.
    for state in ("completed", "not_required", "deferred_until_matched_TT"):
        validate_hf_followup({"hf_followup": state})
    for state in ("pending", "active", "started"):
        try:
            validate_hf_followup({"hf_followup": state})
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid HF state accepted: {state}")
    validate_stop_state(True, False)
    try:
        validate_stop_state(True, True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("STOP did not block execution")
    validate_stop_state(False, True)
    good = {
        "reviewer": "root",
        "accepted": True,
        "forecast_range_seconds": [10, 20],
        "reviewed_elapsed_seconds": 25,
        "contention_disclosed": True,
        "speed_ranking_used": False,
        "forecast_status": "accepted_miss",
    }
    validate_forecast_review(good, (10, 20), 25)
    for bad in (
        {**good, "forecast_range_seconds": [11, 20]},
        {**good, "reviewer": "other"},
        {**good, "forecast_status": "in_range"},
        {**good, "contention_disclosed": False},
        {**good, "speed_ranking_used": True},
    ):
        try:
            validate_forecast_review(bad, (10, 20), 25)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid forecast review accepted")
    # Mock the full P2 terminal-evidence path, including a successful range hit and tamper rejection.
    old = (HERE, ROOT, P2, P2_ADMISSION_SHA)
    try:
        with tempfile.TemporaryDirectory(prefix="candidate-gate-mock-") as td:
            root = Path(td)
            candidate, p2 = root / "candidate", root / "p2"
            (candidate / "predecessors").mkdir(parents=True)
            p2.mkdir()
            for name, content in (("proposal.json", "{}"), ("run_after_p1_x.py", "# mock\n")):
                (p2 / name).write_text(content)
            (root / "mock-source.json").write_text('{"mock":true}\n')
            mock_inventory = {
                "files": {"source_0": {"path": "mock-source.json", "sha256": sha(root / "mock-source.json")}}
            }
            (p2 / "source-hashes.json").write_text(json.dumps(mock_inventory) + "\n")
            command = ["timeout", "1", "env", "MOCK=1"]
            (p2 / "command.json").write_text(json.dumps({"argv": command}))
            admission = {
                "status": "admitted",
                "proposal_sha256": sha(p2 / "proposal.json"),
                "driver_sha256": sha(p2 / "run_after_p1_x.py"),
                "source_inventory_sha256": sha(p2 / "source-hashes.json"),
            }
            (p2 / "admission.json").write_text(json.dumps(admission))
            admission_sha = sha(p2 / "admission.json")
            (p2 / "queue.json").write_text(
                json.dumps({"status": "completed", "exit_code": 0, "candidate_admission_sha256": admission_sha})
            )
            forecast = {"forecast_seconds": {"low": 10, "high": 20}}
            launch = {"candidate_admission_sha256": admission_sha, "command": command, "forecast": forecast}
            (p2 / "launch.json").write_text(json.dumps(launch))
            completion = {
                "status": "completed",
                "exit_code": 0,
                "candidate_admission_sha256": admission_sha,
                "command": command,
                "elapsed_seconds": 15,
            }
            completion_path = p2 / "completion.json"
            completion_path.write_text(json.dumps(completion))
            review = {
                "reviewer": "root",
                "accepted": True,
                "completion_sha256": sha(completion_path),
                "admission_sha256": admission_sha,
                "forecast_status": "in_range",
                "forecast_range_seconds": [10, 20],
                "reviewed_elapsed_seconds": 15,
                "contention_disclosed": True,
                "speed_ranking_used": False,
            }
            (candidate / "predecessors/p2-forecast-review.json").write_text(json.dumps(review))
            HERE, ROOT, P2, P2_ADMISSION_SHA = candidate, root, p2, ""
            assert terminal_p2()["forecast_status"] == "in_range"
            completion["elapsed_seconds"] = 25
            completion_path.write_text(json.dumps(completion))
            review.update(
                {
                    "completion_sha256": sha(completion_path),
                    "forecast_status": "accepted_miss",
                    "reviewed_elapsed_seconds": 25,
                }
            )
            (candidate / "predecessors/p2-forecast-review.json").write_text(json.dumps(review))
            miss = {
                "reviewer": "root",
                "accepted": True,
                "forecast_status": "accepted_miss",
                "forecast_range_seconds": [10, 20],
                "contention_disclosed": True,
                "speed_ranking_used": False,
                "completion_sha256": sha(completion_path),
                "admission_sha256": admission_sha,
                "measured_seconds": 25,
            }
            (p2 / "forecast-miss-accepted.json").write_text(json.dumps(miss))
            assert terminal_p2()["forecast_status"] == "accepted_miss"
            miss["forecast_range_seconds"] = [10, 21]
            (p2 / "forecast-miss-accepted.json").write_text(json.dumps(miss))
            try:
                terminal_p2()
            except RuntimeError:
                pass
            else:
                raise AssertionError("P2 miss record with wrong forecast range was accepted")
            queue = {"status": "completed", "exit_code": 0, "candidate_admission_sha256": "wrong"}
            (p2 / "queue.json").write_text(json.dumps(queue))
            try:
                terminal_p2()
            except RuntimeError:
                pass
            else:
                raise AssertionError("P2 queue/admission mismatch was accepted")
    finally:
        HERE, ROOT, P2, P2_ADMISSION_SHA = old
    print("candidate gate self-test passed; no TTNN import or device query")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("served", "full"), required=False)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    static = validate_static()
    if args.validate_only:
        print(json.dumps(static, sort_keys=True))
        return
    require(args.execute and args.mode, "explicit --mode and --execute required")
    adm, _ = candidate_pins(True)
    validate_stop_state(STOP.exists(), executing=True)
    gates = check_mode_predecessors(args.mode)
    hf = active_hf_processes()
    require(not hf, f"active HF model process(es): {hf}")
    owners = device_owners()
    require(not owners, f"device owner(s) present: {owners}")
    launch_path, completion_path = HERE / f"launch-{args.mode}.json", HERE / f"completion-{args.mode}.json"
    require(not launch_path.exists() and not completion_path.exists(), "duplicate mode launch refused")
    outdir = HERE / "runs" / args.mode
    require(not outdir.exists() or not any(outdir.iterdir()), "nonempty output dir refused")
    outdir.mkdir(parents=True, exist_ok=True)
    spec = read(HERE / "commands.json")[args.mode]
    command = spec["execution_command"]
    require(command[0:3] == ["timeout", "600", "env"], "frozen timeout command is not600s")
    launch = {
        "status": "running",
        "mode": args.mode,
        "command": command,
        "started_unix": time.time(),
        "timeout_seconds": 600,
        "candidate_admission_sha256": sha(HERE / "admission.json"),
        "candidate_proposal_sha256": sha(HERE / "proposal.json"),
        "manifest_sha256": spec["manifest_sha256"],
        "runner_sha256": spec["runner_sha256"],
        "forecast": forecast_for(args.mode),
        "predecessor_gates": gates,
        "active_hf_processes": hf,
        "device_owner_preflight": owners,
    }
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")
    with (HERE / f"panel-{args.mode}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=610, check=False)
    outputs = []
    valid = result.returncode == 0
    if valid:
        jobs = read(ROOT / spec["manifest"])
        for job in jobs:
            p = outdir / (job["name"] + ".json")
            require(p.is_file() and p.stat().st_size > 0, f"missing/empty output: {p}")
            data = read(p)
            fixture = read(ROOT / job["case"])
            chosen = data.get("continuations", {}).get("chosen", {})
            require(
                chosen.get("tokens") == 64 and chosen.get("conditioning_token_ids") == fixture["prompt_token_ids"],
                f"chosen branch incomplete or mismatched input: {job['name']}",
            )
            fr = data.get("free_running", [])
            require(
                len(fr) == 1 and 0 < len(fr[0].get("token_ids", [])) <= 512, f"free run missing/invalid: {job['name']}"
            )
            require(
                fr[0].get("sampling", {}).get("seed") == fixture["metadata"]["seed"],
                f"free-run seed mismatch: {job['name']}",
            )
            outputs.append(
                {"path": str(p.relative_to(ROOT)), "sha256": sha(p), "free_run_tokens": len(fr[0]["token_ids"])}
            )
    completion = {
        **launch,
        "status": "completed" if valid else "failed",
        "ended_unix": time.time(),
        "elapsed_seconds": time.time() - launch["started_unix"],
        "exit_code": result.returncode,
        "outputs": outputs,
    }
    completion_path.write_text(json.dumps(completion, indent=2) + "\n")
    if not valid:
        raise RuntimeError(f"{args.mode} run exited with code {result.returncode}; no retry")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "status": completion["status"],
                "elapsed_seconds": completion["elapsed_seconds"],
                "outputs": len(outputs),
            }
        )
    )


if __name__ == "__main__":
    main()
