#!/usr/bin/env python3
"""Wait for U success, then run unadmitted suffix4096 served/full diagnostics serially."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
OUT = D / "setup/swe85-adaptive-suffix4096-candidate"
ADMISSION = OUT / "admission.json"
PROPOSAL = OUT / "proposal.json"
PREDECESSOR = D / "setup/swe85-adaptive-suffix2048-cap1024-execution-candidate/full-completion.json"
PREDECESSOR_QUEUE = D / "setup/swe85-adaptive-suffix2048-cap1024-execution-candidate/queue.json"
STOP = OUT / "STOP_BEFORE_NEXT_JOB"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


def main():
    admission = json.loads(ADMISSION.read_text())
    assert admission.get("status") == "admitted", "root admission required"
    proposal = json.loads(PROPOSAL.read_text())
    admitted_hash = sha(ADMISSION)
    assert sha(Path(__file__)) == admission["driver_sha256"]
    assert sha(PROPOSAL) == admission["proposal_sha256"]
    state = {
        "status": "waiting_for_U_cap1024_success",
        "admission_sha256": admitted_hash,
        "stop_file": str(STOP),
        "queued_unix": time.time(),
    }
    write(OUT / "queue.json", state)
    deadline = time.monotonic() + 39600
    while not (PREDECESSOR.exists() and PREDECESSOR_QUEUE.exists()):
        if STOP.exists():
            state.update(status="stopped_before_predecessor", end_unix=time.time())
            write(OUT / "queue.json", state)
            return
        if time.monotonic() > deadline:
            raise TimeoutError("U cap1024 completion missing after eleven hours")
        time.sleep(10)
    pred = json.loads(PREDECESSOR.read_text())
    predq = json.loads(PREDECESSOR_QUEUE.read_text())
    assert pred.get("name") == "full" and pred.get("exit_code") == 0, "U full batch must succeed"
    assert (
        predq.get("status") == "all_cap1024_batches_completed" and predq.get("exit_code") == 0
    ), "U queue must explicitly report successful terminal status"
    state["predecessor_completion_sha256"] = sha(PREDECESSOR)
    state["predecessor_queue_sha256"] = sha(PREDECESSOR_QUEUE)
    sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
    import benchmark_server

    for job in proposal["jobs"]:
        if STOP.exists():
            state.update(status="stopped_before_" + job["name"], end_unix=time.time())
            write(OUT / "queue.json", state)
            return
        assert sha(ADMISSION) == admitted_hash and sha(PROPOSAL) == admission["proposal_sha256"]
        for item in proposal["source_hashes"]:
            assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
        for item in proposal["candidate_manifests"].values():
            assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
        if job["name"] == "full":
            for case in json.loads((OUT / "served-cases.json").read_text()):
                assert (OUT / "runs/served" / (case["name"] + ".json")).exists(), case["name"]
        assert not benchmark_server.device_owners(), "device owned; stop"
        launch = OUT / (job["name"] + "-launch.json")
        assert not launch.exists(), "refuse duplicate launch"
        rec = {
            "name": job["name"],
            "command": job["command"],
            "start_unix": time.time(),
            "device_owner_preflight": [],
            "prediction": job["prediction"],
            "admission_sha256": admitted_hash,
            "predecessor_completion_sha256": state["predecessor_completion_sha256"],
        }
        write(launch, rec)
        state.update(status=job["name"] + "_started", start_unix=rec["start_unix"])
        write(OUT / "queue.json", state)
        with (OUT / (job["name"] + ".log")).open("w") as log:
            result = subprocess.run(job["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        rec.update(end_unix=time.time(), exit_code=result.returncode, measured_seconds=time.time() - rec["start_unix"])
        write(OUT / (job["name"] + "-completion.json"), rec)
        state.update(status=job["name"] + "_completed", exit_code=result.returncode, end_unix=time.time())
        write(OUT / "queue.json", state)
        if result.returncode:
            raise RuntimeError("batch failed; stop later work")
    compare_cmd = [str(ROOT / "python_env/bin/python"), str(OUT / "compare_suffix4096_outputs.py")]
    compare_result = subprocess.run(compare_cmd, cwd=ROOT, capture_output=True, text=True)
    state.update(
        comparison_returncode=compare_result.returncode,
        comparison_output=compare_result.stdout.strip(),
        comparison_stderr=compare_result.stderr.strip(),
    )
    write(OUT / "queue.json", state)
    if compare_result.returncode:
        raise RuntimeError("host-only matched-case comparison failed")
    state.update(status="all_suffix4096_jobs_completed", end_unix=time.time())
    write(OUT / "queue.json", state)


if __name__ == "__main__":
    main()
