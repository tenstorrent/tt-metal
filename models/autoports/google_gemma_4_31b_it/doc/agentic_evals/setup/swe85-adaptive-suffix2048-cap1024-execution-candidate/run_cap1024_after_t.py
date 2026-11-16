#!/usr/bin/env python3
"""Inert-until-admitted serial cap1024 follow-up driver."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
OUT = D / "setup/swe85-adaptive-suffix2048-cap1024-execution-candidate"
ADMISSION = OUT / "admission.json"
PROPOSAL = OUT / "proposal.json"
PREDECESSOR = D / "setup/swe85-third-cap-execution/full-completion.json"
PREDECESSOR_QUEUE = D / "setup/swe85-third-cap-execution/execution-queue.json"
STOP = OUT / "STOP_BEFORE_NEXT_JOB"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


def main():
    admission = json.loads(ADMISSION.read_text())
    assert admission.get("status") == "admitted", "root admission required"
    proposal = json.loads(PROPOSAL.read_text())
    ah = sha(ADMISSION)
    assert sha(Path(__file__)) == admission["driver_sha256"]
    assert sha(PROPOSAL) == admission["proposal_sha256"]
    state = {
        "status": "waiting_for_T_third_cap_completion",
        "admission_sha256": ah,
        "stop_file": str(STOP),
        "queued_unix": time.time(),
    }
    write(OUT / "queue.json", state)
    deadline = time.monotonic() + 39600
    while not PREDECESSOR.exists():
        if STOP.exists():
            state.update(status="stopped_before_predecessor", end_unix=time.time())
            write(OUT / "queue.json", state)
            return
        if time.monotonic() > deadline:
            raise TimeoutError("T third-cap completion missing after eleven hours")
        time.sleep(10)
    pred = json.loads(PREDECESSOR.read_text())
    queue = json.loads(PREDECESSOR_QUEUE.read_text())
    assert pred.get("name") == "full" and pred.get("exit_code") == 0, "T full completion must succeed"
    assert (
        queue.get("status") == "all_third_cap_jobs_completed" and queue.get("exit_code") == 0
    ), "T queue must explicitly report complete success"
    sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
    import benchmark_server

    for job in proposal["jobs"]:
        if STOP.exists():
            state.update(status="stopped_before_" + job["name"], end_unix=time.time())
            write(OUT / "queue.json", state)
            return
        assert sha(ADMISSION) == ah and sha(PROPOSAL) == admission["proposal_sha256"]
        for item in proposal["source_hashes"]:
            assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
        for item in proposal["candidate_manifests"].values():
            assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
        assert not benchmark_server.device_owners(), "device owned; stop"
        launch = OUT / (job["name"] + "-launch.json")
        assert not launch.exists()
        rec = {
            "name": job["name"],
            "command": job["command"],
            "start_unix": time.time(),
            "device_owner_preflight": [],
            "prediction_seconds": job["prediction_seconds"],
            "admission_sha256": ah,
        }
        write(launch, rec)
        state.update(status=job["name"] + "_started", start_unix=rec["start_unix"])
        write(OUT / "queue.json", state)
        with (OUT / (job["name"] + ".log")).open("w") as log:
            r = subprocess.run(job["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        rec.update(end_unix=time.time(), exit_code=r.returncode, measured_seconds=time.time() - rec["start_unix"])
        write(OUT / (job["name"] + "-completion.json"), rec)
        state.update(status=job["name"] + "_completed", exit_code=r.returncode, end_unix=time.time())
        write(OUT / "queue.json", state)
        if r.returncode:
            raise RuntimeError("batch failed; stop later work")
    state.update(status="all_cap1024_batches_completed", end_unix=time.time())
    write(OUT / "queue.json", state)


if __name__ == "__main__":
    main()
