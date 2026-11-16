#!/usr/bin/env python3
"""Inert-until-admitted U Requests cap2048 extension queue, gated after V."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
OUT = D / "setup/swe85-adaptive-req5414-cap2048-after-v-candidate"
ADMISSION = OUT / "admission.json"
PROPOSAL = OUT / "proposal.json"
PREDECESSOR = D / "setup/swe85-adaptive-suffix4096-candidate/full-completion.json"
PREDECESSOR_QUEUE = D / "setup/swe85-adaptive-suffix4096-candidate/queue.json"
PREDECESSOR_ADMISSION = D / "setup/swe85-adaptive-suffix4096-candidate/admission.json"
STOP = OUT / "STOP_BEFORE_NEXT_JOB"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


def main():
    admission = json.loads(ADMISSION.read_text())
    assert admission.get("status") == "admitted", "root admission required"
    ah = sha(ADMISSION)
    proposal = json.loads(PROPOSAL.read_text())
    assert sha(Path(__file__)) == admission["driver_sha256"]
    assert sha(PROPOSAL) == admission["proposal_sha256"]
    state = {
        "status": "waiting_for_V_suffix4096_success",
        "admission_sha256": ah,
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
            raise TimeoutError("V suffix4096 completion missing after eleven hours")
        time.sleep(10)
    pred = json.loads(PREDECESSOR.read_text())
    predq = json.loads(PREDECESSOR_QUEUE.read_text())
    expected_pred_admission = proposal["prerequisite"]["admission_sha256_expected"]
    assert sha(PREDECESSOR_ADMISSION) == expected_pred_admission, "V admitted package hash changed"
    assert pred.get("name") == "full" and pred.get("exit_code") == 0, "V full batch must succeed"
    assert (
        predq.get("status") == "all_suffix4096_jobs_completed" and predq.get("exit_code") == 0
    ), "V queue must explicitly report successful terminal status"
    state.update(
        predecessor_admission_sha256=sha(PREDECESSOR_ADMISSION),
        predecessor_completion_sha256=sha(PREDECESSOR),
        predecessor_queue_sha256=sha(PREDECESSOR_QUEUE),
    )
    sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
    import benchmark_server

    for job in proposal["jobs"]:
        if STOP.exists():
            state.update(status="stopped_before_" + job["name"], end_unix=time.time())
            write(OUT / "queue.json", state)
            return
        assert sha(ADMISSION) == ah and sha(PROPOSAL) == admission["proposal_sha256"]
        assert sha(PREDECESSOR_ADMISSION) == proposal["prerequisite"]["admission_sha256_expected"]
        assert (
            sha(PREDECESSOR) == state["predecessor_completion_sha256"]
        ), "V completion changed after prerequisite check"
        assert sha(PREDECESSOR_QUEUE) == state["predecessor_queue_sha256"], "V queue changed after prerequisite check"
        for item in proposal["source_hashes"]:
            assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
        for item in proposal["candidate_manifests"].values():
            assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
        assert sha(PREDECESSOR_ADMISSION) == proposal["prerequisite"]["admission_sha256_expected"]
        assert not benchmark_server.device_owners(), "device owned; stop"
        launch = OUT / (job["name"] + "-launch.json")
        assert not launch.exists(), "refuse duplicate launch"
        rec = {
            "name": job["name"],
            "command": job["command"],
            "start_unix": time.time(),
            "device_owner_preflight": [],
            "prediction": job["prediction"],
            "admission_sha256": ah,
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
    state.update(status="all_req5414_cap2048_extensions_completed", end_unix=time.time())
    write(OUT / "queue.json", state)


if __name__ == "__main__":
    main()
