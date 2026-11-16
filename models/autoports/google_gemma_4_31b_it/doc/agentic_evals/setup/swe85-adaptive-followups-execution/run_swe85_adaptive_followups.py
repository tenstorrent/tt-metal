"""Run the reviewed adaptive suffix followups only after the cap2048 short probe."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[7]
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
OUT = D / "setup/swe85-adaptive-followups-execution"
ADMISSION = OUT / "admission-draft.json"
STOP = OUT / "STOP_BEFORE_NEXT_JOB"
PREDECESSOR = D / "setup/swe85-persistent-short-controls/cap1024-preparation/cap2048-preparation/full-completion.json"
PREDECESSOR_ADMISSION = (
    D / "setup/swe85-persistent-short-controls/cap1024-preparation/cap2048-preparation/admission.json"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    admission_hash = sha(ADMISSION)
    admission = json.loads(ADMISSION.read_text())
    assert admission["status"] == "admitted", "Coordinator must explicitly admit this draft first"
    assert sha(Path(__file__)) == admission["driver_sha256"], "Driver changed after review"
    assert sha(PREDECESSOR_ADMISSION) == admission["prerequisite"]["predecessor_admission_sha256"]
    assert not (OUT / "execution-queue.json").exists(), "Refuse duplicate queue"
    state = {
        "status": "waiting_for_cap2048_short_full_completion",
        "queued_unix": time.time(),
        "admission_sha256": admission_hash,
        "stop_file": str(STOP),
    }
    write(OUT / "execution-queue.json", state)
    deadline = time.monotonic() + 39600
    while not PREDECESSOR.exists():
        if STOP.exists():
            state.update(status="stopped_before_followups", end_unix=time.time())
            write(OUT / "execution-queue.json", state)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("No cap2048 short full completion within eleven hours")
        time.sleep(10)
    completion = json.loads(PREDECESSOR.read_text())
    assert completion["exit_code"] == 0, "Predecessor failed; do not launch followups"
    sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
    import benchmark_server

    for job in admission["jobs"]:
        if STOP.exists():
            state.update(status="stopped_before_" + job["name"], end_unix=time.time())
            write(OUT / "execution-queue.json", state)
            return
        assert sha(ADMISSION) == admission_hash, "Admission changed while queued"
        for evidence in job["hashes"]:
            assert sha(ROOT / evidence["path"]) == evidence["sha256"], evidence["path"]
        assert not benchmark_server.device_owners(), "Device still owned; do not launch"
        assert job["command"][:2] == ["timeout", "3600"]
        launch_path = OUT / (job["name"] + "-launch.json")
        assert not launch_path.exists(), "Refuse duplicate execution"
        launch = {
            "name": job["name"],
            "command": job["command"],
            "start_unix": time.time(),
            "device_owner_preflight": [],
            "prediction": job["prediction"],
            "admission_sha256": admission_hash,
            "scope": "Diagnostic host-wall, not serving performance.",
        }
        write(launch_path, launch)
        state.update(status=job["name"] + "_started", start_unix=launch["start_unix"])
        write(OUT / "execution-queue.json", state)
        with (OUT / (job["name"] + ".log")).open("w") as log:
            result = subprocess.run(job["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        launch.update(end_unix=time.time(), exit_code=result.returncode)
        launch["measured_hours"] = (launch["end_unix"] - launch["start_unix"]) / 3600
        low, high = job["prediction"]["range_hours"]
        launch["prediction_range_hit"] = low <= launch["measured_hours"] <= high
        write(OUT / (job["name"] + "-completion.json"), launch)
        state.update(status=job["name"] + "_completed", exit_code=result.returncode, end_unix=time.time())
        write(OUT / "execution-queue.json", state)
        assert result.returncode == 0, "Job failed; do not launch subsequent device work"
    state.update(status="all_followups_completed", end_unix=time.time())
    write(OUT / "execution-queue.json", state)


if __name__ == "__main__":
    main()
