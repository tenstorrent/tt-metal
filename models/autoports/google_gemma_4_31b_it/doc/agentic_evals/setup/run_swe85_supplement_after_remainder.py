"""Run reviewed semantic supplements only after the existing device queue finishes."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
S = D / "setup/swe85-semantic-supplement"
STOP = S / "STOP_BEFORE_NEXT_JOB"
QUEUE = S / "execution-queue.json"
state = {
    "status": "waiting_for_full_remainder_completion",
    "queued_unix": time.time(),
    "stop_file": str(STOP),
    "rule": "Stop only between jobs; never signal active device work.",
}


def save():
    QUEUE.write_text(json.dumps(state, indent=2) + "\n")


save()
completion = D / "setup/swe85-remainder-execution85/full-completion.json"
deadline = time.monotonic() + 39600
while not completion.exists():
    if STOP.exists():
        state.update(status="stopped_before_supplement", end_unix=time.time())
        save()
        sys.exit(0)
    if time.monotonic() >= deadline:
        raise TimeoutError("No full remainder completion within eleven-hour queue window")
    time.sleep(10)
assert json.loads(completion.read_text())["exit_code"] == 0
sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
import benchmark_server

admission = json.loads((S / "admission.json").read_text())
manifest = S / "admitted-cases.json"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == admission["manifest_sha256"]
for mode in ("served", "full"):
    if STOP.exists():
        state.update(status="stopped_before_" + mode, end_unix=time.time())
        save()
        sys.exit(0)
    assert not benchmark_server.device_owners(), "Device still owned; no launch"
    launch_path = S / (mode + "-launch.json")
    assert not launch_path.exists(), "Refuse duplicate supplemental execution"
    command = admission["staged_commands_not_executed"][mode]
    assert command[:2] == ["timeout", "3600"]
    launch = {
        "mode": mode,
        "command": command,
        "start_unix": time.time(),
        "manifest_sha256": admission["manifest_sha256"],
        "device_owner_preflight": [],
        "scope": "Four reviewed semantic controls; diagnostic times are not serving performance.",
    }
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")
    state.update(status=mode + "_started", start_unix=launch["start_unix"])
    save()
    with (S / (mode + ".log")).open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    launch.update(end_unix=time.time(), exit_code=result.returncode)
    (S / (mode + "-completion.json")).write_text(json.dumps(launch, indent=2) + "\n")
    state.update(status=mode + "_completed", exit_code=result.returncode, end_unix=time.time())
    save()
    assert result.returncode == 0, "Supplement failed; subsequent job not launched"
state.update(status="both_supplement_modes_completed", end_unix=time.time())
save()
print("Supplement served/full completed")
