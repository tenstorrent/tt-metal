"""Queue one bounded full-precision diagnostic after the served job releases its devices."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
OUT = D / "setup/swe85-remainder-execution85"
STOP = OUT / "STOP_BEFORE_FULL"
record_path = OUT / "full-queue.json"
record = {
    "status": "waiting_for_served_completion",
    "queued_unix": time.time(),
    "stop_file": str(STOP),
    "rule": "No active device job is signaled; stop file only prevents next launch.",
}
record_path.write_text(json.dumps(record, indent=2) + "\n")
deadline = time.monotonic() + 15000
while not (OUT / "served-completion.json").exists():
    if STOP.exists():
        record.update(status="stopped_before_full", end_unix=time.time())
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        sys.exit(0)
    if time.monotonic() > deadline:
        raise TimeoutError("No served completion record within bounded queue window")
    time.sleep(10)
served = json.loads((OUT / "served-completion.json").read_text())
assert served["exit_code"] == 0, "Served diagnostic failed; full job not started"
assert not STOP.exists(), "Full job stopped at inter-job boundary"
sys.path.insert(0, str(ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
import benchmark_server

assert not benchmark_server.device_owners(), "Device owner remains; no full launch"
split = json.loads((D / "setup/swe85-pilot-split85/split-report.json").read_text())
manifest = D / "setup/swe85-pilot-split85/rest.json"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == served["manifest_sha256"]
command = split["staged_commands_not_executed"]["rest"]["full"]["argv"]
assert command[:2] == ["timeout", "14400"]
command[1] = "21600"
assert not (OUT / "full-launch.json").exists(), "Refuse duplicate execution"
launch = {
    "command": command,
    "start_unix": time.time(),
    "device_owner_preflight": [],
    "manifest_sha256": served["manifest_sha256"],
    "prediction_hours": 3.576149232076101,
    "timeout_override_basis": "setup/swe85-remainder-cost-gate.json",
    "scope": "Existing160 controls; supplemental semantic gaps tracked separately.",
}
(OUT / "full-launch.json").write_text(json.dumps(launch, indent=2) + "\n")
record.update(status="full_started", start_unix=launch["start_unix"])
record_path.write_text(json.dumps(record, indent=2) + "\n")
with (OUT / "full.log").open("w") as log:
    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
launch.update(end_unix=time.time(), exit_code=result.returncode)
(OUT / "full-completion.json").write_text(json.dumps(launch, indent=2) + "\n")
record.update(status="full_completed", end_unix=time.time(), exit_code=result.returncode)
record_path.write_text(json.dumps(record, indent=2) + "\n")
print("Full remainder exit", result.returncode)
sys.exit(result.returncode)
