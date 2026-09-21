"""Read-only bounded hardware sampling; no clock/power controls."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--seconds", type=int, default=120)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
assert not args.output.exists(), "Use a new telemetry filename"
samples = []
start = time.monotonic()
while time.monotonic() - start < args.seconds:
    item = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        result = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, timeout=20)
        item.update(returncode=result.returncode, stderr=result.stderr)
        if result.returncode == 0:
            item["snapshot"] = json.loads(result.stdout)
        else:
            item["stdout"] = result.stdout
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        item["error"] = str(exc)
    samples.append(item)
    args.output.write_text(json.dumps(samples, indent=2) + "\n")
    time.sleep(5)
print(json.dumps({"samples": len(samples), "output": str(args.output)}))
