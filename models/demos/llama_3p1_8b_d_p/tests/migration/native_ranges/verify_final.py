# Import-free adapter from the paired controller to the frozen range verifier.
import json
import subprocess
import sys
from pathlib import Path

OWNER = Path(__file__).resolve().parent


def verify(plan):
    done = subprocess.run(
        [sys.executable, "-B", str(OWNER / "verify_ranges.py"), "--plan", plan["plan_path"]],
        text=True,
        capture_output=True,
        timeout=1800,
    )
    if done.returncode:
        raise RuntimeError("range verifier failed: " + done.stderr[-4000:])
    return json.loads(done.stdout)
