import datetime
import json
import pathlib
import subprocess
import sys
import time

D = pathlib.Path(__file__).resolve().parent.parent
out = D / "current/swe-extended85-head-bfp8/verifier-recovery/pylint4661-attempt1"
assert not out.exists()
start = time.time()
deadline = start + 7200
while time.time() < deadline:
    r = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True, timeout=30
    )
    names = [n for n in r.stdout.splitlines() if n != "romantic_chebyshev"]
    completed = len(list((D / "setup/swe-jobs/swe-extended85-head-bfp8").glob("*/result.json")))
    if completed >= 74 and len(names) <= 11:
        admission = {
            "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "active_other_containers": names,
            "count": len(names),
            "max_workers": 12,
            "scope": "One verifier recovery worker; no TT access",
            "wait_seconds": time.time() - start,
        }
        record = D / "setup/pylint4661-recovery-admission.json"
        record.write_text(json.dumps(admission, indent=2) + "\n")
        cmd = [sys.executable, str(D / "setup/recover_pylint4661_verifier.py"), "--execute", "--output", str(out)]
        p = subprocess.run(cmd, timeout=1900)
        raise SystemExit(p.returncode)
    time.sleep(30)
raise SystemExit("No host worker slot within7200seconds; recovery not launched")
