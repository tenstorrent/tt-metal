"""Coordinator-authorized serial host diagnostics with per-container admission."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import time

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
B = D / "anomalies/swe85-below-band"
O = B / "fetch-semantic-validation"
O.mkdir(exist_ok=True)
os.umask(0o077)
parser = argparse.ArgumentParser()
parser.add_argument("--attempt", type=int, default=1)
parser.add_argument("--tasks", nargs="*")
args = parser.parse_args()
attempt = "attempt" + str(args.attempt)
started = time.monotonic()
waited = 0
done = []
snap = {
    p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (B / "fetch-controls").glob("*/manifest.json")
}


def save():
    p = O / ("batch-progress-" + attempt + ".json")
    p.write_text(json.dumps({"started_monotonic": started, "waited_seconds": waited, "tasks": done}, indent=2) + "\n")
    p.chmod(0o600)


for task, digest in sorted(snap.items()):
    if args.tasks and task not in args.tasks:
        continue
    output = O / (task + "-" + attempt)
    if output.exists():
        done.append({"task": task, "status": "existing_attempt_preserved"})
        save()
        continue
    while True:
        complete = 0
        for f in (D / "setup/swe-jobs/swe-extended85-head-bfp8").glob("*/result.json"):
            try:
                complete += bool(json.loads(f.read_text()).get("finished_at"))
            except (ValueError, OSError):
                pass
        ps = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=30)
        names = [n for n in ps.stdout.splitlines() if n != "romantic_chebyshev"]
        lock = (D / "setup/.semantic-verifier-worker.lock").open("a")
        free = True
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            free = False
        finally:
            lock.close()
        admission = {
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "completed": complete,
            "active_excluding_romantic": len(names),
            "active_names": names,
            "semantic_lock_free": free,
        }
        if ps.returncode == 0 and complete >= 75 and len(names) <= 10 and free:
            break
        print("FETCH_WAIT", task, complete, len(names), free, flush=True)
        if waited >= 7200:
            done.append({"task": task, "status": "two_hour_admission_wait_exhausted", "admission": admission})
            save()
            raise SystemExit(2)
        time.sleep(20)
        waited += 20
    p = B / "fetch-controls" / task / "manifest.json"
    if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        done.append({"task": task, "status": "changing_manifest_skipped", "admission": admission})
        save()
        continue
    print("FETCH_START", task, flush=True)
    command = [
        "python",
        str(D / "setup/validate_fetch_next_action.py"),
        "--task",
        task,
        "--execute",
        "--output",
        str(output),
    ]
    # Driver limits execution300s plus cleanup60s and saves progress before removal.
    r = subprocess.run(command, capture_output=True, text=True)
    (O / (task + "-" + attempt + ".driver.stdout.private.txt")).write_text(r.stdout)
    (O / (task + "-" + attempt + ".driver.stderr.private.txt")).write_text(r.stderr)
    result = (
        json.loads((output / "semantic-validation.json").read_text())
        if (output / "semantic-validation.json").exists()
        else {}
    )
    done.append(
        {
            "task": task,
            "admission": admission,
            "returncode": r.returncode,
            "status": result.get("status", "driver_failed"),
            "valid_next_action": result.get("valid_next_action"),
            "error": result.get("error"),
            "output": str(output),
            "manifest_sha256": digest,
        }
    )
    save()
    print("FETCH_DONE", task, r.returncode, result.get("valid_next_action"), result.get("error"), flush=True)
    if result.get("status") == "validation_error":
        print("STOP_INFRA_ERROR", task, flush=True)
        break
print("FETCH_BATCH_DONE", len(done), flush=True)
