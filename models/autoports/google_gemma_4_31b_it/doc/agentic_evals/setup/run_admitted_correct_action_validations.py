"""Admitted serial before/after diagnostics. Original test scripts remain unchanged."""
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
O = B / "correct-actions-semantic"
O.mkdir(exist_ok=True)
os.umask(0o077)
parser = argparse.ArgumentParser()
parser.add_argument("--attempt", type=int, default=1)
parser.add_argument("--tasks", nargs="*")
parser.add_argument("--wheelhouse")
parser.add_argument("--collections-callable-compat", action="store_true")
args = parser.parse_args()
attempt = "attempt" + str(args.attempt)
rows = json.loads((B / "correct-actions/all-candidate-action-cases.json").read_text())
snap = {
    pathlib.Path(r["case"])
    .parent.name: hashlib.sha256((pathlib.Path(r["case"]).parent / "manifest.json").read_bytes())
    .hexdigest()
    for r in rows
}
done = []
waited = 0


def save():
    p = O / ("batch-progress-" + attempt + ".json")
    p.write_text(json.dumps({"tasks": done, "waited_seconds": waited}, indent=2) + "\n")
    p.chmod(0o600)


for candidate, digest in snap.items():
    if args.tasks and candidate not in args.tasks:
        continue
    mp = B / "correct-actions" / candidate / "manifest.json"
    manifest = json.loads(mp.read_text())
    network = "bridge" if manifest["task"].startswith("psf__requests-") else "none"
    for state in ["before", "after"]:
        output = O / (candidate + "-" + attempt) / state
        if output.exists():
            done.append({"candidate": candidate, "state": state, "status": "existing_attempt_preserved"})
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
            print("CORRECT_WAIT", candidate, state, complete, len(names), free, flush=True)
            if waited >= 7200:
                done.append(
                    {
                        "candidate": candidate,
                        "state": state,
                        "status": "admission_wait_exhausted",
                        "admission": admission,
                    }
                )
                save()
                raise SystemExit(2)
            time.sleep(20)
            waited += 20
        if hashlib.sha256(mp.read_bytes()).hexdigest() != digest:
            done.append(
                {"candidate": candidate, "state": state, "status": "changing_manifest_skipped", "admission": admission}
            )
            save()
            break
        print("CORRECT_START", candidate, state, network, flush=True)
        command = [
            "python",
            str(D / "setup/validate_correct_action_semantics.py"),
            "--task",
            candidate,
            "--state",
            state,
            "--network",
            network,
            "--execute",
            "--output",
            str(output),
        ]
        if args.collections_callable_compat:
            command += ["--collections-callable-compat"]
        if args.wheelhouse:
            command += ["--wheelhouse", args.wheelhouse]
        r = subprocess.run(command, capture_output=True, text=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        (output.parent / (state + ".driver.stdout.private.txt")).write_text(r.stdout)
        (output.parent / (state + ".driver.stderr.private.txt")).write_text(r.stderr)
        pp = output / "semantic-validation-provenance.json"
        provenance = json.loads(pp.read_text()) if pp.exists() else {}
        rp = output / "verifier/report.json"
        report = json.loads(rp.read_text()) if rp.exists() else {}
        result = report.get(manifest["task"], {})
        done.append(
            {
                "candidate": candidate,
                "state": state,
                "admission": admission,
                "network": network,
                "returncode": r.returncode,
                "status": provenance.get("status", "driver_failed"),
                "error": provenance.get("error"),
                "resolved": result.get("resolved"),
                "tests_status": result.get("tests_status"),
                "output": str(output),
                "manifest_sha256": digest,
            }
        )
        save()
        print(
            "CORRECT_DONE", candidate, state, r.returncode, result.get("resolved"), provenance.get("error"), flush=True
        )
        if provenance.get("status") in ["recovery_error", "incomplete_no_report"] or not provenance:
            print("STOP_INFRA_ERROR", candidate, state, flush=True)
            raise SystemExit(1)
print("CORRECT_BATCH_DONE", len(done), flush=True)
