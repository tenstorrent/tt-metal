#!/usr/bin/env python3
# Root-reviewed single-attempt wrapper; it never kills owners or resets devices.
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATTEMPT = None


def write(name, value):
    with (ATTEMPT / name).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    global ATTEMPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--plan-sha256", required=True)
    args = ap.parse_args()
    plan_path = args.plan.resolve()
    ATTEMPT = plan_path.parent
    raw = plan_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.plan_sha256:
        raise RuntimeError("reviewed plan bytes differ")
    plan = json.loads(raw)
    if plan.get("reviewed") is not True or plan.get("launch_authorized") is not True:
        raise RuntimeError("plan is unarmed")
    if Path(plan["run_dir"]) != ATTEMPT / "run":
        raise RuntimeError("plan names another run")
    if (ATTEMPT / "run").exists() or (ATTEMPT / "controller-started.json").exists():
        raise RuntimeError("attempt already started")
    argv = [
        sys.executable,
        "-u",
        str(HERE / "controller.py"),
        "--plan",
        str(plan_path),
        "--plan-sha256",
        args.plan_sha256,
    ]
    write("wrapper-started.json", dict(pid=os.getpid(), argv=argv, started_utc=now(), run_nonce=plan["run_nonce"]))
    with (ATTEMPT / "controller.log").open("xb") as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        write("controller-started.json", dict(pid=child.pid, argv=argv, started_utc=now(), run_nonce=plan["run_nonce"]))
        rc = child.wait()
    write("controller-exit.json", dict(actual_exit=rc, pid=child.pid, ended_utc=now(), run_nonce=plan["run_nonce"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
