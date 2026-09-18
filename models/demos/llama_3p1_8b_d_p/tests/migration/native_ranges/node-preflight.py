import datetime as dt
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def process_stat(proc_root, pid):
    text = (proc_root / str(pid) / "stat").read_text()
    fields = text[text.rfind(")") + 2 :].split()
    return (int(fields[1]), int(fields[19]))


def ancestor_walk(proc_root, start_pid):
    trusted = set()
    records = []
    pid = start_pid
    child = None
    boundary = None
    while pid > 1 and pid not in trusted:
        try:
            (parent, started) = process_stat(proc_root, pid)
            if child is not None:
                (child_pid, child_parent, child_started) = child
                if process_stat(proc_root, child_pid) != (child_parent, child_started) or started > child_started:
                    boundary = {"pid": pid, "reason": "parent_link_or_identity_changed"}
                    break
        except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
            boundary = {"pid": pid, "reason": type(error).__name__, "detail": str(error)}
            break
        trusted.add(pid)
        records.append({"pid": pid, "parent": parent, "start_time_ticks": started})
        child = (pid, parent, started)
        pid = parent
    if boundary is None:
        boundary = {"pid": pid, "reason": "root_boundary" if pid <= 1 else "cycle"}
    return (
        trusted,
        {
            "trusted_pids": sorted(trusted),
            "records": records,
            "truncated": boundary["reason"] != "root_boundary",
            "boundary": boundary,
        },
    )


def process_guard(attempt, proc_root=Path("/proc"), start_pid=None):
    (ancestors, walk) = ancestor_walk(proc_root, os.getpid() if start_pid is None else start_pid)
    dump(attempt / "ancestor-walk-before-native.json", walk)
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            if re.search("pytest|tt-smi|tt-triage|tt-run|prefill|llama|_ttnn", command, re.I):
                found.append({"pid": int(entry.name), "command": command})
            for fd in (entry / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except (FileNotFoundError, PermissionError):
                    continue
                if target.startswith("/dev/tenstorrent"):
                    found.append({"pid": int(entry.name), "device_fd": target})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    dump(attempt / "scoped-processes-before-native.json", found)
    if found:
        raise RuntimeError("Competing workload/device process detected; no native probe or test launched")


plan_path = Path(sys.argv[1])
digest = sys.argv[2]
assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == digest
plan = json.loads(plan_path.read_bytes())
role = sys.argv[3]
assert role in ("source", "passive")
local = plan[role]
run = Path(plan["run_dir"]) / (role + "-node")
run.mkdir(exist_ok=False)
assert socket.gethostname() == local["host"] and os.environ.get("SLURM_JOB_ID") == str(local["job_id"])
assert os.environ.get("SLURM_CPUS_PER_TASK") == "1"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    assert os.environ.get(key) == "1", key
raw = subprocess.check_output(
    ["scontrol", "show", "job", str(local["job_id"]), "-o"], text=True, env=dict(os.environ, TZ="UTC")
)
fields = dict(x.split("=", 1) for x in raw.split() if "=" in x)
assert fields["JobState"] == "RUNNING" and fields["UserId"].startswith(plan["allocation_owner"] + "(")
assert fields["NodeList"] == plan["allocation_nodelist"]
assert local["host"] in plan["allocation_nodes"] and len(plan["allocation_nodes"]) == 2
end = dt.datetime.fromisoformat(fields["EndTime"]).replace(tzinfo=dt.timezone.utc)
assert end == dt.datetime.fromisoformat(local["lease_end_utc"])
required = sum(plan["supervisor"][k] for k in ("normal_seconds", "cancel_seconds", "recovery_reserve_seconds"))
assert (end - dt.datetime.now(dt.timezone.utc)).total_seconds() >= required
actual = {}
for name, expected in plan["pins"].items():
    h = hashlib.sha256()
    with open(name, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    actual[name] = h.hexdigest()
    assert actual[name] == expected, name
dump(run / "source-on-node.json", actual)
dump(
    run / "node-contract.json",
    {
        "job": str(local["job_id"]),
        "node": local["host"],
        "cpus_per_task": 1,
        "affinity": sorted(os.sched_getaffinity(0)),
        "lease_end_utc": end.isoformat(),
        "scheduler": raw,
    },
)
process_guard(run)
print("TRANSFER_NODE_PROCESS_SOURCE_LEASE_GUARDS_PASSED", flush=True)
