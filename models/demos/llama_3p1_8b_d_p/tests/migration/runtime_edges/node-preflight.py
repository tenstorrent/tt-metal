# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import datetime
import hashlib
import json
import os
import re
import shutil
import socket
import sys
from pathlib import Path

THREAD_LIMITS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def validate_cpu_contract(environ, allowed_cpus):
    allowed = sorted(allowed_cpus)
    if not allowed:
        raise ValueError("Linux CPU affinity mask must be nonempty")
    if environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("SLURM_CPUS_PER_TASK must be 1")
    wrong = {name: environ.get(name) for name in THREAD_LIMITS if environ.get(name) != "1"}
    if wrong:
        raise ValueError(f"host thread limits must all be 1, got {wrong}")
    return allowed


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


def main():
    evidence = Path(sys.argv[1])
    mode = sys.argv[2]
    assert mode in ("collection", "device")
    assert socket.gethostname() == os.environ["WRITER_EXPECTED_NODE"]
    assert os.environ["SLURM_JOB_ID"] == os.environ["WRITER_EXPECTED_JOB"]
    allowed = validate_cpu_contract(os.environ, os.sched_getaffinity(0))
    prefix = "collection-" if mode == "collection" else ""
    pins = json.loads(Path(os.environ["WRITER_SOURCE_PINS"]).read_text())
    observed = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in pins}
    dump(evidence / f"{prefix}node-source-pins.json", observed)
    if observed != pins:
        raise RuntimeError("source changed before node payload")

    ancestors, walk = ancestor_walk(Path("/proc"), os.getpid())
    dump(evidence / f"{prefix}ancestor-walk-before-native.json", walk)
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if re.search("pytest|tt-smi|tt-triage|tt-run|prefill|llama|_ttnn|kv_manager", command, re.I):
                found.append({"pid": int(entry.name), "command": command})
            for descriptor in (entry / "fd").iterdir():
                try:
                    target = os.readlink(descriptor)
                except (FileNotFoundError, PermissionError):
                    continue
                if target.startswith("/dev/tenstorrent"):
                    found.append({"pid": int(entry.name), "device_fd": target})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    dump(evidence / f"{prefix}scoped-processes-before-native.json", found)
    if found:
        raise RuntimeError("competing or retained device process detected")

    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            memory[key] = int(value.strip().split()[0]) * 1024
    disk = shutil.disk_usage(evidence.parent)
    if disk.free < 10 * 1024**3:
        raise RuntimeError(f"less than 10 GiB free for snapshots and task-local Metal cache: {disk.free}")
    dump(
        evidence / f"{prefix}host-resource.json",
        {
            "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "node": socket.gethostname(),
            "job_id": os.environ["SLURM_JOB_ID"],
            "slurm_cpus_per_task": os.environ["SLURM_CPUS_PER_TASK"],
            "thread_limits": {name: os.environ[name] for name in THREAD_LIMITS},
            "observed_linux_cpu_affinity": allowed,
            "cpu_exclusivity_claimed": False,
            "memory_bytes": memory,
            "disk_bytes": {"total": disk.total, "used": disk.used, "free": disk.free},
            "device_opened": False,
            "mode": mode,
        },
    )


if __name__ == "__main__":
    main()
