"""Closed runtime-only plan contract; scheduler/process guards stay with root's launcher."""
import datetime
import fcntl
import json
import os
import re
import shutil
import socket
from pathlib import Path

from support import sha256


def check_plan(plan, environ, hostname):
    if plan.get("reviewed") is not True or plan.get("dispatch_enabled") is not True:
        raise ValueError("CLOSED: root must review and bind this plan before device use")
    if plan.get("scope") != "2k_runtime_edges_without_manager":
        raise ValueError("wrong scope")
    match = re.fullmatch(r"bh-glx-(\d+)-(\w+)", str(plan.get("node", "")))
    if match is None or not re.fullmatch(r"[0-9]+", str(plan.get("job_id", ""))):
        raise ValueError("wrong assigned endpoint")
    lock = Path(str(plan.get("lock_path", "")))
    if not lock.is_absolute() or lock.name != "prefill-device-" + match[1] + "-" + match[2] + ".lock":
        raise ValueError("wrong physical node lock")
    if hostname != plan["node"] or environ.get("SLURM_JOB_ID") != plan["job_id"]:
        raise ValueError("actual node/job differs")
    if environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ValueError("one CPU required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if environ.get(name) != "1":
            raise ValueError("all host thread limits must be1")
    if environ.get("PREFILL_FABRIC_MODE") != "1d_ring" or environ.get("TT_METAL_SLOW_DISPATCH_MODE"):
        raise ValueError("fast-dispatch1D ring environment required")
    if not re.fullmatch("[0-9a-f]{32}", plan.get("run_nonce", "")):
        raise ValueError("fresh nonce required")


def check_health(health, plan, now):
    if any(health.get(key) != plan[key] for key in ("node", "job_id")):
        raise ValueError("health endpoint differs")
    if any(type(health.get(k)) is not int or health[k] != 0 for k in ("actual_exit", "verified_exit")):
        raise ValueError("health not accepted")
    if not health.get("clean_close"):
        raise ValueError("health has no clean close")
    ended = datetime.datetime.fromisoformat(health["ended_utc"])
    if not 0 <= (now - ended).total_seconds() <= 1500:
        raise ValueError("health receipt stale")


def preflight(plan_path, expected_hash):
    if sha256(plan_path) != expected_hash:
        raise ValueError("plan hash changed")
    plan = json.loads(Path(plan_path).read_bytes())
    check_plan(plan, os.environ, socket.gethostname())
    # The root launcher retains this exact physical lock and passes its open descriptor.
    fd = int(os.environ["EDGE_NODE_LOCK_FD"])
    stat = os.fstat(fd)
    target = Path(plan["lock_path"]).stat()
    if (stat.st_dev, stat.st_ino) != (target.st_dev, target.st_ino):
        raise ValueError("wrong inherited physical lock")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launch = json.loads(Path(plan["launch_receipt"]).read_bytes())
    if sha256(plan["launch_receipt"]) != plan["launch_receipt_sha256"]:
        raise ValueError("launcher receipt changed")
    if any(launch.get(k) != plan[k] for k in ("node", "job_id", "run_nonce")):
        raise ValueError("launcher identity differs")
    if not all(
        launch.get(k) is True
        for k in ("scheduler_owner_verified", "empty_steps_verified", "no_competing_device_fds", "lock_held")
    ):
        raise ValueError("launcher ownership checks missing")
    now = datetime.datetime.now(datetime.timezone.utc)
    checked = datetime.datetime.fromisoformat(launch["checked_utc"])
    end = datetime.datetime.fromisoformat(launch["lease_end_utc"])
    if not 0 <= (now - checked).total_seconds() <= 120 or (end - now).total_seconds() < 2700:
        raise ValueError("stale launch receipt or insufficient cleanup lease")
    if sha256(plan["health_receipt"]) != plan["health_receipt_sha256"]:
        raise ValueError("root health receipt changed")
    check_health(json.loads(Path(plan["health_receipt"]).read_bytes()), plan, now)
    for path, digest in plan["source_pins"].items():
        if sha256(path) != digest:
            raise ValueError("source changed: " + path)
    spec = json.loads(Path(plan["accepted_gate_spec"]).read_bytes())
    for name, digest in spec["checkpoint_metadata"].items():
        if sha256(Path(plan["checkpoint"]) / name) != digest:
            raise ValueError("checkpoint identity differs")
    disk = shutil.disk_usage(Path(plan["output"]).parent).free
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(fields["MemAvailable"].split()[0]) * 1024
    if disk < 4 * 1024**3 or available < 64 * 1024**3:
        raise ValueError("insufficient snapshot disk or model-load memory headroom")
    return plan, spec, dict(disk_free_bytes=disk, mem_available_bytes=available, checked_utc=now.isoformat())
