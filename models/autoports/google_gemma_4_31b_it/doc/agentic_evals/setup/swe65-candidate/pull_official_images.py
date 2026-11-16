"""Bounded serial authorized DockerHub setup; no builds, evals, or cleanup."""
import datetime
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time

base = pathlib.Path(__file__).resolve().parent
stop = base / "STOP_PULLS"
manifest = base / "pull_manifest.json"
logs = base / "pull-logs"
logs.mkdir(exist_ok=True)
if manifest.exists():
    raise SystemExit("Manifest exists; never retry automatically")
source = base / "tasks.json"
data = json.loads(source.read_text())
rows = [
    {"task": r["task"], "image": "docker.io/" + r["official_image"], "status": "pending"}
    for r in data["tasks"]
    if not r["image_cached"]
]
assert len(rows) == 55
for row in rows:
    assert row["image"].startswith("docker.io/swebench/sweb.eval.x86_64.") and row["image"].endswith(":latest")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def disk():
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "/root/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12",
        "--mount",
        "type=bind,source=/var/lib/docker,target=/host-docker,readonly",
        "v5-swe-offline-verifier:pylint-dev__pylint-6903",
        "-c",
        'import os,json;s=os.statvfs("/host-docker");print(json.dumps({"available_bytes":s.f_bavail*s.f_frsize,"total_bytes":s.f_blocks*s.f_frsize}))',
    ]
    return json.loads(subprocess.check_output(cmd, text=True, timeout=30))


start = time.monotonic()
state = {
    "started_utc": now(),
    "status": "running",
    "stop_file": str(stop),
    "candidate_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "rows": rows,
    "limits": {
        "overall_seconds": 7200,
        "per_image_seconds": 600,
        "minimum_free_bytes": 170_000_000_000,
        "start_headroom_bytes": 32_000_000_000,
        "inflight_cancel_free_bytes": 180_000_000_000,
    },
    "disk_method": "Read-only host /var/lib/docker bind, os.statvfs f_bavail; existing local image network none.",
    "contention_note": "Concurrent CPU HF Astropy controls; Docker pulls/decompression may contend for CPU, RAM and disk. Setup timestamps are not model diagnostic performance evidence.",
    "authorization_scope": "Official DockerHub pulls only; no builds/tests/evals/GitHub/cleanup.",
}


def save():
    temp = manifest.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(manifest)


def cancel(proc):
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


save()
for row in rows:
    if stop.exists():
        state["status"] = "stopped_stop_file"
        break
    if time.monotonic() - start >= 7200:
        state["status"] = "stopped_overall_deadline"
        break
    before = disk()
    row["disk_before"] = before
    if before["available_bytes"] < 202_000_000_000:
        state["status"] = "stopped_disk_headroom"
        break
    row.update(status="pulling", started_utc=now())
    log = logs / (row["task"] + ".log")
    row["log"] = str(log)
    save()
    t = time.monotonic()
    reason = None
    with log.open("w") as f:
        proc = subprocess.Popen(
            ["docker", "pull", row["image"]], stdout=f, stderr=subprocess.STDOUT, start_new_session=True
        )
        while proc.poll() is None:
            time.sleep(5)
            if time.monotonic() - t >= 600:
                reason = "per_image_timeout"
            if time.monotonic() - start >= 7200:
                reason = "overall_deadline"
            current = disk()
            row["latest_disk"] = current
            if current["available_bytes"] < 180_000_000_000:
                reason = "disk_reserve_guard"
            if reason:
                cancel(proc)
                break
        code = proc.wait()
    row.update(finished_utc=now(), exit_code=code, disk_after=disk(), elapsed_setup_seconds=time.monotonic() - t)
    if reason or code != 0:
        row["status"] = reason or "pull_failed"
        state["status"] = "stopped_" + row["status"]
        save()
        print(json.dumps({"task": row["task"], "status": row["status"]}), flush=True)
        break
    info = json.loads(subprocess.check_output(["docker", "image", "inspect", row["image"]], text=True, timeout=30))[0]
    row.update(
        status="pulled",
        image_id=info["Id"],
        repo_digests=info["RepoDigests"],
        size_bytes=info["Size"],
        rootfs_layers=info["RootFS"]["Layers"],
    )
    assert row["repo_digests"]
    save()
    print(
        json.dumps(
            {
                "task": row["task"],
                "status": "pulled",
                "completed": sum(r["status"] == "pulled" for r in rows),
                "free_GB": row["disk_after"]["available_bytes"] / 1e9,
            }
        ),
        flush=True,
    )
else:
    state["status"] = "completed"
state["finished_utc"] = now()
state["final_disk"] = disk()
save()
print(
    json.dumps(
        {
            "status": state["status"],
            "pulled": sum(r["status"] == "pulled" for r in rows),
            "final_free_GB": state["final_disk"]["available_bytes"] / 1e9,
        }
    ),
    flush=True,
)
