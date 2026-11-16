import datetime
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time

root = pathlib.Path("/home/mvasiljevic/tt-metal-gemma4-31b-pipeline")
s = root / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup"
o = s / "swe85-candidate"


def run(a):
    return subprocess.check_output(a, text=True, timeout=120)


def inspect(t):
    return json.loads(run(["docker", "image", "inspect", t]))[0]


def bash(t, c):
    return run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/bash", t, "-c", c])


def disk():
    return int(
        run(
            [
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
                'import os;s=os.statvfs("/host-docker");print(s.f_bavail*s.f_frsize)',
            ]
        )
    )


check = """set -eu; stat -c '%a %u %g %n' /root /root/.local /root/.cache /root/.cache/uv /root/.cache/uv/environments-v2 /root/.cache/uv/environments-v2/parser-466021493977ef71 /tmp; git -C /testbed rev-parse HEAD; /root/.local/bin/uv --version; /root/.local/bin/uv tool list; test "$LITELLM_LOCAL_MODEL_COST_MAP" = True; test ! -e /tests; test ! -e /solution; /root/.cache/uv/environments-v2/parser-466021493977ef71/bin/python -c 'import importlib.metadata as m; print({p:m.version(p) for p in ["swebench","datasets","fastcore"]})'; find /root/.local /root/.cache/uv/environments-v2/parser-466021493977ef71 -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum"""
manifest = o / "linked-remaining3-preparation.json"
assert not manifest.exists(), "No automatic retries"
alltasks = json.loads((o / "tasks.json").read_text())["tasks"]
done = {
    r["task"] for r in json.loads((o / "linked-next20-preparation.json").read_text())["rows"] if r["status"] == "ready"
}
done.update(
    r["task"]
    for r in json.loads((o / "linked-remaining5-preparation.json").read_text())["rows"]
    if r["status"] == "ready"
)
tasks = [t for t in alltasks if t["task"] not in done]
assert len(tasks) == 3
oldtasks = json.loads((s / "swe65-candidate/tasks.json").read_text())["tasks"]
pullcost = []
for name in ["pull_manifest.json", "pull_retry_20261114T0958Z.json"]:
    for r in json.loads((s / "swe65-candidate" / name).read_text())["rows"]:
        if r.get("status") == "pulled":
            pullcost.append(
                (
                    r["task"].split("__")[0],
                    max(0, r["disk_before"]["available_bytes"] - r["disk_after"]["available_bytes"]),
                )
            )
state = {
    "status": "running",
    "rows": [],
    "source_sha256": hashlib.sha256((o / "tasks.json").read_bytes()).hexdigest(),
    "dockerfile_sha256": hashlib.sha256((o / "Dockerfile.offline-linked").read_bytes()).hexdigest(),
    "floor_bytes": 170000000000,
    "stopfile": str(o / "STOP_PREPARATION"),
    "prediction_rule": "Per-repository max observed official pull growth (fallback global max), plus linked build max(1.3GB, observed build growth*1.10). No prediction guarantees unknown image size; inflight guard170.5GB.",
    "protected65": {t["task"]: inspect("v5-swe-offline-verifier:" + t["task"])["Id"] for t in oldtasks},
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}


def save():
    manifest.write_text(json.dumps(state, indent=2) + "\n")


def bounded(cmd, log, timeout):
    with log.open("w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        start = time.monotonic()
        while p.poll() is None:
            time.sleep(5)
            if disk() < 170500000000 or time.monotonic() - start > timeout:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=15)
                raise RuntimeError("inflight disk/time guard")
        assert p.returncode == 0, ("command failed", p.returncode, str(log))


reference = bash("v5-swe-offline-linked:pallets__flask-5014-mode-restored", check).splitlines()[8:]
shared = json.loads((o / "linked-pilot.json").read_text())["assessment"]["shared_payload_digests"]
build_est = 1300000000
save()
try:
    for t in tasks:
        if (o / "STOP_PREPARATION").exists():
            state["status"] = "stopped_stopfile"
            break
        task = t["task"]
        cost = max([v for k, v in pullcost if k == task.split("__")[0]] or [v for k, v in pullcost])
        before = disk()
        row = {
            "task": task,
            "status": "pending",
            "free_before": before,
            "predicted_pull_bytes": cost,
            "predicted_build_bytes": build_est,
        }
        state["rows"].append(row)
        save()
        if before - cost - build_est < 170000000000:
            state["status"] = "stopped_projected_disk_floor"
            save()
            break
        base = "docker.io/" + t["official_image"]
        new = "v5-swe-offline-linked:" + task + "-mode-restored"
        cmd = ["docker", "pull", base]
        row["pull_command"] = cmd
        row["status"] = "pulling"
        save()
        bounded(cmd, o / (task + ".pull.log"), 600)
        orig = inspect(base)
        afterpull = disk()
        row.update(
            status="pulled",
            base_image_id=orig["Id"],
            repo_digests=orig["RepoDigests"],
            base_size_bytes=orig["Size"],
            free_after_pull=afterpull,
        )
        save()
        if afterpull - build_est < 170000000000:
            state["status"] = "stopped_projected_build_floor"
            break
        paths = "/root /root/.local /root/.cache /root/.cache/uv /root/.cache/uv/environments-v2 /tmp"
        row["base_ancestor_modes"] = bash(
            base,
            "for p in "
            + paths
            + '; do test ! -L "$p" || exit 3; if test -e "$p"; then stat -c "%a %u %g %n" "$p"; fi; done',
        )
        mode = bash(base, "stat -c %a /root").strip()
        row["base_root_mode"] = mode
        assert mode.isdigit()
        cmd = [
            "docker",
            "build",
            "--network",
            "none",
            "--pull=false",
            "--build-arg",
            "OFFICIAL_IMAGE=" + base,
            "--build-arg",
            "BASE_ROOT_MODE=" + mode,
            "-f",
            str(o / "Dockerfile.offline-linked"),
            "-t",
            new,
            str(root / "bringup/artifacts"),
        ]
        row["build_command"] = cmd
        row["status"] = "building"
        save()
        log = o / (task + ".linked-build.log")
        bounded(cmd, log, 300)
        assert "PASSED" in log.read_text()
        image = inspect(new)
        layers = image["RootFS"]["Layers"]
        assert layers[: len(orig["RootFS"]["Layers"])] == orig["RootFS"]["Layers"]
        added = layers[len(orig["RootFS"]["Layers"]) :]
        assert added[1:3] == shared
        validation = bash(new, check)
        lines = validation.splitlines()
        assert lines[8:] == reference
        assert lines[7] == bash(base, "git -C /testbed rev-parse HEAD").strip()
        for old in row["base_ancestor_modes"].splitlines():
            assert old in lines[:7], ("ancestor permission mismatch", old, lines[:7])
        after = disk()
        growth = afterpull - after
        build_est = max(build_est, int(max(0, growth) * 1.1))
        row.update(
            status="ready",
            derived_image_id=image["Id"],
            derived_tag=new,
            added_layers=added,
            validation=validation,
            free_after=after,
            actual_pull_growth=before - afterpull,
            actual_build_growth=growth,
        )
        save()
        print(
            json.dumps(
                {"task": task, "ready": sum(r["status"] == "ready" for r in state["rows"]), "free_GB": after / 1e9}
            ),
            flush=True,
        )
    else:
        state["status"] = "completed"
except Exception as e:
    state["status"] = "stopped_error"
    state["error"] = repr(e)
state["final_free_bytes"] = disk()
assert state["protected65"] == {t["task"]: inspect("v5-swe-offline-verifier:" + t["task"])["Id"] for t in oldtasks}
state["protected65_unchanged"] = True
save()
print(state["status"], state.get("error"), state["final_free_bytes"], flush=True)
