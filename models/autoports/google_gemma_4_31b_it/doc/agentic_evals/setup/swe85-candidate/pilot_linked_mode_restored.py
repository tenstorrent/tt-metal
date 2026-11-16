import hashlib
import json
import pathlib
import subprocess

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


state = {
    "status": "running",
    "rows": [],
    "free_before": disk(),
    "dockerfile_sha256": hashlib.sha256((o / "Dockerfile.offline-linked").read_bytes()).hexdigest(),
    "scope": "Only two large COPY instructions add --link; network none, no frontend directive, no repo tests/model work.",
}


def save():
    (o / "linked-pilot.json").write_text(json.dumps(state, indent=2) + "\n")


tasks = json.loads((s / "swe65-candidate/tasks.json").read_text())["tasks"]
state["protected_images"] = {t["task"]: inspect("v5-swe-offline-verifier:" + t["task"])["Id"] for t in tasks}
save()
check = """set -eu; stat -c '%a %u %g %n' /root /root/.local /root/.cache /root/.cache/uv /root/.cache/uv/environments-v2 /root/.cache/uv/environments-v2/parser-466021493977ef71 /tmp; git -C /testbed rev-parse HEAD; /root/.local/bin/uv --version; /root/.local/bin/uv tool list; test "$LITELLM_LOCAL_MODEL_COST_MAP" = True; test ! -e /tests; test ! -e /solution; /root/.cache/uv/environments-v2/parser-466021493977ef71/bin/python -c 'import importlib.metadata as m; print({p:m.version(p) for p in ["swebench","datasets","fastcore"]})'; find /root/.local /root/.cache/uv/environments-v2/parser-466021493977ef71 -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum"""
try:
    for task in ["pallets__flask-5014", "psf__requests-1724"]:
        assert disk() >= 190000000000
        base = "swebench/sweb.eval.x86_64." + task.replace("__", "_1776_") + ":latest"
        old = "v5-swe-offline-verifier:" + task
        new = "v5-swe-offline-linked:" + task + "-mode-restored"
        mode = bash(base, "stat -c %a /root").strip()
        assert mode.isdigit()
        paths = bash(
            base,
            'set -eu; for p in /root /root/.local /root/.cache /root/.cache/uv /root/.cache/uv/environments-v2 /tmp; do test ! -L "$p"; printf "%s not_symlink\\n" "$p"; done',
        )
        row = {
            "task": task,
            "base": inspect(base)["Id"],
            "destination_checks": paths,
            "base_root_mode": mode,
            "free_before": disk(),
        }
        state["rows"].append(row)
        save()
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
        row["command"] = cmd
        with (o / (task + ".linked-build.log")).open("w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=300)
        assert "PASSED" in (o / (task + ".linked-build.log")).read_text()
        a = inspect(base)
        b = inspect(new)
        assert b["RootFS"]["Layers"][: len(a["RootFS"]["Layers"])] == a["RootFS"]["Layers"]
        before = bash(old, check)
        after = bash(new, check)
        assert before == after, (before, after)
        assert bash(base, "git -C /testbed rev-parse HEAD").strip() == after.splitlines()[7]
        row.update(
            status="validated",
            image_id=b["Id"],
            added_layers=b["RootFS"]["Layers"][len(a["RootFS"]["Layers"]) :],
            validation=after,
            free_after=disk(),
        )
        save()
        print(task, "validated", flush=True)
    state["shared_added_layers"] = list(set(state["rows"][0]["added_layers"]) & set(state["rows"][1]["added_layers"]))
    state["status"] = "completed"
except Exception as e:
    state["status"] = "stopped_error"
    state["error"] = repr(e)
state["free_after"] = disk()
state["actual_growth_bytes"] = state["free_before"] - state["free_after"]
assert state["protected_images"] == {t["task"]: inspect("v5-swe-offline-verifier:" + t["task"])["Id"] for t in tasks}
state["protected65_unchanged"] = True
save()
print(state["status"], state.get("error"), state["actual_growth_bytes"], flush=True)
