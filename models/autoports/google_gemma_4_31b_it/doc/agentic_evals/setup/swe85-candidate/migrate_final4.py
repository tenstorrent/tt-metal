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


check = """set -eu; stat -c '%a %u %g %n' /root /root/.local /root/.cache /root/.cache/uv /root/.cache/uv/environments-v2 /root/.cache/uv/environments-v2/parser-466021493977ef71 /tmp; git -C /testbed rev-parse HEAD; /root/.local/bin/uv --version; /root/.local/bin/uv tool list; test "$LITELLM_LOCAL_MODEL_COST_MAP" = True; test ! -e /tests; test ! -e /solution; /root/.cache/uv/environments-v2/parser-466021493977ef71/bin/python -c 'import importlib.metadata as m; print({p:m.version(p) for p in ["swebench","datasets","fastcore"]})'; find /root/.local /root/.cache/uv/environments-v2/parser-466021493977ef71 -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum"""

manifest = o / "migration-final4.json"
assert not manifest.exists()
rows = []


def save():
    manifest.write_text(
        json.dumps(
            {
                "rows": rows,
                "status": "running",
                "scope": "Verified equivalent targeted migrations; original reports immutable",
            },
            indent=2,
        )
        + "\n"
    )


tasks = json.loads((s / "swe65-candidate/tasks.json").read_text())["tasks"]
selected = tasks[20:24]
for t in selected:
    task = t["task"]
    oldtag = "v5-swe-offline-verifier:" + task
    newtag = "v5-swe-offline-linked:" + task + "-mode-restored"
    old = inspect(oldtag)
    base = inspect(t["official_image"])
    before = disk()
    row = {
        "task": task,
        "old_inspect": old,
        "official_base": base["Id"],
        "official_repo_digests": base["RepoDigests"],
        "free_before": before,
        "status": "validating",
    }
    rows.append(row)
    save()
    try:
        assert old["RepoTags"] == [oldtag]
        assert not run(["docker", "ps", "-aq", "--filter", "ancestor=" + old["Id"]]).strip()
        mode = bash(t["official_image"], "stat -c %a /root").strip()
        row["base_root_mode"] = mode
        paths = "/root /root/.local /root/.cache /root/.cache/uv /root/.cache/uv/environments-v2 /tmp"
        bash(t["official_image"], "for p in " + paths + '; do test ! -L "$p" || exit 3; done')
        existing = subprocess.run(["docker", "image", "inspect", newtag], capture_output=True).returncode == 0
        if not existing:
            assert before - 1400000000 >= 170000000000
            cmd = [
                "docker",
                "build",
                "--network",
                "none",
                "--pull=false",
                "--build-arg",
                "OFFICIAL_IMAGE=" + t["official_image"],
                "--build-arg",
                "BASE_ROOT_MODE=" + mode,
                "-f",
                str(o / "Dockerfile.offline-linked"),
                "-t",
                newtag,
                str(root / "bringup/artifacts"),
            ]
            row["build_command"] = cmd
            save()
            with (o / (task + ".migration-build.log")).open("w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=300)
        new = inspect(newtag)
        row["new_inspect"] = new
        assert new["RootFS"]["Layers"][: len(base["RootFS"]["Layers"])] == base["RootFS"]["Layers"]
        a = bash(oldtag, check)
        b = bash(newtag, check)
        assert a == b
        row["validation"] = b
        row["status"] = "equivalent"
        save()
        row["commands"] = []
        for cmd in [["docker", "tag", new["Id"], oldtag], ["docker", "image", "rm", old["Id"]]]:
            r = subprocess.run(cmd, text=True, capture_output=True)
            row["commands"].append({"argv": cmd, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr})
            save()
            assert r.returncode == 0
        row.update(status="migrated", free_after=disk())
        row["actual_gain_bytes"] = row["free_after"] - before
        save()
        print(task, row["actual_gain_bytes"], row["free_after"], flush=True)
    except Exception as e:
        row["status"] = "stopped_error"
        row["error"] = repr(e)
        save()
        break
coverage = {t["task"]: inspect("v5-swe-offline-verifier:" + t["task"])["Id"] for t in tasks}
manifest.write_text(
    json.dumps(
        {"rows": rows, "status": "finished", "coverage65_current": coverage, "final_free_bytes": disk()}, indent=2
    )
    + "\n"
)
