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
task = "scikit-learn__scikit-learn-13135"
oldtag = "v5-swe-offline-verifier:" + task
newtag = "v5-swe-offline-linked:" + task + "-mode-restored"
old = inspect(oldtag)
new = inspect(newtag)
record = {
    "task": task,
    "old_inspect": old,
    "new_inspect": new,
    "source_validation": "linked-pilot.json",
    "original_recipe": "../swe65-candidate/Dockerfile.offline",
    "new_recipe": "Dockerfile.offline-linked",
    "status": "validating",
    "free_before": disk(),
}


def save():
    (o / "migration-sklearn13135.json").write_text(json.dumps(record, indent=2) + "\n")


save()
assert old["RepoTags"] == [oldtag], old["RepoTags"]
refs = run(["docker", "ps", "-aq", "--filter", "ancestor=" + old["Id"]]).strip()
assert not refs, refs
before = bash(oldtag, check)
after = bash(newtag, check)
assert before == after
record["validation"] = after
base = inspect("swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-13135:latest")
assert new["RootFS"]["Layers"][: len(base["RootFS"]["Layers"])] == base["RootFS"]["Layers"]
record["official_base_id"] = base["Id"]
record["official_repo_digests"] = base["RepoDigests"]
record["status"] = "validated"
save()
commands = []
for cmd in [["docker", "tag", new["Id"], oldtag], ["docker", "image", "rm", old["Id"]]]:
    r = subprocess.run(cmd, text=True, capture_output=True)
    commands.append({"argv": cmd, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr})
    record["commands"] = commands
    save()
    assert r.returncode == 0
assert inspect(oldtag)["Id"] == new["Id"]
tasks = json.loads((s / "swe65-candidate/tasks.json").read_text())["tasks"]
record["coverage65"] = {t["task"]: inspect("v5-swe-offline-verifier:" + t["task"])["Id"] for t in tasks}
record["free_after"] = disk()
record["actual_free_delta_bytes"] = record["free_after"] - record["free_before"]
record["status"] = "completed"
save()
print(record["free_before"], record["free_after"], record["actual_free_delta_bytes"])
