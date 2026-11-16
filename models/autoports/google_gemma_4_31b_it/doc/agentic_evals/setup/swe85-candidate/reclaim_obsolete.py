import datetime
import hashlib
import json
import pathlib
import subprocess

S = pathlib.Path(__file__).resolve().parent
inv = json.loads((S / "reclamation-inventory.json").read_text())
E = S / "reclamation-private"
E.mkdir(exist_ok=True)
E.chmod(0o700)
report = {
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "commands": [],
    "containers": [],
    "images": [],
    "cache": "Not pruned: local CLI help exposes generic filters but does not document ID-scoped semantics; no broad prune authorized.",
}


def call(args):
    r = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    report["commands"].append({"argv": args, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr})
    return r


def free():
    r = subprocess.check_output(
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
        ],
        text=True,
    )
    return int(r)


def snapshot():
    tasks = json.loads((S.parent / "swe65-candidate/tasks.json").read_text())["tasks"]
    tags = ["v5-swe-offline-verifier:" + t["task"] for t in tasks] + [t["official_image"] for t in tasks]
    imgs = json.loads(subprocess.check_output(["docker", "image", "inspect", *tags]))
    return dict(zip(tags, [i["Id"] for i in imgs]))


report["protected_before"] = snapshot()
report["free_before"] = free()
# All generated Dockerfiles must use official bases, never obsolete preparation tags.
refs = []
for folder in ["swe65-candidate", "swe85-candidate", "fixed10-offline-verifier"]:
    for p in (S.parent / folder).rglob("*"):
        if p.is_file() and p.name.startswith("Dockerfile"):
            if any(
                x in p.read_text(errors="replace") for x in ["v5-swe-prepared:", "v5-swe-preflight:", "v5-swe-deps:"]
            ):
                refs.append(str(p))
assert not refs, refs
report["current_dockerfile_obsolete_references"] = refs
for c in inv["stopped_container_candidates"]:
    cid = c["id"]
    full = json.loads(subprocess.check_output(["docker", "inspect", cid]))[0]
    assert full["State"]["Status"] in ["exited", "created"]
    assert full["Config"]["Image"].startswith(("v5-swe-deps:", "v5-swe-preflight:"))
    assert not any("harbor" in k.lower() or "compose" in k.lower() for k in (full["Config"].get("Labels") or {}))
    files = []
    for kind, cmd in [
        ("inspect", ["docker", "inspect", cid]),
        ("logs", ["docker", "logs", cid]),
        ("diff", ["docker", "diff", cid]),
    ]:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p = E / (cid + "." + kind + ".private.txt")
        p.write_bytes(r.stdout)
        p.chmod(0o600)
        files.append({"path": str(p), "sha256": hashlib.sha256(r.stdout).hexdigest(), "returncode": r.returncode})
    # Full inspect and diffs retain exact recreation settings; original image/payload remains reproducible.
    files.append(
        {
            "recreate_provenance": "inspect retains Entrypoint, Cmd, Env, mounts, network; image ID is inventoried; runtime payload bringup/artifacts/v5-swe-runtime/root-local preserved"
        }
    )
    r = call(["docker", "rm", cid])
    report["containers"].append({"id": cid, "preserved": files, "removed": r.returncode == 0})
    assert r.returncode == 0
for i in inv["obsolete_image_candidates"]:
    for tag in i["tags"]:
        assert tag.startswith(("v5-swe-prepared:", "v5-swe-preflight:", "v5-swe-deps:"))
        now = json.loads(subprocess.check_output(["docker", "image", "inspect", tag]))[0]
        assert now["Id"] == i["id"]
        r = call(["docker", "image", "rm", tag])
        report["images"].append({"tag": tag, "id": i["id"], "removed": r.returncode == 0})
report["protected_after"] = snapshot()
assert report["protected_before"] == report["protected_after"]
report["free_after"] = free()
report["actual_free_delta_bytes"] = report["free_after"] - report["free_before"]
report["headroom_over170GB"] = report["free_after"] - 170000000000
report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(S / "reclamation-result.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps({k: report[k] for k in ["free_before", "free_after", "actual_free_delta_bytes", "headroom_over170GB"]})
)
