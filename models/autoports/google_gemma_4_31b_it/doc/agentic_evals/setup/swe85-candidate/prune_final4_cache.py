import datetime
import json
import pathlib
import re
import subprocess

S = pathlib.Path(__file__).resolve().parent
inv = json.loads((S / "reclamation-inventory.json").read_text())
report = {
    "status": "running",
    "authorization": "Root verified Docker primary prune docs id equality; installed du exact-filter proof required per record",
    "documentation": "https://docs.docker.com/reference/cli/docker/buildx/prune/",
    "rows": [],
}


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


def save():
    (S / "final4-cache-prune.json").write_text(json.dumps(report, indent=2) + "\n")


save()

source = json.loads((S / "final4-cache-inspection.json").read_text())


def ts(x):
    return datetime.datetime.fromisoformat(x[:26].replace(" ", "T")).timestamp()


selected = []
for image in source["removed_images"]:
    rows = [
        r
        for r in source["same_day_cache_records"]
        if 0 <= ts(image["created"]) - ts(r["Created at"]) < 6
        and r.get("Shared") == "false"
        and r.get("Reclaimable") == "true"
        and any(
            x in r.get("Description", "")
            for x in [
                "COPY v5-swe-runtime/root-local",
                "COPY swe-verifier-image-context",
                "mount / from exec /bin/sh -c mkdir -p /logs/verifier /tests",
            ]
        )
    ]
    rows.sort(key=lambda r: ts(r["Created at"]), reverse=True)
    for r in rows:
        r["matched_removed_image"] = image["id"]
    selected += rows
report["selection_evidence"] = selected
save()
for old in selected:
    ident = old["ID"]
    cmd = ["docker", "buildx", "du", "--verbose", "--filter", "id=" + ident]
    r = subprocess.run(cmd, text=True, capture_output=True)
    row = {
        "id": ident,
        "inspection_command": cmd,
        "before": r.stdout,
        "inspect_returncode": r.returncode,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    report["rows"].append(row)
    ids = re.findall(r"^ID:\s*(\S+)", r.stdout, re.M)
    if not ids:
        row["status"] = "already_absent"
        save()
        continue
    assert ids == [ident], ids
    fields = dict(re.findall(r"^([^:\n]+):\s*(.*)$", r.stdout, re.M))
    assert fields.get("Description") == old["Description"] and fields.get("Created at") == old["Created at"]
    if fields.get("Shared") != "false" or fields.get("Reclaimable") != "true":
        row["status"] = "skipped_state_changed"
        save()
        continue
    cmd = ["docker", "buildx", "prune", "--filter", "id=" + ident, "--force"]
    r = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
    row.update(
        prune_command=cmd,
        returncode=r.returncode,
        stdout=r.stdout,
        stderr=r.stderr,
        status="prune_completed" if r.returncode == 0 else "error",
    )
    save()
    if r.returncode:
        break
report["protected_after"] = snapshot()
assert report["protected_after"] == report["protected_before"]
report["free_after"] = free()
report["actual_free_delta_bytes"] = report["free_after"] - report["free_before"]
report["headroom_over170GB"] = report["free_after"] - 170000000000
report["status"] = "completed"
save()
print(
    json.dumps(
        {k: report[k] for k in ["status", "free_before", "free_after", "actual_free_delta_bytes", "headroom_over170GB"]}
    )
)
