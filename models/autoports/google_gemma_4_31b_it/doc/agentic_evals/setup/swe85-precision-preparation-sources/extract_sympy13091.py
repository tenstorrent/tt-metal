import json
import pathlib
import subprocess

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
O = D / "anomalies/swe85-below-band/correct-actions"
O.mkdir(parents=True, exist_ok=True)
O.chmod(0o700)
tasks = ["sympy__sympy-13091"]
for task in tasks:
    d = O / task
    d.mkdir(exist_ok=True)
    c = json.loads((D / "setup/swe85-candidate/tasks85-offline-verifier" / task / "tests/config.json").read_text())
    files = [s.split(" b/", 1)[1] for s in c["patch"].splitlines() if s.startswith("diff --git ")]
    files += ["sympy"]
    tag = (
        __import__("re")
        .search(
            r'docker_image = "([^"]+)"',
            (D / "setup/swe85-candidate/tasks85-offline-verifier" / task / "task.toml").read_text(),
        )
        .group(1)
    )
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", tag]))[0]
    cid = (
        subprocess.check_output(["docker", "create", "--name", "v5-diagnostic-source-" + task, image["Id"]])
        .decode()
        .strip()
    )
    results = []
    try:
        for rel in files + [".git/HEAD", ".git/refs/heads/main", ".git/refs/heads/master", ".git/packed-refs"]:
            dst = d / "base" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.run(["docker", "cp", cid + ":/testbed/" + rel, str(dst)], capture_output=True, text=True)
            results.append({"path": rel, "returncode": p.returncode, "stderr": p.stderr.strip()})
    finally:
        subprocess.run(["docker", "rm", cid], check=True, stdout=subprocess.DEVNULL)
    (d / "extraction.json").write_text(
        json.dumps(
            {
                "image_tag": tag,
                "image_id": image["Id"],
                "base_commit": c["base_commit"],
                "container_never_started": True,
                "container_removed": True,
                "files": results,
            },
            indent=2,
        )
        + "\n"
    )
    (d / "reference.patch.private").write_text(c["patch"])
    (d / "reference.tests.private.patch").write_text(c["test_patch"])
    (d / "test_names.private.json").write_text(json.dumps({k: c[k] for k in ["FAIL_TO_PASS", "PASS_TO_PASS"]}))
    print(task, "files", len(files), "HEAD", (d / "base/.git/HEAD").read_text().strip(), flush=True)
