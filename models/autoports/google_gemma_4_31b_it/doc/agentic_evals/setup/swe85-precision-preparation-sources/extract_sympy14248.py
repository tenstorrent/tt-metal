import json
import pathlib
import re
import subprocess

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
P = D / "anomalies/swe85-below-band/correct-actions/sympy__sympy-14248"
P.mkdir(exist_ok=False)
P.chmod(0o700)
T = D / "setup/swe85-candidate/tasks85-offline-verifier/sympy__sympy-14248"
cfg = json.loads((T / "tests/config.json").read_text())
tag = re.search(r'docker_image = "([^"]+)"', (T / "task.toml").read_text()).group(1)
im = json.loads(subprocess.check_output(["docker", "image", "inspect", tag]))[0]
cid = subprocess.check_output(["docker", "create", im["Id"]]).decode().strip()
base = P / "base"
base.mkdir()
try:
    for rel in ["sympy", ".git/HEAD", ".git/refs/heads/main", ".git/refs/heads/master", ".git/packed-refs"]:
        dst = base / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["docker", "cp", cid + ":/testbed/" + rel, str(dst)], capture_output=True, text=True)
        print(rel, r.returncode)
finally:
    subprocess.run(["docker", "rm", cid], stdout=subprocess.DEVNULL)
(P / "reference.patch.private").write_text(cfg["patch"])
(P / "reference.tests.private.patch").write_text(cfg["test_patch"])
(P / "extraction.json").write_text(
    json.dumps(
        {
            "image_tag": tag,
            "image_id": im["Id"],
            "dataset_base_commit": cfg["base_commit"],
            "stopped_image_copy": True,
            "container_removed": True,
        },
        indent=2,
    )
    + "\n"
)
