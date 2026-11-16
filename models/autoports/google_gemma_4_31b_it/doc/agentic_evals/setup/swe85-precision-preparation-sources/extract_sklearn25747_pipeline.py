import json
import pathlib
import re
import subprocess

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
O = D / "anomalies/swe85-below-band/fetch-controls"
for t, files in [("scikit-learn__scikit-learn-25747", ["sklearn/__init__.py", "sklearn/pipeline.py"])]:
    d = (D / "anomalies/swe85-below-band/correct-actions" / t) if t.endswith("7590") else O / t
    d.mkdir(exist_ok=True)
    tag = re.search(
        r'docker_image = "([^"]+)"',
        (D / "setup/swe85-candidate/tasks85-offline-verifier" / t / "task.toml").read_text(),
    )[1]
    im = json.loads(subprocess.check_output(["docker", "image", "inspect", tag]))[0]
    cid = subprocess.check_output(["docker", "create", im["Id"]]).decode().strip()
    rows = []
    try:
        paths = (
            ["/testbed/.git/refs/heads/master"]
            if t.endswith("7590")
            else [
                "/opt/miniconda3/envs/testbed/conda-meta/history",
                "/testbed/.git/HEAD",
                "/testbed/.git/refs/heads/main",
                "/testbed/.git/refs/heads/master",
            ]
            + ["/testbed/" + f for f in files]
        )
        for p in paths:
            dst = d / "base" / (p.removeprefix("/testbed/") if t.endswith("7590") else p.lstrip("/"))
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["docker", "cp", cid + ":" + p, str(dst)], capture_output=True, text=True)
            rows.append({"source": p, "returncode": r.returncode, "stderr": r.stderr})
    finally:
        subprocess.run(["docker", "rm", cid], check=True, stdout=subprocess.DEVNULL)
    (d / ("head-extraction.json" if t.endswith("7590") else "extraction.json")).write_text(
        json.dumps(
            {
                "image_tag": tag,
                "image_id": im["Id"],
                "stopped_container_only": True,
                "container_removed": True,
                "files": rows,
            },
            indent=2,
        )
    )
    print(t, flush=True)
