import json
import pathlib
import subprocess

O = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals/anomalies/swe85-below-band/correct-actions")
for task, paths in {
    "psf__requests-1142": ["build/lib/requests/models.py"],
    "scikit-learn__scikit-learn-10908": [
        "sklearn/feature_extraction/dict_vectorizer.py",
        "sklearn/grid_search.py",
        "sklearn/model_selection/_search.py",
    ],
}.items():
    d = O / task
    meta = json.loads((d / "extraction.json").read_text())
    cid = (
        subprocess.check_output(["docker", "create", "--name", "v5-diagnostic-gap-" + task, meta["image_id"]])
        .decode()
        .strip()
    )
    try:
        for rel in paths:
            dest = d / "base" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.run(["docker", "cp", cid + ":/testbed/" + rel, str(dest)], capture_output=True, text=True)
            meta["files"].append({"path": rel, "returncode": p.returncode, "stderr": p.stderr})
    finally:
        subprocess.run(["docker", "rm", cid], check=True, stdout=subprocess.DEVNULL)
    (d / "extraction.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(task, [(x["path"], x["returncode"]) for x in meta["files"][-len(paths) :]])
