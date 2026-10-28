# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Writes goldens/kairos/MANIFEST.json from what the golden jobs wrote (test/MANIFEST.json, bench/cases.json).

Run after make_goldens.py and make_bench_goldens.py, in the reference venv:
    KAIROS_GOLDEN_JOBS='{"test_goldens": 1040, "bench_goldens": 1041}' python -I write_manifest.py
Also writes sha256 of every golden file, so a re-make can be compared bit for bit.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "reference"))

import load_reference as R  # noqa: E402
from make_bench_goldens import CASES as BENCH_CASES  # noqa: E402

ROOT = Path("/home/andy/bringup/goldens/kairos")
VENV = Path("/home/andy/bringup/venvs/kairos")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    test = json.loads((ROOT / "test" / "MANIFEST.json").read_text())
    bench = json.loads((ROOT / "bench" / "cases.json").read_text())
    manifest = {
        "pins": {
            "repo": f"foundation-model-research/Kairos@{R.REPO_COMMIT}",
            "hf": f"{R.HF_REPO}@{R.HF_REVISION}",
            "safetensors_sha256": R.SAFETENSORS_SHA256,
        },
        **R.manifest_fields(),
        "loaded_dtypes": test.get("loaded_dtypes"),
        "bias0": test["bias0"],
        "num_threads": 8,  # load_reference.load() default, used by both golden scripts
        "input_generator": "models/experimental/kairos_forecast/tests/golden/inputs.py",
        "bench_seeds": {k: v[0] for k, v in BENCH_CASES.items()},
        "jobs": json.loads(os.environ.get("KAIROS_GOLDEN_JOBS", "{}")),
        "pip_freeze": (VENV / "pip_freeze.txt").read_text().split("\n"),
        "test_cases": test["cases"],
        "bench_cases": bench["cases"],
        # D4 item 6: the two JSONs the golden jobs write are hashed too (19 files)
        "sha256": {
            f"{d}/{p.name}": sha256(p)
            for d in ("test", "bench")
            for p in sorted([*(ROOT / d).glob("*.pt"), ROOT / d / ("MANIFEST.json" if d == "test" else "cases.json")])
        },
    }
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", ROOT / "MANIFEST.json", len(manifest["sha256"]), "golden files")


if __name__ == "__main__":
    main()
