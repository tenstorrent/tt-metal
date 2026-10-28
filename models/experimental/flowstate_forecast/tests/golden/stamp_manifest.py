# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Put a repo session stamp and the sha256 of every golden in MANIFEST.json (D4 R1-0d).

    python models/experimental/flowstate_forecast/tests/golden/stamp_manifest.py --jobs 957 959 971

``stamp`` is this job's session stamp (check_gates.session_stamp: git_head, git_dirty, started_at), and
``made_by`` each golden job with its start time and the branch commit that was HEAD when it started.
test_dataflow.test_goldens_match_manifest then checks the goldens against ``golden_sha256``."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS
from models.experimental.flowstate_forecast.tests.perf.check_gates import session_stamp

MANIFEST = Path(__file__).resolve().parent / "MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, nargs="+", required=True, help="the ttq jobs that wrote the goldens")
    args = ap.parse_args()
    stamp = session_stamp()
    made_by = []
    for j in args.jobs:
        job = json.loads(subprocess.run(["ttq", "show", str(j)], capture_output=True, text=True, check=True).stdout)
        head = subprocess.run(
            ["git", "rev-list", "-1", f"--before=@{int(job['started_at'])}", "HEAD"],
            cwd=MANIFEST.parent, capture_output=True, text=True, check=True,
        ).stdout.strip()
        made_by.append({"job": j, "argv": job["argv"][-1], "started_at": job["started_at"], "git_head": head})
    m = json.loads(MANIFEST.read_text())
    m["golden_sha256"] = {p.stem: sha256(p) for p in sorted(GOLDENS.glob("*.pt"))}
    m["stamp"] = {**stamp, "made_by": made_by}
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    print(json.dumps(m["stamp"], indent=1))


if __name__ == "__main__":
    main()
