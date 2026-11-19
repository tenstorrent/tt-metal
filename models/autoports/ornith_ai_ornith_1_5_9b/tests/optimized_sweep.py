# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial, bounded candidate queue; STOP file is checked between device jobs."""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "doc/optimized_decoder"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    args = ap.parse_args()
    jobs = json.loads(Path(args.queue).read_text())
    for job in jobs:
        if (ART / "STOP").exists():
            break
        out = ART / (job["id"] + ".json")
        cmd = [
            "timeout",
            "1200",
            "python",
            "-m",
            "models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_candidate",
            "--layer",
            str(job.get("layer", 0)),
            "--lengths",
            job.get("lengths", "1024"),
            "--policy",
            json.dumps(job.get("policy", {})),
            "--benchmark",
            "--output",
            str(out),
            *job.get("flags", []),
        ]
        begin = time.time()
        with out.with_suffix(".log").open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        record = dict(job, command=cmd, returncode=result.returncode, start=begin, elapsed=time.time() - begin)
        with (ART / "queue_results.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(job["id"], result.returncode, round(record["elapsed"], 2), flush=True)
        if result.returncode in (124, 137):
            break


if __name__ == "__main__":
    main()
