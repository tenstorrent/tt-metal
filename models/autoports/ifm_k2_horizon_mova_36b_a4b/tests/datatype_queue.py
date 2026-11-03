# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One hardware owner at a time; stop on a runtime failure for diagnosis."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    rejected = {r["config_id"] for r in json.loads((DOC / "capacity_rejections.json").read_text())}
    for name in args.only or json.loads((DOC / "candidate_matrix.json").read_text()):
        if name in rejected:
            continue
        dest = DOC / ("smokes" if args.smoke else "runs") / f"{name}.json"
        if dest.exists() and json.loads(dest.read_text())["status"] in ("pass", "accuracy_fail", "smoke_pass"):
            continue
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.datatype_candidate",
            "--config",
            str(DOC / "configs" / f"{name}.json"),
            "--output",
            str(dest),
        ]
        if args.smoke:
            cmd.append("--smoke")
        env = {
            **os.environ,
            "TT_METAL_TRACE_ALLOC_TRACKING": "1" if args.smoke else "0",
            "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        }
        log = DOC / "logs" / f'{"smoke_" if args.smoke else ""}{name}.log'
        if dest.exists():
            previous = dest.read_bytes()
            archive = DOC / "debug" / "failures"
            archive.mkdir(parents=True, exist_ok=True)
            stamp = name + "." + hashlib.sha256(previous).hexdigest()[:12]
            (archive / (stamp + ".json")).write_bytes(previous)
            if log.exists():
                (archive / (stamp + ".log")).write_bytes(log.read_bytes())
        print("RUN", name, flush=True)
        with log.open("w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        print("EXIT", name, rc, flush=True)
        if rc:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
