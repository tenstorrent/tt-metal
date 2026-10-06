# SPDX-License-Identifier: Apache-2.0
"""Collect final decoder and isolated reader profiles serially, without watcher."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ["MC_ARTIFACT_DIR"])
    assert not os.environ.get("TT_METAL_WATCHER")
    prefix = "models.autoports.aleph_alpha_kolibri_1_bf16.tests."
    source_sha = hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest()
    reader = out / "reader_profile"
    reader.mkdir(exist_ok=True)
    jobs = [
        ([sys.executable, "-m", prefix + "multichip_profiles"], out / "profiles.log", {}),
        (
            [
                sys.executable,
                "-m",
                "tracy",
                "-r",
                "-p",
                "--no-web-server",
                "-o",
                str(reader / "raw"),
                "-m",
                prefix + "optimized_multichip_geometry",
                "--readers-only",
                "--repetitions",
                "1",
            ],
            reader / "capture.log",
            {"MC_ARTIFACT_DIR": str(reader)},
        ),
        (
            [
                sys.executable,
                "-m",
                prefix + "optimized_multichip_reader_report",
                str(reader),
                str(reader / "dense_geometry_0_readers.json"),
            ],
            reader / "analysis.log",
            {},
        ),
        ([sys.executable, "-m", prefix + "multichip_profile_analysis"], out / "profile_analysis.log", {}),
        ([sys.executable, "-m", prefix + "optimized_multichip_accounting"], out / "accounting.log", {}),
    ]
    for command, log, changes in jobs:
        env = os.environ | {"TT_METAL_TRACE_ALLOC_TRACKING": "0"} | changes
        with log.open("w") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        with (out / "final_profile_commands.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        command=command,
                        environment=changes | {"TT_METAL_TRACE_ALLOC_TRACKING": "0"},
                        source_sha256=source_sha,
                        exit_code=result.returncode,
                        log=str(log),
                        time=time.time(),
                    )
                )
                + "\n"
            )
        print(log, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
