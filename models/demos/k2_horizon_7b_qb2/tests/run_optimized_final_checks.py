"""Serialized final same-harness timings, candidate controls, pytest and profile."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parents[1] / "doc/optimized_decoder"
MODULE = "models.demos.k2_horizon_7b_qb2.tests."


def main():
    cases = [
        (
            "pytest",
            [
                "-m",
                "pytest",
                str(Path(__file__).with_name("test_optimized_decoder.py")),
                "-q",
                "--junitxml=" + str(DOC / "pytest.xml"),
            ],
        )
    ]
    for seq in [4096, 128, 4097, 32768]:
        flags = ["--prefill-trace"] if seq == 4096 else []
        cases.append(
            (
                "final_timing_" + str(seq),
                [
                    "-m",
                    MODULE + "sweep_optimized",
                    "--candidates",
                    "fused",
                    "default",
                    "--seq",
                    str(seq),
                    *flags,
                    "--output",
                    str(DOC / f"final_timing_{seq}.json"),
                ],
            )
        )
    cases.append(
        (
            "qkv_comparison",
            [
                "-m",
                MODULE + "sweep_optimized",
                "--candidates",
                "default",
                "separateqkv_16_8_1",
                "separateqkv_16_16_2",
                "separateqkv_32_4_1",
                "separateqkv_32_8_2",
                "separateqkv_32_16_2",
                "separateqkv_64_8_2",
                "--output",
                str(DOC / "qkv_comparison.json"),
            ],
        )
    )
    cases.append(("profile_final", ["-m", MODULE + "profile_optimized", "--label", "final"]))
    env = os.environ.copy()
    for key in ["TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"]:
        env.pop(key, None)
    records = []
    for name, args in cases:
        command = [sys.executable, *args]
        log = DOC / "logs" / (name + ".log")
        print("START", name, flush=True)
        with log.open("w") as f:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        records.append(dict(name=name, argv=command, exit_code=proc.returncode, log=str(log)))
        (DOC / "final_checks.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END", name, proc.returncode, flush=True)
        if proc.returncode:
            print(log.read_text()[-5000:], flush=True)
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
