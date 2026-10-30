"""Serialize final-source acceptance and profiling without mixing diagnostics."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc/fused_decoder"
BASE = "models.autoports.google_gemma_4_31b_it.tests."


def main():
    jobs = []
    for layer in (0, 5):
        jobs.append(
            (
                f"final_smoke_{layer}",
                [
                    sys.executable,
                    "-m",
                    BASE + "run_fused_decoder",
                    "--layer",
                    str(layer),
                    "--seq",
                    "33",
                    "--result",
                    str(OUT / f"final_smoke_{layer}.json"),
                ],
                True,
            )
        )
        for batch in (1, 32):
            jobs.append(
                (
                    f"final_timing_{layer}_{batch}",
                    [
                        sys.executable,
                        "-m",
                        BASE + "time_fusion_candidate",
                        "--layer",
                        str(layer),
                        "--batch",
                        str(batch),
                        "--result",
                        str(OUT / f"final_timing_{layer}_{batch}.json"),
                    ],
                    False,
                )
            )
            jobs.append(
                (
                    f"equivalence_{layer}_{batch}",
                    [
                        sys.executable,
                        "-m",
                        BASE + "test_fused_equivalence",
                        "--layer",
                        str(layer),
                        "--batch",
                        str(batch),
                        "--seq",
                        "1025" if batch == 1 else "33",
                        "--result",
                        str(OUT / f"equivalence_{layer}_{batch}.json"),
                    ],
                    True,
                )
            )
    for layer in (0, 5):
        for batch in (16, 8):
            jobs.append(
                (
                    f"final_timing_{layer}_{batch}",
                    [
                        sys.executable,
                        "-m",
                        BASE + "time_fusion_candidate",
                        "--layer",
                        str(layer),
                        "--batch",
                        str(batch),
                        "--result",
                        str(OUT / f"final_timing_{layer}_{batch}.json"),
                    ],
                    False,
                )
            )
    jobs += [
        ("validation_final", [sys.executable, "-m", BASE + "run_fused_validation_queue"], True),
        (
            "pytest_final",
            [
                sys.executable,
                "-m",
                "pytest",
                "--confcutdir=" + str(ROOT / "tests"),
                "-q",
                str(ROOT / "tests/test_fused_decoder.py"),
            ],
            True,
        ),
        ("profiles_final", [sys.executable, "-m", BASE + "run_fused_profile_queue"], False),
    ]
    records = []
    for name, args, tracking in jobs:
        env = dict(
            os.environ,
            TT_METAL_TRACE_ALLOC_TRACKING="1" if tracking else "0",
            TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
        )
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        for k in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
            env.pop(k, None)
        sources = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "tt/fused_decoder.py",
                ROOT / "tests/run_fused_decoder.py",
                ROOT / "tests/profile_fused_decoder.py",
            ]
        }
        print("START", name, flush=True)
        with (OUT / (name + ".log")).open("w") as f:
            r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
        records.append(dict(name=name, command=args, exit_code=r.returncode, source_hashes=sources, tracking=tracking))
        (OUT / "acceptance_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END", name, r.returncode, flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
