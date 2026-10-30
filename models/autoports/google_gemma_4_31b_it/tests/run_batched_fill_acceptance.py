"""Revalidate the selected batched-prefill rewrite; decode arithmetic is unchanged."""
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
        for batch in (8, 16, 32):
            name = f"batched_final_{layer}_{batch}"
            args = [
                sys.executable,
                "-m",
                BASE + "run_fused_decoder",
                "--layer",
                str(layer),
                "--batch",
                str(batch),
                "--seq",
                "2049" if batch == 32 else "33",
                "--result",
                str(OUT / (name + ".json")),
            ]
            extra = {}
            if batch == 32:
                args += ["--extended", "--new-bucket-case"]
                extra = {
                    "TT_METAL_WATCHER": "10",
                    "TT_METAL_LOGS_PATH": str(OUT / "watcher_batched_fill" / f"layer{layer}"),
                }
            jobs.append((name, args, True, extra))
        name = f"batched_equivalence_{layer}"
        jobs.append(
            (
                name,
                [
                    sys.executable,
                    "-m",
                    BASE + "test_fused_equivalence",
                    "--layer",
                    str(layer),
                    "--batch",
                    "32",
                    "--seq",
                    "1025",
                    "--result",
                    str(OUT / (name + ".json")),
                ],
                True,
                {},
            )
        )
        for batch in (8, 16, 32):
            name = f"batched_timing_{layer}_{batch}"
            jobs.append(
                (
                    name,
                    [
                        sys.executable,
                        "-m",
                        BASE + "time_fusion_candidate",
                        "--layer",
                        str(layer),
                        "--batch",
                        str(batch),
                        "--result",
                        str(OUT / (name + ".json")),
                    ],
                    False,
                    {},
                )
            )
    jobs.append(("batch_boundaries", [sys.executable, "-m", BASE + "check_fused_batch_boundaries"], True, {}))
    jobs.append(
        (
            "batched_profiles",
            [
                sys.executable,
                "-m",
                BASE + "run_fused_profile_queue",
                "--artifact-root",
                "tracy_batched_fill",
                "--batches",
                "32",
                "16",
                "8",
                "--retain-batch1",
            ],
            False,
            {},
        )
    )
    previous = (
        json.loads((OUT / "batched_acceptance_manifest.json").read_text())
        if "--resume" in sys.argv and (OUT / "batched_acceptance_manifest.json").exists()
        else []
    )
    records = []
    for name, args, tracking, extra in jobs:
        env = dict(
            os.environ,
            TT_METAL_TRACE_ALLOC_TRACKING="1" if tracking else "0",
            TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
        )
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
            env.pop(key, None)
        env.update(extra)
        source_hashes = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "tt/fused_decoder.py", ROOT / "tests/run_fused_decoder.py")
        }
        saved = next(
            (
                row
                for row in previous
                if row["name"] == name
                and row["exit_code"] == 0
                and row["command"] == args
                and row["source_hashes"] == source_hashes
            ),
            None,
        )
        if saved:
            records.append(saved)
            print("REUSE_PASS", name, flush=True)
            continue
        print("START", name, flush=True)
        with (OUT / (name + ".log")).open("w") as f:
            r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
        records.append(
            dict(
                name=name,
                command=args,
                exit_code=r.returncode,
                source_hashes=source_hashes,
                tracking=tracking,
                extra_environment=extra,
            )
        )
        (OUT / "batched_acceptance_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        print("END", name, r.returncode, flush=True)
        if r.returncode:
            raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
