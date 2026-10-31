"""Serialize device profiles and preserve complete-window and report provenance."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, choices=[0, 5])
p.add_argument("--batches", nargs="+", type=int, default=[1, 32, 16, 8])
a = p.parse_args()
base = Path("models/autoports/google_gemma_4_31b_it")
module = "models.autoports.google_gemma_4_31b_it.tests."
for layer in [a.layer] if a.layer is not None else [0, 5]:
    kind = "sliding_attention" if layer == 0 else "full_attention"
    for batch in a.batches:
        out = base / "doc/multichip_decoder/tracy_final" / kind / f"batch{batch}"
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, TT_METAL_DEVICE_PROFILER="1", TT_METAL_TRACE_ALLOC_TRACKING="0")
        env.pop("TT_METAL_WATCHER", None)
        cmd = [
            sys.executable,
            "-m",
            "tracy",
            "-r",
            "-p",
            "-v",
            "--no-web-server",
            "--op-support-count",
            "16384",
            "-o",
            str(out / "profile"),
            "-m",
            module + "profile_multichip_decoder",
            "--layer",
            str(layer),
            "--batch",
            str(batch),
            "--replicated",
            "--result",
            str(out / "host.json"),
        ]
        files = [
            base / "tt/multichip_decoder.py",
            base / "tt/optimized_decoder.py",
            base / "tests/profile_multichip_decoder.py",
        ]
        (out / "manifest.json").write_text(
            json.dumps(
                dict(
                    command=cmd,
                    source_hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                    watcher=False,
                    profiler=True,
                ),
                indent=2,
            )
            + "\n"
        )
        print("PROFILE", kind, batch, flush=True)
        with (out / "profile.log").open("w") as stream:
            run = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        if run.returncode:
            raise SystemExit(f"Failed {out}; inspect and recover before any next device command")
        csvs = list((out / "profile/reports").glob("*/ops_perf_results_*.csv"))
        assert len(csvs) == 1, csvs
        raw = csvs[0]
        commands = [
            [
                sys.executable,
                "-m",
                module + "summarize_multichip_profile",
                str(raw),
                "--trace-host-data",
                str(out / "profile/.logs/tracy_ops_data.csv"),
                "--layer-type",
                kind,
                "--batch",
                str(batch),
                "--expected-weight-dtype",
                "BFLOAT4_B",
                "--expected-fidelity",
                "LoFi",
                "--output",
                str(out / "performance.json"),
            ],
            [sys.executable, "-m", module + "select_multichip_replay", str(raw), str(out / "decode_replay.csv")],
        ]
        for i, command in enumerate(commands):
            with (out / f"analysis_{i}.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        for phase in ["prefill", "decode"]:
            source = raw if phase == "prefill" else out / "decode_replay.csv"
            command = [
                str(Path(sys.executable).parent / "tt-perf-report"),
                str(source),
                "--start-signpost",
                f"PERF_{phase.upper()}",
                "--end-signpost",
                f"PERF_{phase.upper()}_END",
            ]
            with (out / f"{phase}_perf_report.txt").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            with (out / f"{phase}_perf_report.console.log").open("w") as stream:
                subprocess.run(
                    command + ["--csv", str(out / f"{phase}_perf_report.csv")],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        (out / "analysis_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print("PASS", kind, batch, flush=True)
