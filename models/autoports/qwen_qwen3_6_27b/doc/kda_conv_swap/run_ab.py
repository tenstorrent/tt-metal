# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profile the linear-attention prefill chunk with and without the fused KDA conv.

Both arms run the same harness, sequence, and iteration count and differ only
in the candidate policy, so the op-level delta is attributable to the one knob.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
HARNESS = REPO / "models/autoports/qwen_qwen3_6_27b/tests/linear_attention_synthetic_pcc.py"
ARMS = ("linear_final", "linear_kda_conv")


# One prefill of this harness dispatches ~1.4k programs.  The device profiler
# silently drops rows for programs past its cap, which the post-processor then
# reports as missing device data, so raise the cap above the op count.
PROFILER_ENV = {"TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT": "8192"}


def profile(candidate, artifact, sequence, iterations):
    artifact.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-p",
        "-o",
        str(artifact),
        str(HARNESS),
        "--mode",
        "prefill",
        "--sequence",
        str(sequence),
        "--optimized",
        "--candidate",
        candidate,
        "--iterations",
        str(iterations),
    ]
    run = subprocess.run(
        argv,
        cwd=REPO,
        env={**os.environ, **PROFILER_ENV},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (artifact / "profile.log").write_text(run.stdout)
    reports = sorted(artifact.glob("reports/*/ops_perf_results_*.csv"))
    if run.returncode != 0 or len(reports) != 1:
        raise SystemExit(f"{candidate}: exit={run.returncode} reports={[str(r) for r in reports]}")

    perf_argv = [
        str(REPO / "python_env/bin/tt-perf-report"),
        str(reports[0]),
        "--start-signpost",
        "PERF_PREFILL",
        "--end-signpost",
        "PERF_PREFILL_END",
        "--no-color",
        "--csv",
        str(artifact / "perf.csv"),
    ]
    perf = subprocess.run(perf_argv, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (artifact / "tt_perf_report.txt").write_text(perf.stdout)
    return {
        "candidate": candidate,
        "raw_report": str(reports[0].relative_to(REPO)),
        "profile_exit_status": run.returncode,
        "perf_exit_status": perf.returncode,
        "harness_lines": [ln for ln in run.stdout.splitlines() if "_PCC " in ln or "_LATENCY " in ln],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    results = [profile(arm, args.out / arm, args.sequence, args.iterations) for arm in ARMS]
    manifest = {"sequence": args.sequence, "iterations": args.iterations, "arms": results}
    (args.out / "ab_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
