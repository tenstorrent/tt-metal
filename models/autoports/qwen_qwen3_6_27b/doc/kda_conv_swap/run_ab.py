# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profile the linear-attention layer with and without the fused KDA conv.

Both arms run the same harness, shape and iteration count and differ only in
the candidate policy, so the op-level delta is attributable to the one knob.
`--phase prefill` uses the eager chunked-prefill harness at batch 1;
`--phase decode` uses the traced batch-32 decode harness, which is the shape
the layer actually spends its time in.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
PREFILL_HARNESS = REPO / "models/autoports/qwen_qwen3_6_27b/tests/linear_attention_synthetic_pcc.py"
DECODE_HARNESS = REPO / "models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py"
ARMS = ("linear_final", "linear_kda_conv")

# One prefill of this harness dispatches ~1.4k programs.  The device profiler
# silently drops rows for programs past its cap, which the post-processor then
# reports as missing device data, so raise the cap above the op count.
PROFILER_ENV = {"TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT": "8192"}


KEEP_LOG = re.compile(
    r"FALLBACK_AUDIT|OPTIMIZED_POLICY|_PCC |_LATENCY |ops_perf_results_|"
    r"Traceback|Error|error:|Warning|WARNING|FATAL|Assertion",
)


def _retained_log(stdout):
    """Keep the provenance lines, drop the per-op profiler chatter.

    A traced decode capture emits hundreds of KB of routine device logging,
    which blows the repo's 500 KB file limit and carries no evidence.
    """
    kept = [line for line in stdout.splitlines() if KEEP_LOG.search(line)]
    return "\n".join(kept) + "\n"


def harness_argv(candidate, sequence, iterations, phase):
    if phase == "prefill":
        return [
            str(PREFILL_HARNESS),
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
    return [
        str(DECODE_HARNESS),
        "--kind",
        "linear",
        "--batch",
        "32",
        "--optimized",
        "--candidate",
        candidate,
        "--steps",
        str(iterations + 1),
    ]


def profile(candidate, artifact, sequence, iterations, phase):
    artifact.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "tracy", "-r", "-p", "-o", str(artifact)]
    argv += harness_argv(candidate, sequence, iterations, phase)
    run = subprocess.run(
        argv,
        cwd=REPO,
        env={**os.environ, **PROFILER_ENV},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (artifact / "profile.log").write_text(_retained_log(run.stdout))
    reports = sorted(artifact.glob("reports/*/ops_perf_results_*.csv"))
    if run.returncode != 0 or len(reports) != 1:
        raise SystemExit(f"{candidate}: exit={run.returncode} reports={[str(r) for r in reports]}")

    signpost = "PERF_PREFILL" if phase == "prefill" else "PERF_DECODE"
    perf_argv = [
        str(REPO / "python_env/bin/tt-perf-report"),
        str(reports[0]),
        "--start-signpost",
        signpost,
        "--end-signpost",
        f"{signpost}_END",
        "--no-color",
        "--csv",
        str(artifact / "perf.csv"),
    ]
    perf = subprocess.run(perf_argv, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (artifact / "tt_perf_report.txt").write_text(perf.stdout)
    return {
        "candidate": candidate,
        "phase": phase,
        "raw_report": str(reports[0].relative_to(REPO)),
        "profile_exit_status": run.returncode,
        "perf_exit_status": perf.returncode,
        "harness_lines": [ln for ln in run.stdout.splitlines() if "_PCC " in ln or "_LATENCY " in ln],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (Path(__file__).parent / args.phase)
    results = [profile(arm, out / arm, args.sequence, args.iterations, args.phase) for arm in ARMS]
    manifest = {
        "phase": args.phase,
        "sequence": args.sequence,
        "iterations": args.iterations,
        "arms": results,
    }
    (out / "ab_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
