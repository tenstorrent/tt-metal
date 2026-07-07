# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Whole-pipeline per-op profile from several SEGMENTED captures.

A full 928-op LTX block overflows the profiler perf-counter / noc-trace DRAM marker buffer, so a
single capture can never carry counter-grade per-op data for the whole pipeline (the tail stages —
VAE decode, audio decode — run last and lose their markers to the overflow). The workaround is to
capture each stage (or stage group that fits the buffer) in isolation, then stitch the per-op CSVs
back into one whole-pipeline table for a single ranking.

This driver does the stitch (``--merge``) and, given a segmentation map, prints the exact per-segment
capture commands to cover the pipeline (``--plan``). Each input is a scoped capture pinned to a stage
(``STAGE=path``); the merge concatenates their per-op rows in canonical stage order, rebasing GLOBAL
CALL COUNT into per-stage bands (so a stage from one run never collides with a stage from another),
tags each row with its STAGE, and hands the merged CSV to ltx_stage_bottlenecks for the rollup.

Dedup/stitch rule: a stage's rows come from exactly one capture — the last ``STAGE=`` given for that
stage wins (a later, cleaner re-capture supersedes an earlier one), so re-running one segment and
appending it replaces that stage without disturbing the rest.

stdlib only; imports the sibling ltx_stage_bottlenecks for the rollup.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import ltx_stage_bottlenecks as lsb

# Per-stage GLOBAL CALL COUNT band. Each stage's rebased call counts occupy [i*BAND, i*BAND+BAND);
# wide enough that no real per-device op stream (≤ a few thousand ops/stage) ever crosses a band.
_BAND = 1_000_000
_STAGE_COL = "STAGE"
_CALL = lsb.COL_CALL
_DEV = lsb.COL_DEV


def _parse_spec(spec: str) -> tuple[str, str]:
    """'VAE decode=/path.csv' -> ('VAE decode', '/path.csv'). Requires the STAGE= tag."""
    if "=" not in spec:
        raise ValueError(f"--csv needs a 'STAGE=path' tag (got {spec!r}); e.g. 'VAE decode=out.csv'")
    stage, path = (s.strip() for s in spec.split("=", 1))
    if stage not in lsb.CANONICAL_STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {lsb.CANONICAL_STAGES}")
    return stage, path


def merge_segments(specs: list[str]) -> tuple[list[dict], list[str]]:
    """Stitch scoped per-op CSVs into one whole-pipeline row list ordered by canonical stage.

    Returns (rows, fieldnames). GLOBAL CALL COUNT is rebased per stage-band and per device so the
    stitched stream stays cross-device rank-alignable and canonically ordered; a STAGE column is
    added for transparency. Last STAGE= for a stage wins (re-capture supersedes).
    """
    # Last-wins per stage.
    by_stage: dict[str, str] = {}
    for spec in specs:
        stage, path = _parse_spec(spec)
        by_stage[stage] = path

    fieldnames: list[str] = []
    out_rows: list[dict] = []
    for stage in lsb.CANONICAL_STAGES:
        path = by_stage.get(stage)
        if not path:
            continue
        rows = lsb.load_rows(path)
        if not rows:
            continue
        band_base = lsb.CANONICAL_STAGES.index(stage) * _BAND
        # Rebase call counts within the stage, per device, preserving original order.
        by_dev: dict[str, list[dict]] = {}
        for r in rows:
            by_dev.setdefault(r.get(_DEV, "0"), []).append(r)
        for dev_rows in by_dev.values():
            dev_rows.sort(key=lambda r: int(r.get(_CALL) or 0))
            for i, r in enumerate(dev_rows):
                r = dict(r)
                r[_CALL] = str(band_base + i)
                r[_STAGE_COL] = stage
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
                out_rows.append(r)
    if _STAGE_COL not in fieldnames:
        fieldnames.append(_STAGE_COL)
    return out_rows, fieldnames


def write_merged(rows: list[dict], fieldnames: list[str], out_path: str) -> None:
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _boundaries_for(rows: list[dict]) -> str:
    """Derive the --boundaries spec (per-stage GLOBAL CALL COUNT range) from the merged rows."""
    lo: dict[str, int] = {}
    hi: dict[str, int] = {}
    for r in rows:
        stage = r.get(_STAGE_COL)
        c = int(r.get(_CALL) or 0)
        if stage is None:
            continue
        lo[stage] = min(lo.get(stage, c), c)
        hi[stage] = max(hi.get(stage, c), c)
    return ",".join(f"{s}={lo[s]}:{hi[s]}" for s in lsb.CANONICAL_STAGES if s in lo)


# --- capture planning ------------------------------------------------------------------

# Which stages can be captured together in one perf-counter pass without overflowing the DRAM marker
# buffer. Ordered groups; the driver prints one capture command per group. Derived from the
# segmentation map (see PERF_COUNTERS.md): the light decode tail (VAE + audio + upsample) fits in one
# counter pass; each dense denoise block is captured on its own scoped harness.
_DEFAULT_SEGMENTS = [
    ("decode-tail", ["VAE decode", "Audio decode", "Latent upsample"]),
    ("stage1", ["Stage 1 denoise"]),
    ("stage2", ["Stage 2 denoise"]),
]


def plan_commands(workspace: str, env_yaml: str, out_dir: str, groups: str | None) -> list[str]:
    """Emit the per-segment prewarmed capture commands covering the whole pipeline.

    Each command warms kernels off-device then runs the scoped harness for one segment under tracy
    with perf counters, archiving the CSV under out_dir/<segment>/. The agent runs these serially on
    the broker, then feeds the resulting CSVs back to --merge.
    """
    prewarm = os.path.join(workspace, "tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh")
    harness = "models/tt_dit/tests/models/ltx/test_ltx_stage_scoped.py::test_stage_scoped"
    groups = groups or "fpu,instrn"
    cmds = []
    for seg, stages in _DEFAULT_SEGMENTS:
        stage_env = ",".join(_STAGE_TO_SELECTOR[s] for s in stages if s in _STAGE_TO_SELECTOR)
        pytest_cmd = (
            f"cd {workspace} && LTX_PROFILE_STAGES={stage_env} LTX_FAST=1 "
            f"python_env/bin/python -m tracy -p -r --profiler-capture-perf-counters {groups} "
            f"-m pytest {harness} -k bh_2x4sp1tp0 -s"
        )
        cmds.append(
            f"# segment '{seg}' -> {', '.join(stages)}\n{prewarm} -e {env_yaml} -w {workspace} -- \"{pytest_cmd}\""
        )
    return cmds


# Stage name -> LTX_PROFILE_STAGES selector token (the scoped harness reads these).
_STAGE_TO_SELECTOR = {
    "Stage 1 denoise": "s1",
    "Stage 2 denoise": "stage2",
    "Latent upsample": "upsample",
    "VAE decode": "vae",
    "Audio decode": "audio",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    m = sub.add_parser("merge", help="stitch scoped per-op CSVs into one whole-pipeline rollup")
    m.add_argument("--csv", action="append", required=True, metavar="STAGE=PATH", help="scoped capture; repeatable")
    m.add_argument("--out", required=True, help="write the merged whole-pipeline per-op CSV here")
    m.add_argument("--log", help="pipeline stderr with the per-stage wall-clock timers")
    m.add_argument("--budget", type=float, default=6.0, help="e2e budget seconds for the rollup (default 6.0)")
    m.add_argument("--stage-budgets", help="per-stage targets passed through to the rollup")
    m.add_argument("--top", type=int, default=6, help="top-N ops per stage in the rollup")
    m.add_argument("--baseline", help="baseline JSON for the rollup regression compare")
    m.add_argument("--save-baseline", help="write a rollup baseline JSON")
    m.add_argument("--html", help="write the rollup HTML here")
    m.add_argument("--gate", action="store_true", help="exit nonzero on over-budget / regression")

    p = sub.add_parser("plan", help="print the per-segment prewarmed capture commands for full coverage")
    p.add_argument("-w", "--workspace", required=True, help="built merged tree to run captures from")
    p.add_argument("-e", "--env-yaml", required=True, help="broker env yaml (TT_METAL_CACHE etc.)")
    p.add_argument("--out-dir", default="~/traces/ltx_pipeline", help="where each segment CSV lands")
    p.add_argument("--groups", help="perf-counter groups (default fpu,instrn)")

    args = ap.parse_args(argv)

    if args.mode == "plan":
        for c in plan_commands(args.workspace, args.env_yaml, args.out_dir, args.groups):
            print(c)
            print()
        return 0

    rows, fieldnames = merge_segments(args.csv)
    if not rows:
        sys.exit("no rows merged — check the STAGE=path inputs")
    write_merged(rows, fieldnames, args.out)
    boundaries = _boundaries_for(rows)
    print(f"merged {len(rows)} rows across {len({r[_STAGE_COL] for r in rows})} stages -> {args.out}")
    print(f"stage bands: {boundaries}\n")

    # Feed the merged CSV back through the rollup with the derived boundaries.
    lsb_argv = ["--csv", args.out, "--boundaries", boundaries, "--budget", str(args.budget), "--top", str(args.top)]
    if args.log:
        lsb_argv += ["--log", args.log]
    if args.stage_budgets:
        lsb_argv += ["--stage-budgets", args.stage_budgets]
    if args.baseline:
        lsb_argv += ["--baseline", args.baseline]
    if args.save_baseline:
        lsb_argv += ["--save-baseline", args.save_baseline]
    if args.html:
        lsb_argv += ["--html", args.html]
    if args.gate:
        lsb_argv += ["--gate"]
    return lsb.main(lsb_argv)


if __name__ == "__main__":
    raise SystemExit(main())
