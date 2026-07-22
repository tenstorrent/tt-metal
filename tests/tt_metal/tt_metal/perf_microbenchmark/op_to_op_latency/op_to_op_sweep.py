#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep driver for the op-to-op latency microbenchmark.

Runs test_op_to_op_latency across a set of configs (permutations of the pinned
knobs), post-processes each run with op_to_op_postprocess.compute_metrics, and
prints one summary table. This is a *characterization* tool -- the "script to
sweep over the permutations and do the post-processing" -- it does NOT gate.
CI still gates a single pinned config (runtime_perf_profiler_tests.yaml); the
sweep is how we explore knobs and pick which config to promote to a gated
golden next.

Every config is the fixed BASE below plus a hardcoded per-config override map
from PRESETS, so the command handed to the binary is built entirely from
in-repo constants -- no free-form string is ever passed in on the command line.
To add a new axis, add an entry to PRESETS (a one-line, reviewed change); for a
truly one-off run, invoke the binary directly (see the README).

Examples:
  op_to_op_sweep.py --list                        # show the built-in axes
  op_to_op_sweep.py --preset compute-nops --accumulate
  op_to_op_sweep.py --preset reader-mode --build
  op_to_op_sweep.py --preset writer-barrier --dry-run   # print commands, no HW
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Defense-in-depth allow-patterns. Every flag/value handed to the binary is built
# from the constants below (BASE + PRESETS), never from user input, and is also
# checked against these patterns before exec. The call is list-form (shell=False),
# so no shell parses the tokens regardless.
_FLAG_RE = re.compile(r"^--[a-z][a-z0-9-]*$")
_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9,._-]*$")

# Fixed part of every config, as an ordered {flag: value} map. Value knobs map to
# a string; bare flags map to True. A preset's override map is merged *in place* over
# this map (see render_argv) -- the binary's arg parser takes the FIRST match of a
# flag (std::find in test_common.hpp), so a swept knob MUST replace the base entry
# rather than be appended after it, or the override would be silently ignored.
BASE: dict[str, object] = {
    "--use-trace": True,
    "--trace-warmup-replays": "2",
    "--num-programs": "8",
    "--num-pages-per-core": "4",
    "--compute-nops": "2000",
    "--use-device-profiler": True,
}

REL_BIN = "build/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
DEFAULT_DEVICE_LOG = "generated/profiler/.logs/profile_log_device.csv"

# Preset sweeps: name -> list of (label, override-map). Each override map is a
# hardcoded {flag: value|True} constant merged over BASE; only knobs the binary
# already parses, nothing here needs new C++.
PRESETS: dict[str, list[tuple[str, dict[str, object]]]] = {
    "compute-nops": [
        ("nops0", {"--compute-nops": "0"}),
        ("nops500", {"--compute-nops": "500"}),
        ("nops1000", {"--compute-nops": "1000"}),
        ("nops2000", {"--compute-nops": "2000"}),
        ("nops4000", {"--compute-nops": "4000"}),
    ],
    "num-programs": [
        ("prog4", {"--num-programs": "4"}),
        ("prog8", {"--num-programs": "8"}),
        ("prog16", {"--num-programs": "16"}),
    ],
    "num-pages": [
        ("pages2", {"--num-pages-per-core": "2"}),
        ("pages4", {"--num-pages-per-core": "4"}),
        ("pages8", {"--num-pages-per-core": "8"}),
        ("pages16", {"--num-pages-per-core": "16"}),
    ],
    "reader-mode": [
        ("reader0-push1", {}),
        ("reader1-batch", {"--reader-batch-push": True}),
        (
            "reader2-dbuf",
            {"--reader-dbuf-trid": True, "--reader-trid-in-flight": "4", "--input-cb-depth-tiles": "16"},
        ),
    ],
    "writer-barrier": [
        ("barrier0-ack", {"--writer-end-barrier-mode": "0"}),
        ("barrier1-flush", {"--writer-end-barrier-mode": "1"}),
        ("barrier2-none", {"--writer-end-barrier-mode": "2"}),
    ],
    "noc": [
        ("noc-default", {}),
        ("noc-swapped", {"--swap-nocs": True}),
    ],
    "active-cores": [
        ("cores8", {"--num-active-cores": "8"}),
        ("cores32", {"--num-active-cores": "32"}),
        ("cores56", {"--num-active-cores": "56"}),
    ],
    "read-only": [
        ("readwrite", {}),
        ("readonly", {"--read-only": True, "--skip-output-validation": True}),
    ],
    "page-size": [
        ("page1tile", {"--page-size-tiles": "1"}),
        ("page2tile", {"--page-size-tiles": "2"}),
        ("page4tile", {"--page-size-tiles": "4"}),
    ],
}


def repo_root() -> Path:
    """Repo root: $TT_METAL_HOME if set, else derived from this file's known
    tests/... location (self-locating, so the script runs from anywhere)."""
    env = os.environ.get("TT_METAL_HOME")
    if env:
        return Path(env)
    # .../tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/op_to_op_sweep.py
    return Path(__file__).resolve().parents[6]


def render_argv(overrides: dict[str, object]) -> list[str]:
    """Merge a preset's override map over BASE (in place) and render to an argv list.

    Both dicts are in-repo constants; each flag/value is still checked against the
    fixed allow-patterns as defense-in-depth before it is used."""
    merged = dict(BASE)
    merged.update(overrides)
    argv: list[str] = []
    for flag, val in merged.items():
        if not _FLAG_RE.match(str(flag)):
            raise SystemExit(f"invalid flag {flag!r} (allowed: --lower-case-with-dashes)")
        if val is True:
            argv.append(str(flag))
        elif val is False or val is None:
            continue
        else:
            sval = str(val)
            if not _VALUE_RE.match(sval):
                raise SystemExit(f"invalid value {sval!r} for {flag} (allowed: alnum plus , . _ -)")
            argv.extend([str(flag), sval])
    return argv


def run_one(root: Path, bin_path: Path, overrides: dict[str, object], env: dict, dry_run: bool) -> bool:
    """Run the binary for one config. Returns True on success (exit 0)."""
    # cmd is built from constants (bin_path is a resolved in-repo path; render_argv
    # merges only BASE + a hardcoded preset map). shell=False (list form) is explicit.
    cmd = [str(bin_path), *render_argv(overrides)]
    if dry_run:
        print("  $ " + " ".join(shlex.quote(c) for c in cmd))
        return True
    proc = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True, shell=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-8:] + proc.stderr.splitlines()[-8:])
        print(f"  run FAILED (exit {proc.returncode}); last lines:\n{tail}", file=sys.stderr)
        return False
    return True


def fmt(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), help="built-in sweep axis (see --list)")
    ap.add_argument("--min-prog-id", type=int, default=3, help="drop trace-start transitions below this prog id")
    ap.add_argument(
        "--accumulate", action="store_true", help="set TT_METAL_PROFILER_ACCUMULATE=1 (defer L1->DRAM dump)"
    )
    ap.add_argument("--build", action="store_true", help="cmake --build the target before sweeping")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running anything")
    ap.add_argument("--list", action="store_true", help="list presets and exit")
    args = ap.parse_args()

    if args.list:
        print("presets:")
        for name, entries in sorted(PRESETS.items()):
            print(f"  {name:14s} {', '.join(lbl for lbl, _ in entries)}")
        return 0

    if not args.preset:
        raise SystemExit("nothing to sweep: pass --preset NAME (see --list)")

    root = repo_root()
    bin_path = (root / REL_BIN).resolve()
    configs = PRESETS[args.preset]

    if args.build and not args.dry_run:
        print("building test_op_to_op_latency ...")
        rc = subprocess.run(
            ["cmake", "--build", "build", "--target", "test_op_to_op_latency", "-j"], cwd=str(root)
        ).returncode
        if rc != 0:
            print("build failed", file=sys.stderr)
            return 1

    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    if args.accumulate:
        env["TT_METAL_PROFILER_ACCUMULATE"] = "1"
    # compute_metrics imports the profiler-team parser from tools/; make sure it resolves.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "tools"), str(root), str(root / "ttnn"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    if not args.dry_run and not bin_path.exists():
        print(f"ERROR: binary not found: {bin_path} (build it, or pass --build)", file=sys.stderr)
        return 1

    # Import the post-processor lazily so --dry-run works without the tracy env.
    compute_metrics = None
    if not args.dry_run:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        for p in (str(root / "tools"), str(root), str(root / "ttnn")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from op_to_op_postprocess import compute_metrics as _cm  # noqa: E402

        compute_metrics = _cm

    device_csv = root / DEFAULT_DEVICE_LOG
    rows: list[tuple[str, dict | None]] = []
    for label, overrides in configs:
        desc = " ".join(k if v is True else f"{k} {v}" for k, v in overrides.items()) or "(base)"
        print(f"[{label}] {desc}")
        ok = run_one(root, bin_path, overrides, env, args.dry_run)
        if args.dry_run:
            rows.append((label, None))
            continue
        if not ok:
            rows.append((label, None))
            continue
        try:
            rows.append((label, compute_metrics(device_csv, None, args.min_prog_id)))
        except Exception as exc:  # noqa: BLE001 - report and continue the sweep
            print(f"  post-process FAILED: {exc}", file=sys.stderr)
            rows.append((label, None))

    if args.dry_run:
        return 0

    # Summary table.
    print()
    header = (
        f"{'config':22s} {'official_op2op_us':>18s} {'n':>6s} "
        f"{'pack_to_unpack_us':>18s} {'n':>6s} {'kernel_dur_us':>14s}"
    )
    print(header)
    print("-" * len(header))
    any_fail = False
    for label, m in rows:
        if m is None:
            print(f"{label:22s} {'FAILED':>18s}")
            any_fail = True
            continue
        print(
            f"{label:22s} {fmt(m.get('official_op2op_us')):>18s} {str(m.get('official_op2op_n', '-')):>6s} "
            f"{fmt(m.get('pack_to_unpack_op2op_us')):>18s} {str(m.get('pack_to_unpack_op2op_n', '-')):>6s} "
            f"{fmt(m.get('device_kernel_dur_us')):>14s}"
        )
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
