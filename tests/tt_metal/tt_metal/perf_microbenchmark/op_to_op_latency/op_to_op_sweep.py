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

Every config is the fixed base below plus per-config overrides, so the sweep
only exercises knobs the binary already supports. `--list` prints the presets.

Examples:
  # sweep compute load, device profiler + accumulate (the CI-like flow)
  op_to_op_sweep.py --preset compute-nops --accumulate

  # sweep reader modes, rebuild first
  op_to_op_sweep.py --preset reader-mode --build

  # custom: two named configs (each is extra args appended to the base)
  op_to_op_sweep.py --config "nops0:--compute-nops 0" \\
                    --config "batch:--reader-batch-push"

  # see the exact commands without touching hardware
  op_to_op_sweep.py --preset writer-barrier --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Fixed part of every config, as an ordered {flag: value} map. Value knobs map to
# a string; bare flags map to True. Per-config overrides are merged *in place* over
# this map (see parse_overrides) -- the binary's arg parser takes the FIRST match of
# a flag (std::find in test_common.hpp), so a swept knob MUST replace the base entry
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

# Preset sweeps: name -> list of (label, extra-arg-string appended to BASE_ARGS).
# Only knobs the binary already parses; nothing here needs new C++. Repeated
# value-knobs (e.g. a second --compute-nops) take the last occurrence.
PRESETS: dict[str, list[tuple[str, str]]] = {
    "compute-nops": [
        ("nops0", "--compute-nops 0"),
        ("nops500", "--compute-nops 500"),
        ("nops1000", "--compute-nops 1000"),
        ("nops2000", "--compute-nops 2000"),
        ("nops4000", "--compute-nops 4000"),
    ],
    "num-programs": [
        ("prog4", "--num-programs 4"),
        ("prog8", "--num-programs 8"),
        ("prog16", "--num-programs 16"),
    ],
    "num-pages": [
        ("pages2", "--num-pages-per-core 2"),
        ("pages4", "--num-pages-per-core 4"),
        ("pages8", "--num-pages-per-core 8"),
        ("pages16", "--num-pages-per-core 16"),
    ],
    "reader-mode": [
        ("reader0-push1", ""),
        ("reader1-batch", "--reader-batch-push"),
        ("reader2-dbuf", "--reader-dbuf-trid --reader-trid-in-flight 4 --input-cb-depth-tiles 16"),
    ],
    "writer-barrier": [
        ("barrier0-ack", "--writer-end-barrier-mode 0"),
        ("barrier1-flush", "--writer-end-barrier-mode 1"),
        ("barrier2-none", "--writer-end-barrier-mode 2"),
    ],
    "noc": [
        ("noc-default", ""),
        ("noc-swapped", "--swap-nocs"),
    ],
    "active-cores": [
        ("cores8", "--num-active-cores 8"),
        ("cores32", "--num-active-cores 32"),
        ("cores56", "--num-active-cores 56"),
    ],
    "read-only": [
        ("readwrite", ""),
        ("readonly", "--read-only --skip-output-validation"),
    ],
    "page-size": [
        ("page1tile", "--page-size-tiles 1"),
        ("page2tile", "--page-size-tiles 2"),
        ("page4tile", "--page-size-tiles 4"),
    ],
}


def repo_root() -> Path:
    """Repo root: $TT_METAL_HOME if set, else three levels up from this file's
    tests/... location (self-locating, so the script runs from anywhere)."""
    env = os.environ.get("TT_METAL_HOME")
    if env:
        return Path(env)
    # .../tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/op_to_op_sweep.py
    return Path(__file__).resolve().parents[6]


def parse_overrides(extra: str) -> dict[str, object]:
    """Parse an extra-arg string ('--foo 3 --bar') into a {flag: value|True} map,
    mirroring the binary's parser: a flag consumes the next token as its value
    unless that token is itself a flag (then it's a bare boolean flag)."""
    tokens = shlex.split(extra)
    out: dict[str, object] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise SystemExit(f"expected a --flag, got {tok!r} in {extra!r}")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[tok] = tokens[i + 1]
            i += 2
        else:
            out[tok] = True
            i += 1
    return out


def render_argv(overrides: dict[str, object]) -> list[str]:
    """Merge overrides over BASE (in place) and render to an argv list."""
    merged = dict(BASE)
    merged.update(overrides)
    argv: list[str] = []
    for flag, val in merged.items():
        if val is True:
            argv.append(flag)
        elif val is False or val is None:
            continue
        else:
            argv.extend([flag, str(val)])
    return argv


def build_configs(args) -> list[tuple[str, dict[str, object]]]:
    """Return [(label, overrides)] for the selected preset and/or --config entries."""
    configs: list[tuple[str, dict[str, object]]] = []
    if args.preset:
        for label, extra in PRESETS[args.preset]:
            configs.append((label, parse_overrides(extra)))
    for spec in args.config or []:
        if ":" not in spec:
            raise SystemExit(f"--config must be 'label:args', got {spec!r}")
        label, extra = spec.split(":", 1)
        configs.append((label.strip(), parse_overrides(extra.strip())))
    if not configs:
        raise SystemExit("nothing to sweep: pass --preset NAME and/or --config 'label:args' (see --list)")
    return configs


def run_one(root: Path, bin_path: Path, overrides: dict[str, object], env: dict, dry_run: bool) -> bool:
    """Run the binary for one config. Returns True on success (exit 0)."""
    cmd = [str(bin_path), *render_argv(overrides)]
    if dry_run:
        print("  $ " + " ".join(shlex.quote(c) for c in cmd))
        return True
    proc = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True)
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
    ap.add_argument("--preset", choices=sorted(PRESETS), help="built-in sweep axis")
    ap.add_argument(
        "--config",
        action="append",
        help="custom config as 'label:extra args' (appended to the base); repeatable",
    )
    ap.add_argument("--min-prog-id", type=int, default=3, help="drop trace-start transitions below this prog id")
    ap.add_argument(
        "--accumulate", action="store_true", help="set TT_METAL_PROFILER_ACCUMULATE=1 (defer L1->DRAM dump)"
    )
    ap.add_argument("--build", action="store_true", help="cmake --build the target before sweeping")
    ap.add_argument("--bin", type=Path, default=None, help=f"test binary (default $TT_METAL_HOME/{REL_BIN})")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running anything")
    ap.add_argument("--list", action="store_true", help="list presets and exit")
    args = ap.parse_args()

    if args.list:
        print("presets:")
        for name, entries in sorted(PRESETS.items()):
            print(f"  {name:14s} {', '.join(lbl for lbl, _ in entries)}")
        return 0

    root = repo_root()
    bin_path = args.bin or (root / REL_BIN)
    configs = build_configs(args)

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
        print(f"ERROR: binary not found: {bin_path} (build it, or pass --bin / --build)", file=sys.stderr)
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
    header = f"{'config':22s} {'official_op2op_us':>18s} {'n':>6s} {'pack_to_unpack_us':>18s} {'n':>6s} {'kernel_dur_us':>14s}"
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
