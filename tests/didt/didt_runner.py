# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the DIDT ND/hang screens on one Blackhole galaxy and write status.json naming every failing chip.

Steps, fallbacks and settings come from didt_runner_config.json; rules for changing any of it are in
tests/didt/runner/CLAUDE.md. Run from the repo root with python_env active, on the galaxy itself (never on a login
node):

  python tests/didt/didt_runner.py                       every step
  python tests/didt/didt_runner.py --only nd:ff1,nd:mcast-ack
  python tests/didt/didt_runner.py --co                  collect gate only, no device work

Under sbatch, launch with `srun --propagate=NONE` and wrap the command in `sg slurmusers -c '...'`: otherwise the
job inherits ulimit -u 512 (too few processes for metal) and tt-smi -glx_reset_auto fails with Operation not
permitted.

Exit codes: 0 every run step PASS or WARN; 1 any step FAIL or HANG (first attempt); 2 config/usage error or chip
snapshot failed.
"""

import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path

if __package__:
    from .runner.chips import snapshot
    from .runner.config import ConfigError, load
    from .runner.steps import REPO, host, repo_commit, run
else:
    from runner.chips import snapshot
    from runner.config import ConfigError, load
    from runner.steps import REPO, host, repo_commit, run

DEFAULT_CONFIG = Path(__file__).resolve().parent / "didt_runner_config.json"
DEFAULT_OUT = REPO / "generated" / "didt_runner"


def _keys(text: str) -> set[str]:
    return {k.strip() for k in text.split(",") if k.strip()}


def _run_dir(out: Path) -> Path:
    base = out / f"LOG_{datetime.now():%Y_%m_%d_%H_%M}_{host()}_{repo_commit()}"
    run_dir, n = base, 1
    while run_dir.exists():
        n += 1
        run_dir = base.with_name(f"{base.name}_{n}")
    run_dir.mkdir(parents=True)
    return run_dir


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="didt_runner.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="default: %(default)s")
    ap.add_argument("--only", default="", help="comma-separated step keys to run")
    ap.add_argument("--skip", default="", help="comma-separated step keys to leave out")
    ap.add_argument("--co", action="store_true", help="collect gate only, no device work")
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="run dir parent, default: %(default)s")
    a = ap.parse_args(argv)

    try:
        cfg = load(a.config)
    except (ConfigError, OSError) as e:
        for line in getattr(e, "errors", [str(e)]):
            print(f"config: {line}", file=sys.stderr)
        return 2
    if a.list:
        for s in cfg.steps:
            print(f"{s.key}\n  ids:       {' '.join(s.ids)}")
            print(f"  fallbacks: {', '.join(f.name for f in s.fallbacks) or '-'}\n  note:      {s.note}")
        return 0

    only, skip = (_keys(a.only) or None), _keys(a.skip)
    unknown = sorted(((only or set()) | skip) - {s.key for s in cfg.steps})
    if unknown:
        print(f"unknown step keys: {', '.join(unknown)}; see --list", file=sys.stderr)
        return 2

    chips = snapshot()
    if not chips:
        print("chip snapshot failed: no chips under /sys/class/tenstorrent", file=sys.stderr)
        return 2
    run_dir = _run_dir(a.out.resolve())
    print(f"run dir {run_dir}  ({len(chips)} chips)", flush=True)
    # SIGTERM / SIGHUP (closed ssh or tmux) take the same path as Ctrl-C: kill the workload, mark the attempt.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)
    return run(cfg, run_dir, chips, only, skip, a.co, snap=snapshot)


def _interrupt(signum, frame):
    raise KeyboardInterrupt(signal.Signals(signum).name)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
