# Settings for running tt-metal from a Slurm login node (source me). Every value can be overridden
# from the environment; nothing here names a user or a machine.
#
# XRUN_DATA        per-user directory on the shared filesystem (logs, tmp, ccache, eval DB, build HOME)
# XRUN_REPO        the tt-metal checkout to build and run in: always the one these scripts live in, so a clone
#                  (an eval run's) builds and tests itself even when a parent shell exported another checkout's
#                  XRUN_REPO; XRUN_REPO_OVERRIDE=<path> picks another. It must be on the shared filesystem.
# NODE_JOB         the reservation job (a running Slurm job of yours on a device node) to run steps in;
#                  default: found by on-node (XRUN_RESERVATION_NAME, else your only running device job)
# XRUN_BUILD_PARTITION / XRUN_BUILD_EXCLUDE / XRUN_TOOLCHAIN   where and how build-metal builds
_xrun_here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
export XRUN_DATA="${XRUN_DATA:-/data/$USER}"
export XRUN_REPO="${XRUN_REPO_OVERRIDE:-$(git -C "$_xrun_here" rev-parse --show-toplevel 2>/dev/null)}"
export XRUN_LOGS="${XRUN_LOGS:-$XRUN_DATA/logs}"
export XRUN_BUILD_PARTITION="${XRUN_BUILD_PARTITION:-cpu_only}"
export XRUN_BUILD_EXCLUDE="${XRUN_BUILD_EXCLUDE:-}"      # comma-separated build nodes to avoid (e.g. full /tmp)
export XRUN_TOOLCHAIN="${XRUN_TOOLCHAIN:-cmake/x86_64-linux-gcc-12-toolchain.cmake}"
unset _xrun_here

# The eval harness: keep its DB and run output on the shared filesystem (no node-local /localdev, and
# the central eval Postgres may be unreachable from the cluster).
export EVAL_FORCE_SQLITE="${EVAL_FORCE_SQLITE:-1}"
export EVAL_SQLITE_PATH="${EVAL_SQLITE_PATH:-$XRUN_DATA/evaldb/eval_runs.db}"
export EVAL_RUNS_DIR="${EVAL_RUNS_DIR:-$XRUN_DATA/evalruns}"   # pass as: run_eval.py --base-dir "$EVAL_RUNS_DIR"
mkdir -p "$XRUN_LOGS" "$(dirname "$EVAL_SQLITE_PATH")" "$EVAL_RUNS_DIR" 2>/dev/null || true
