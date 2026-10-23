#!/bin/bash
# sbatch wrapper for run_sol_validation.sh on a Blackhole Galaxy node.
#
#   sbatch -p bh_wan_machines -N 1 -t 01:00:00 -J optimization_run --propagate=NONE run.sh [step ...]
#
# Runs ./run_sol_validation.sh once per step (default: m11) from the repo checkout, with the environment the
# tests need: the Python venv, TT_METAL_HOME / PYTHONPATH / LD_LIBRARY_PATH, ARCH_NAME, the 180 W TDP limit.
# Every step writes its own log_val_<step>_*.txt files in the repo (and profile_<step>_logs/ for a profile
# sub-step); this script's own output goes to Slurm's slurm-<jobid>.out in the submit directory, and the
# step's logs are also bundled into sol_logs_<step>_<jobid>.tgz in the repo for copying back.
#
# Knobs (environment variables, pass with sbatch --export=ALL,NAME=value or export them before sbatch):
#   REPO=/path        repo checkout (default /data/sosborne/sol-attn/tt-metal)
#   VENV=/path        Python venv (default $REPO/python_env, the create_venv.sh default)
#   BUILD=1           run ./build_metal.sh --release first (profiler stays on, the default; ~20-40 min, so raise -t)
#   RESET=1           tt-smi -r before the steps (after a hang the wedged cores block the next device open)
#   TDP_WATTS=180     firmware TDP limit handed to the chips when the cluster opens (0 = board default)
#   SOL_STEPS="a b"   steps, if none are given as arguments
# TT_SOL_* / SOL_* variables are passed through untouched (e.g. TT_SOL_CHAIN_CREDITS=0 for the M11 A/B); the
# ones that are set are printed at the start so the log says what was measured.
set -u

REPO=${REPO:-/data/sosborne/sol-attn/tt-metal}
VENV=${VENV:-$REPO/python_env}
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && read -r -a STEPS <<< "${SOL_STEPS:-m11}"
JOB=${SLURM_JOB_ID:-nojob}
T0=$(date +%s)

cd "$REPO" || { echo "run.sh: repo not found: $REPO"; exit 2; }

echo "=== run.sh  $(date)  host=$(hostname)  job=$JOB  steps: ${STEPS[*]}"
echo "    repo:   $REPO"
echo "    branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)  head: $(git log -1 --format='%h %s' 2>/dev/null)"
dirty=$(git status --porcelain --untracked-files=no 2>/dev/null)
[ -n "$dirty" ] && { echo "    uncommitted tracked changes:"; echo "$dirty" | sed 's/^/      /'; }

# --- environment -----------------------------------------------------------------------------------------------
export TT_METAL_HOME="$REPO"
export PYTHONPATH="$REPO"
export LD_LIBRARY_PATH="$REPO/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ARCH_NAME=blackhole
export LOGURU_LEVEL=${LOGURU_LEVEL:-INFO}
export TT_METAL_TDP_LIMIT_WATTS=${TDP_WATTS:-180}
# The steps set the watcher / profiler / dprint themselves where they need them; a stray global one would slow
# every step and skew the timings.
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER TT_METAL_DPRINT_CORES TT_METAL_DPRINT_CHIPS
export PYTHONUNBUFFERED=1

if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "run.sh: venv not found at $VENV (create it with ./create_venv.sh, or set VENV=...)"; exit 2
fi

echo "    python: $(command -v python)  ($(python --version 2>&1))"
echo "    TT_METAL_TDP_LIMIT_WATTS=$TT_METAL_TDP_LIMIT_WATTS  ARCH_NAME=$ARCH_NAME"
sol_env=$(env | grep -E '^(TT_SOL_|SOL_)' | sort)
[ -n "$sol_env" ] && { echo "    Sol knobs in the environment:"; echo "$sol_env" | sed 's/^/      /'; }

# --- optional board reset / build ------------------------------------------------------------------------------
if [ "${RESET:-0}" = 1 ]; then
    echo "=== tt-smi -r"
    tt-smi -r || echo "    (reset returned $?; continuing)"
fi

if [ "${BUILD:-0}" = 1 ]; then
    echo "=== build  (log_build.txt)"
    ./build_metal.sh --release > log_build.txt 2>&1 || { echo "    build FAILED (see log_build.txt)"; tail -30 log_build.txt; exit 3; }
    echo "    build ok"
fi

# --- sanity: the build and the venv agree ----------------------------------------------------------------------
LIB=build/lib/libtt_metal.so
[ -f "$LIB" ] || { echo "run.sh: $LIB missing: build first (./build_metal.sh --release, or BUILD=1)"; exit 3; }
stale=$(find tt_metal ttnn/cpp -name '*.cpp' -newer "$LIB" -print -quit 2>/dev/null)
[ -n "$stale" ] && echo "    WARNING: C++ sources newer than $LIB (e.g. $stale): rebuild, or the run measures the old binary"
python - <<'EOF' || { echo "run.sh: 'import ttnn' failed: venv and build do not match (pip install -e . in the venv after the build)"; exit 3; }
import ttnn
print("    ttnn import ok:", ttnn.__file__)
EOF
ls /dev/tenstorrent >/dev/null 2>&1 || echo "    WARNING: /dev/tenstorrent not visible on this node"

# --- the steps -------------------------------------------------------------------------------------------------
rc=0
for step in "${STEPS[@]}"; do
    echo
    echo "=== step $step  $(date)"
    ./run_sol_validation.sh "$step"
    src=$?
    [ "$src" != 0 ] && rc=$src
    # Bundle the step's logs (and its profile dump if any) for copying back.
    files=$(ls log_val_"${step}"*.txt 2>/dev/null)
    [ -d "profile_${step}_logs" ] && files="$files profile_${step}_logs"
    # shellcheck disable=SC2086
    [ -n "$files" ] && tar czf "sol_logs_${step}_${JOB}.tgz" $files && echo "    logs bundled: $REPO/sol_logs_${step}_${JOB}.tgz"
done

echo
echo "=== done  $(date)  elapsed $(( ($(date +%s) - T0) / 60 )) min  exit $rc"
exit $rc
