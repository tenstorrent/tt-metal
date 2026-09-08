#!/usr/bin/env bash
# Bring up Versim on this machine and run the harness branch's own smoke test.
#
# Versim runs on the host CPU with no device, so it can run while a sweep owns the
# board. Everything it touches is kept apart from the sweep: its own git worktree of
# lpremovic/versim-harness, its own build directory, its own output directory.
#
# usage: versim_smoke.sh [/path/to/versim-wormhole-b0]
#        or VERSIM=/path/to/versim-wormhole-b0 versim_smoke.sh
set -uo pipefail
say() { echo "=== $* -- $(date -u +%H:%M:%SZ) ==="; }

VERSIM="${VERSIM:-${1:-}}"
if [ -z "$VERSIM" ]; then
    VERSIM=$(ls -d /proj_sw/user_dev/"$USER"/tt-umd-simulators/build/versim-wormhole* 2>/dev/null | head -1)
fi
if [ -z "$VERSIM" ] || [ ! -f "$VERSIM/run.sh" ]; then
    echo "FATAL: no Versim build found. Pass its directory:"
    echo "       versim_smoke.sh /path/to/tt-umd-simulators/build/versim-wormhole-b0"
    echo "Looked in: /proj_sw/user_dev/$USER/tt-umd-simulators/build/"
    find /proj_sw/user_dev -maxdepth 4 -type d -name 'versim*' 2>/dev/null | head
    exit 1
fi
say "versim build: $VERSIM"; ls "$VERSIM" | head -8

MAIN=~/tt-metal; WT=~/versim-wt; OUT=~/versim
mkdir -p "$OUT"

# A separate worktree of the harness branch. The main tree stays on our branch and
# may be mid-sweep with patched files; nothing here touches it.
if [ ! -d "$WT" ]; then
    say "creating worktree $WT"
    ( cd "$MAIN" && git fetch -q origin lpremovic/versim-harness \
        && git worktree add -q "$WT" origin/lpremovic/versim-harness ) \
        || { echo "FATAL: could not create the worktree"; exit 1; }
fi
LLK="$WT/tt_metal/tt-llk"

# Reuse the main tree's SFPI toolchain and venv. TOOL_PATH is derived from LLK_ROOT,
# so the worktree needs tests/sfpi present; a symlink is enough.
[ -e "$LLK/tests/sfpi" ]  || ln -s "$MAIN/tt_metal/tt-llk/tests/sfpi"  "$LLK/tests/sfpi"
[ -e "$LLK/tests/.venv" ] || ln -s "$MAIN/tt_metal/tt-llk/tests/.venv" "$LLK/tests/.venv"
PY="$LLK/tests/.venv/bin/python"
say "tt-exalens $("$PY" -m pip show tt-exalens 2>/dev/null | awk '/^Version/{print $2}')  (branch needs >= 0.3.29)"

export LLK_HOME="$LLK"
export CHIP_ARCH=wormhole
export TT_METAL_SIMULATOR="$VERSIM"
export TT_SIMULATOR_LOCALHOST=1
export NNG_SOCKET_NAME="llk_$$"
export RUNNER_TEMP="$HOME/llk-versim-build"   # never the sweep's build directory
export USER=gitlab-ci                          # the only switch that turns the VCD off

cd "$LLK/tests/python_tests"
say "smoke test: smallest matmul, one tile per operand, LoFi (expect 1.5 to 3.5 min)"
"$PY" -u -m pytest --run-simulator -p no:randomly -v \
 'test_matmul.py::test_matmul[math_fidelity:LoFi-format_dest_acc_and_dims:(InputOutputFormat[A:Float16_b,B:Float16_b,out:Float16_b], <DestAccumulation.No: False>, ([32, 32], [32, 32]))]' \
 > "$OUT/smoke.log" 2>&1
RC=$?
tail -15 "$OUT/smoke.log"
say "pytest rc=$RC"

# Versim writes versim_<timestamp>_<socket>.log into the cwd, which is inside the repo.
ASSERTS=$(cat versim_*.log 2>/dev/null | grep -c 'ASSERTION FAILED' || true)
echo "RTL assertions in versim logs: ${ASSERTS:-0}"
mv -f versim_*.log "$OUT/" 2>/dev/null
ls -la "$OUT" | tail -5
say "DONE  (rc=$RC, 0 means Versim runs LLK kernels on this machine)"
