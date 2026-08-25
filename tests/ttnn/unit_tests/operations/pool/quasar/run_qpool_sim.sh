#!/usr/bin/env bash
# Quasar pool debug runner — wraps test_qpool_debug.py with the craq-sim environment.
#
# Usage:  [QPOOL_* vars] ./run_qpool_sim.sh [extra pytest args]
# See test_qpool_debug.py's docstring for the QPOOL_* trace/pattern controls. Examples:
#   ./run_qpool_sim.sh                                          # tiny default trace, random input
#   QPOOL_SHAPE=112,112,64 QPOOL_PATTERN=sticks ./run_qpool_sim.sh
#   QPOOL_PATTERN=ones QPOOL_CORES=1 ./run_qpool_sim.sh
#
# Sim library resolution (in order):
#   1. QPOOL_SIM_SO=<path to libttsim.so>   (a soc_descriptor.yaml is staged beside it if missing)
#   2. $HOME/sim/qsr/libttsim.so
#   3. auto-staged from ${CRAQ_SIM_HOME:-/localdev/$USER/craq-sim}/src/_out/release_qsr/libttsim.so
#
# Runs are wrapped in `timeout` (QPOOL_TIMEOUT seconds, default 1800): a functional sim never
# fails a hang — it just spins — so a timeout exit (124) should be read as a device-side hang.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)

SIM_SO="${QPOOL_SIM_SO:-$HOME/sim/qsr/libttsim.so}"
if [[ ! -f "$SIM_SO" ]]; then
    CRAQ="${CRAQ_SIM_HOME:-/localdev/$USER/craq-sim}"
    BUILT="$CRAQ/src/_out/release_qsr/libttsim.so"
    if [[ -f "$BUILT" ]]; then
        mkdir -p "$(dirname "$SIM_SO")"
        cp "$BUILT" "$SIM_SO"
        echo "QPOOL: staged $BUILT -> $SIM_SO"
    else
        cat >&2 <<EOF
QPOOL: no Quasar sim library at $SIM_SO and no craq-sim build at $BUILT.
Build one with:
    git clone -b quasar git@github.com:tenstorrent/craq-sim.git $CRAQ
    cd $CRAQ && ./make.py --env TTSIM_MARCH=-march=x86-64-v3 src/_out/release_qsr/libttsim.so
(TTSIM_MARCH=-march=x86-64-v3 is only needed on non-AVX-512 hosts, e.g. Zen 2. Keep LTO on:
 disabling it with TTSIM_LTO=0 makes the sim ~20x slower; add it only if the LTO link fails.)
EOF
        exit 1
    fi
fi
# ttsim derives the SOC descriptor path from the .so path — it must sit beside it.
if [[ ! -f "$(dirname "$SIM_SO")/soc_descriptor.yaml" ]]; then
    cp "$REPO_ROOT/tt_metal/soc_descriptors/quasar_32_arch.yaml" "$(dirname "$SIM_SO")/soc_descriptor.yaml"
    echo "QPOOL: staged quasar_32_arch.yaml -> $(dirname "$SIM_SO")/soc_descriptor.yaml"
fi

export TT_METAL_SIMULATOR="$SIM_SO"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_FORCE_JIT_COMPILE=1
# Kernel asserts must be OFF for quasar pool (execution has to reach the reduce).
unset TT_METAL_LLK_ASSERTS TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS 2>/dev/null || true

cd "$REPO_ROOT"
TIMEOUT="${QPOOL_TIMEOUT:-1800}"
set +e
timeout --foreground "$TIMEOUT" pytest -q -s "$SCRIPT_DIR/test_qpool_debug.py" "$@"
rc=$?
set -e
if [[ $rc -eq 124 ]]; then
    echo "QPOOL: TIMED OUT after ${TIMEOUT}s — treat as a device-side HANG (the sim spins silently on hangs)."
fi
exit $rc
