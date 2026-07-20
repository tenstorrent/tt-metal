#!/bin/bash
# Preserve tt-exalens / emu launcher logs before the next pytest run.
# ExalensServer opens tt-exalens.log in 'w' mode each new process, so prior
# [4B MODE] / NNG lines are lost unless archived first.
#
# Usage (from python_tests/quasar):
#   ./preserve_exalens_logs.sh
#   pytest -x --run-simulator --port=5556 --timeout=1000 <test> | tee "perf_$(date -u +%Y%m%dT%H%M%SZ).txt"

set -euo pipefail
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ARCH_DIR="exalens_log_archive/${STAMP}"
mkdir -p "${ARCH_DIR}"

for f in tt-exalens.log emu_*_.log; do
    if [[ -e "$f" ]]; then
        cp -a "$f" "${ARCH_DIR}/"
        echo "archived $f -> ${ARCH_DIR}/"
    fi
done

# Optional: install instrumented launcher into the build tree.
SIM_BUILD="${TT_METAL_SIMULATOR:-${TT_UMD_SIMULATOR_PATH:-}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTRUMENTED="${INSTRUMENTED_LAUNCHER:-/proj_sw/user_dev/${USER}/tt-umd-simulators/emu/quasar-1x3/quasar-1x3_run_dev.instrumented.sh}"
if [[ -n "$SIM_BUILD" && -d "$SIM_BUILD" && -f "$INSTRUMENTED" ]]; then
    TARGET="${SIM_BUILD}/quasar-1x3_run_dev.sh"
    if [[ -f "$TARGET" && ! -f "${TARGET}.pre_instrument" ]]; then
        cp -a "$TARGET" "${TARGET}.pre_instrument"
        echo "backed up $TARGET -> ${TARGET}.pre_instrument"
    fi
    cp -a "$INSTRUMENTED" "$TARGET"
    echo "installed instrumented launcher -> $TARGET"
    echo "restore later with: cp -a ${TARGET}.pre_instrument $TARGET"
else
    echo "skip launcher install (set TT_METAL_SIMULATOR; need $INSTRUMENTED)"
fi

echo "archive dir: ${ARCH_DIR}"
echo "after pytest, also copy the new emu_*.log and tt-exalens.log into ${ARCH_DIR}/"
