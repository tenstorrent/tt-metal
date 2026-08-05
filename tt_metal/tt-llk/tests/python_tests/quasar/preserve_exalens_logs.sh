#!/bin/bash
# Preserve tt-exalens / emu launcher logs before the next pytest run.
# ExalensServer opens tt-exalens.log in 'w' mode each new process, so prior
# [4B MODE] / NNG lines are lost unless archived first.
#
# Usage (from python_tests/quasar):
#   ./preserve_exalens_logs.sh
#   pytest -x --run-simulator --port=5556 --timeout=1000 <test> | tee "perf_$(date -u +%Y%m%dT%H%M%SZ).txt"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=perf_suite_common.sh
source "${SCRIPT_DIR}/perf_suite_common.sh"
cd "$PERF_SUITE_DIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ARCH_DIR="${PERF_SUITE_DIR}/exalens_log_archive/${STAMP}"
mkdir -p "${ARCH_DIR}"

for f in tt-exalens.log emu_*_.log; do
    if [[ -e "$f" ]]; then
        cp -a "$f" "${ARCH_DIR}/"
        echo "archived $f -> ${ARCH_DIR}/"
    fi
done

# Optionally install the instrumented launcher into the build tree. Missing
# launchers are informational so this script remains useful for log archiving.
if ! install_instrumented_launcher; then
    echo "skip launcher install; logs were still archived"
fi

echo "archive dir: ${ARCH_DIR}"
echo "after pytest, also copy the new emu_*.log and tt-exalens.log into ${ARCH_DIR}/"
