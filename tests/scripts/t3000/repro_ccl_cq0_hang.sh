#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Local reproducer for the MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksCQ0
# hang on T3K (chip 3 stuck on AllGather after unit_tests_ttnn_ccl_ops).
#
# Runs the two relevant binaries back-to-back with the in-process operation
# timeout enabled and the triage hook wired up so a hang dumps dispatcher /
# worker state before the process is killed. Prints triage / hang-report
# artifact paths at the end.
#
# Usage:
#   tests/scripts/t3000/repro_ccl_cq0_hang.sh               # full back-to-back
#   tests/scripts/t3000/repro_ccl_cq0_hang.sh --solo        # only the 2nd binary
#   tests/scripts/t3000/repro_ccl_cq0_hang.sh --predecessor # only the 1st binary
#
# Relies on build/test/ttnn/* having already been built.

set -u
set -o pipefail

: "${TT_METAL_HOME:=$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${TT_METAL_HOME}"

MODE="full"
case "${1:-}" in
    --solo)        MODE="solo" ;;
    --predecessor) MODE="predecessor" ;;
    "")            MODE="full" ;;
    *)             echo "unknown arg: $1" >&2; exit 2 ;;
esac

LOG_DIR="${TT_METAL_HOME}/generated/repro_ccl_cq0_hang"
mkdir -p "${LOG_DIR}"
TRIAGE_DIR="${TT_METAL_HOME}/generated"
mkdir -p "${TRIAGE_DIR}"

# In-process operation timeout. Short enough to fire well before we get bored
# but long enough that a legitimate slow AllGather completes.
export TT_METAL_OPERATION_TIMEOUT_SECONDS="${TT_METAL_OPERATION_TIMEOUT_SECONDS:-30}"
export TT_METAL_LOGGER_LEVEL="${TT_METAL_LOGGER_LEVEL:-Info}"

# Wire up the triage / hang-report hook the same way the CI setup action does,
# but only if the user hasn't already configured something. See
# .github/actions/setup-job/action.yml.
if [[ -z "${TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE:-}" ]]; then
    HANG_REPORT_SCRIPT="${TT_METAL_HOME}/.github/scripts/utils/hang_report.py"
    TRIAGE_SCRIPT="${TT_METAL_HOME}/tools/tt-triage.py"
    if command -v python3 >/dev/null && [[ -f "${HANG_REPORT_SCRIPT}" && -f "${TRIAGE_SCRIPT}" ]]; then
        HANG_REPORT="python3 ${HANG_REPORT_SCRIPT}"
        TRIAGE="python3 ${TRIAGE_SCRIPT} --disable-progress --path=${TRIAGE_DIR}/triage_dump.txt --triage-summary-path=${TRIAGE_DIR}/triage_summary.txt"
        export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="${HANG_REPORT}; ${TRIAGE} 2>&1 | tee ${TRIAGE_DIR}/triage_output.txt 1>&2; ${HANG_REPORT} --update"
    fi
fi

run_step() {
    local name="$1"
    shift
    local log_file="${LOG_DIR}/${name}.log"
    echo "================================================================" | tee -a "${log_file}"
    echo "[repro] ${name}: $*" | tee -a "${log_file}"
    echo "[repro] log: ${log_file}" | tee -a "${log_file}"
    echo "================================================================" | tee -a "${log_file}"
    # Hard wall-clock ceiling is 5 minutes per step; relies on in-process
    # TT_METAL_OPERATION_TIMEOUT_SECONDS firing first for useful diagnostics.
    timeout 300 "$@" 2>&1 | tee -a "${log_file}"
    local rc="${PIPESTATUS[0]}"
    echo "[repro] ${name}: exit=${rc}" | tee -a "${log_file}"
    return "${rc}"
}

fail=0
if [[ "${MODE}" == "full" || "${MODE}" == "predecessor" ]]; then
    run_step predecessor ./build/test/ttnn/unit_tests_ttnn_ccl_ops || fail=$((fail + 1))
    sleep 2
fi
if [[ "${MODE}" == "full" || "${MODE}" == "solo" ]]; then
    run_step async_cq0 ./build/test/ttnn/test_ccl_multi_cq_multi_device \
        --gtest_filter="MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksCQ0" \
        || fail=$((fail + 1))
fi

echo "================================================================"
echo "[repro] logs in ${LOG_DIR}"
for f in "${TRIAGE_DIR}/triage_output.txt" "${TRIAGE_DIR}/triage_summary.txt" "${TRIAGE_DIR}/triage_dump.txt"; do
    [[ -f "${f}" ]] && echo "[repro] triage artifact: ${f}"
done
find "${TT_METAL_HOME}/generated/test_reports" -name "hang_report*" -printf "[repro] hang_report: %p\n" 2>/dev/null || true
echo "================================================================"

exit "${fail}"
