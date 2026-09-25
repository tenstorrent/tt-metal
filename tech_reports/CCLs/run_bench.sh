#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
#   ./run_bench.sh [loudbox|galaxy] [check]
#
# Sweeps line at 2, 4 and 8 devices and ring where the axis closes, over DRAM,
# into data/runs/<timestamp>/, then times back-to-back calls at the smallest
# sizes into a second run. Then rebuilds the reports in results/ and the figures
# in images/ from every run kept there. `check` runs correctness only.
#
# Every other setting is an environment variable read by the test. Defaults
# live at the top of the test file. To sweep something else, set it here:
#
#   CCL_TOPOLOGY=ring CCL_MEMORY=dram,l1 CCL_SUBMESHES=1x8 CCL_OPS=all_gather ./run_bench.sh loudbox
#
# pytest, ttnn and tracy together emit a great many lines, so the run's log goes
# to its directory. On failure the tail is printed here.

set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1
export TT_METAL_HOME="$(pwd)"
export TT_METAL_RUNTIME_ROOT="$(pwd)"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source python_env/bin/activate

export TT_METAL_DEVICE_PROFILER=1
export ENABLE_TRACY=1
export TTNN_RUN_CCL_BANDWIDTH_BENCHMARK=1

REPORT_DIR=tech_reports/CCLs
TEST=tests/ttnn/unit_tests/benchmarks/test_ccl_bandwidth.py
CONFIGS="${REPORT_DIR}/data/ccl_bench_configs.jsonl"

case "${1:-loudbox}" in
    loudbox) MESH=1x8  SUBMESHES=1x2,1x4,1x8  AXIS=1 ;;
    galaxy)  MESH=8x4  SUBMESHES=2x1,4x1,8x1  AXIS=0 ;;
    *) echo "usage: $0 [loudbox|galaxy] [check]" >&2; exit 1 ;;
esac
export CCL_MESH=${CCL_MESH:-$MESH}
export CCL_SUBMESHES=${CCL_SUBMESHES:-$SUBMESHES}
export CCL_AXIS=${CCL_AXIS:-$AXIS}

if [ "${2:-}" = "check" ]; then
    python -m pytest "${TEST}" -k test_correctness -x
    exit $?
fi

# One pytest pass into its own run directory, then parse it. Arguments are
# NAME=value settings for this pass only.
bench() {
    RUN_DIR="${REPORT_DIR}/data/runs/$(date +%Y%m%d_%H%M%S)"
    LOG="${RUN_DIR}/bench.log"
    mkdir -p "${RUN_DIR}"
    printf -- '--- run %s\n    log: %s   (tail -f to follow)\n' "${RUN_DIR}" "${LOG}"

    # The parser pairs profiler signposts with this file in order, so a stale one
    # from an earlier run would break the pairing.
    rm -f "${CONFIGS}"
    env "$@" python -m tracy -o "${RUN_DIR}/profiler" -r -p -v -m pytest "${TEST}" -k test_perf > "${LOG}" 2>&1
    # `python -m tracy` exits 0 even when pytest fails, so read its summary line.
    if grep -qE '^=+ .*[0-9]+ (failed|error)' "${LOG}"; then
        printf -- '\npytest reported failures. Last 40 lines of %s:\n\n' "${LOG}"
        tail -40 "${LOG}"
        exit 1
    fi
    # A run where every cell skipped leaves nothing to parse.
    if [ ! -s "${CONFIGS}" ]; then
        printf -- '\nno measurements recorded. Last 40 lines of %s:\n\n' "${LOG}"
        tail -40 "${LOG}"
        exit 1
    fi
    mv "${CONFIGS}" "${RUN_DIR}/configs.jsonl"

    python "${REPORT_DIR}/parse.py" "${RUN_DIR}" || exit 1
    # The ops CSV is enough to re-parse. The rest is hundreds of MB.
    rm -rf "${RUN_DIR}/profiler/.logs"
    find "${RUN_DIR}/profiler" \( -name profile_log_device.csv -o -name '*.tracy' \) -delete
}

bench
# Time is all fixed latency at the smallest sizes, so time back-to-back calls there.
bench CCL_CALLS=8 CCL_MEMORY=dram CCL_MAX_BYTES=16384

python "${REPORT_DIR}/report.py" || exit 1
printf -- '\nreports in %s/results, figures in %s/images\n' "${REPORT_DIR}" "${REPORT_DIR}"
