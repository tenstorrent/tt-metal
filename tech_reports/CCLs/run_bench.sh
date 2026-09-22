#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
#   ./run_bench.sh [loudbox|galaxy] [check]
#
# Sweeps ring over DRAM at 2, 4 and 8 devices, then writes the tables. A device
# count whose axis has no wrap-around link runs as a line instead; the tables
# record which. `check` runs correctness only, no timing.
#
# Every other setting is an environment variable read by the test. Defaults
# live at the top of the test file. To sweep something else, set it here:
#
#   CCL_RUNS="ring:dram line:dram" CCL_DTYPE=bfloat8_b ./run_bench.sh galaxy
#
# pytest, ttnn and tracy together emit a great many lines, so each run goes to
# its own log under logs/. On failure the tail is printed here.

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
DATA_DIR="${REPORT_DIR}/data"
LOG_DIR="${DATA_DIR}/logs"

mkdir -p generated "${LOG_DIR}"

case "${1:-loudbox}" in
    loudbox) MESH=1x8  SUBMESHES=1x2,1x4,1x8  AXIS=1 ;;
    galaxy)  MESH=8x4  SUBMESHES=2x1,4x1,8x1  AXIS=0 ;;
    *) echo "usage: $0 [loudbox|galaxy] [check]" >&2; exit 1 ;;
esac
export CCL_MESH=${CCL_MESH:-$MESH}
export CCL_SUBMESHES=${CCL_SUBMESHES:-$SUBMESHES}
export CCL_AXIS=${CCL_AXIS:-$AXIS}

RUNS=${CCL_RUNS:-"ring:dram"}

if [ "${2:-}" = "check" ]; then
    python -m pytest "${TEST}" -k test_correctness -x
    exit $?
fi

for run in ${RUNS}; do
    topology=${run%%:*}
    memory=${run##*:}
    log="${LOG_DIR}/${topology}_${memory}.log"

    printf -- '--- %s %s\n' "${topology}" "${memory}"
    printf -- '    log: %s   (tail -f to follow)\n' "${log}"

    # The parser pairs profiler signposts with this file in order, so a stale
    # one from an earlier run makes it refuse to parse.
    rm -f "${CONFIGS}"

    CCL_TOPOLOGY="${topology}" CCL_MEMORY="${memory}" \
        python -m tracy -r -p -v -m pytest "${TEST}" -k test_perf \
        > "${log}" 2>&1
    # `python -m tracy` exits 0 even when pytest fails, so read its summary line.
    if grep -qE '^=+ .*[0-9]+ (failed|error)' "${log}"; then
        printf -- '\npytest reported failures. Last 40 lines of %s:\n\n' "${log}"
        tail -40 "${log}"
        continue
    fi

    # A run where every cell skipped leaves nothing to parse, and an all-skipped
    # summary says neither "failed" nor "error".
    if [ ! -s "${CONFIGS}" ]; then
        printf -- '\nno measurements recorded. Last 40 lines of %s:\n\n' "${log}"
        tail -40 "${log}"
        continue
    fi

    # Parse before the next run: the parser reads the newest profiler report.
    python "${REPORT_DIR}/parse_results.py" || printf -- '    parse failed\n'
done

python "${REPORT_DIR}/plot_results.py" || printf -- 'plot failed\n'

printf -- '\ntables and CSVs in %s/data, figures in %s/images\n' "${REPORT_DIR}" "${REPORT_DIR}"
