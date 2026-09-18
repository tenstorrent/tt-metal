#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
#   ./run_bench.sh          the four runs the report needs, then the tables
#   ./run_bench.sh check    correctness only, no timing
#
# Every other setting is an environment variable read by the test. Defaults
# live at the top of the test file. To sweep something else, set it here:
#
#   CCL_RUNS="ring:dram" CCL_DTYPE=bfloat8_b ./run_bench.sh
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

# topology:memory pairs. Ring and line feed the main figure, L1 the residency
# section.
RUNS=${CCL_RUNS:-"ring:dram line:dram ring:l1 line:l1"}

if [ "${1:-}" = "check" ]; then
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
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        printf -- '\npytest exited %s. Last 40 lines of %s:\n\n' "${rc}" "${log}"
        tail -40 "${log}"
        continue
    fi

    # Parse before the next run: the parser reads the newest profiler report.
    python "${REPORT_DIR}/parse_results.py" || printf -- '    parse failed\n'
done

printf -- '\ntables and CSVs in %s/data\n' "${REPORT_DIR}"
