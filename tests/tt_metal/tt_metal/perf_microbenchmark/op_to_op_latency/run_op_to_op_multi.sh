#!/usr/bin/env bash
# Run op-to-op latency benchmark N times, export each run, then aggregate.
#
# Usage:
#   ./run_op_to_op_multi.sh [NUM_RUNS]
#
# Env overrides:
#   CONFIG_LABEL  - subdirectory name under op_to_op_runs/ (default: "default")
#   EXTRA_ARGS    - extra CLI args to pass to the test binary
#                   (default: --use-trace --num-programs 2 --compute-nops 1000
#                             --use-device-profiler --use-realtime-profiler)
#   BUILD         - set to "0" to skip cmake build (default: build once)
#   TILES_PER_CORE - tiles/core for export buffering breakdown (default: 2)
#   INPUT_CB_DEPTH - input CB depth for export buffering breakdown (default: 2)
#   READER_PUSH    - reader push tiles for export (default: 1)
set -uo pipefail
# Intentionally NOT -e: a failing run shouldn't kill all subsequent runs.

NUM_RUNS="${1:-5}"
CONFIG_LABEL="${CONFIG_LABEL:-default}"
EXTRA_ARGS="${EXTRA_ARGS:---use-trace --num-programs 2 --compute-nops 1000 --use-device-profiler --use-realtime-profiler}"
TILES_PER_CORE="${TILES_PER_CORE:-2}"
INPUT_CB_DEPTH="${INPUT_CB_DEPTH:-2}"
READER_PUSH="${READER_PUSH:-1}"
MIN_PROG_ID="${MIN_PROG_ID:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME
export TT_METAL_DEVICE_PROFILER=1

BUILD_DIR="${BUILD_DIR:-${TT_METAL_HOME}/build_Release}"
TEST_BIN="${BUILD_DIR}/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
EXPORT_PY="${SCRIPT_DIR}/export_op_to_op_profiler_csv.py"
LOG_DIR="${TT_METAL_HOME}/generated/profiler/.logs"
RUNS_PARENT="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/${CONFIG_LABEL}"

echo "TT_METAL_HOME=${TT_METAL_HOME}"
echo "CONFIG_LABEL=${CONFIG_LABEL}"
echo "EXTRA_ARGS=${EXTRA_ARGS}"
echo "Runs: ${NUM_RUNS} -> ${RUNS_PARENT}/run_*"

if [[ "${BUILD:-1}" != "0" ]]; then
  cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
fi

mkdir -p "${RUNS_PARENT}"

# shellcheck disable=SC2086
EXTRA_ARGS_ARR=(${EXTRA_ARGS})

for i in $(seq 1 "${NUM_RUNS}"); do
  run_dir="${RUNS_PARENT}/run_${i}"
  mkdir -p "${run_dir}"
  echo "=== ${CONFIG_LABEL} Run ${i}/${NUM_RUNS} ==="
  rm -f "${LOG_DIR}/profile_log_device.csv" "${LOG_DIR}/profile_log_device_rt.csv"
  "${TEST_BIN}" "${EXTRA_ARGS_ARR[@]}"
  cp "${LOG_DIR}/profile_log_device.csv" "${run_dir}/profile_log_device.csv"
  if [[ -f "${LOG_DIR}/profile_log_device_rt.csv" ]]; then
    cp "${LOG_DIR}/profile_log_device_rt.csv" "${run_dir}/profile_log_device_rt.csv"
  fi
  python3 "${EXPORT_PY}" \
    --input-file "${run_dir}/profile_log_device.csv" \
    --rt-input-file "${run_dir}/profile_log_device_rt.csv" \
    --tiles-per-core "${TILES_PER_CORE}" \
    --input-cb-depth-tiles "${INPUT_CB_DEPTH}" \
    --reader-push-tiles "${READER_PUSH}" \
    --min-prog-id "${MIN_PROG_ID}" \
    --output-dir "${run_dir}"
done

python3 "${EXPORT_PY}" --aggregate-runs-dir "${RUNS_PARENT}" --output-dir "${RUNS_PARENT}"
echo "Done. Per-run CSVs under ${RUNS_PARENT}/run_*/"
echo "Multi-run summary: ${RUNS_PARENT}/multi_run_*.csv"
