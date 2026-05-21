#!/usr/bin/env bash
# Input CB depth vs DRAM bandwidth sweep, then op-to-op latency at smallest depth at peak BW.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME
export TT_METAL_DEVICE_PROFILER=1

BUILD_DIR="${BUILD_DIR:-${TT_METAL_HOME}/build_Release}"
TEST_BIN="${BUILD_DIR}/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
EXPORT_PY="${SCRIPT_DIR}/export_op_to_op_profiler_csv.py"
LOG_DIR="${TT_METAL_HOME}/generated/profiler/.logs"

INPUT_DEPTHS="${INPUT_DEPTHS:-2,4,6,8,12,16,24,32}"
OUTPUT_DEPTHS="${OUTPUT_DEPTHS:-}"
BW_PAGES="${BW_PAGES:-32}"
LATENCY_PAGES="${LATENCY_PAGES:-4}"
LATENCY_NOPS="${LATENCY_NOPS:-1000}"
NUM_PROGRAMS="${NUM_PROGRAMS:-2}"

cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
rm -rf "${HOME}/.cache/tt-metal-cache" "${TT_METAL_CACHE:-}" 2>/dev/null || true
mkdir -p "${LOG_DIR}"

ARGS=(
  --buffer-tune
  --buffer-tune-input-depths "${INPUT_DEPTHS}"
  --buffer-tune-pages-per-core "${BW_PAGES}"
  --reader-push-tiles 2
  --num-pages-per-core "${LATENCY_PAGES}"
  --compute-nops "${LATENCY_NOPS}"
  --num-programs "${NUM_PROGRAMS}"
  --use-trace
  --use-device-profiler
  --use-realtime-profiler
)
if [[ -n "${OUTPUT_DEPTHS}" ]]; then
  ARGS+=(--buffer-tune-output-depths "${OUTPUT_DEPTHS}")
fi

"${TEST_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_DIR}/buffer_tune_run.log"

INPUT_DEPTH=4
if grep -q 'smallest_input_cb_depth_at_peak=' "${LOG_DIR}/buffer_tune_run.log"; then
  INPUT_DEPTH="$(
    grep 'smallest_input_cb_depth_at_peak=' "${LOG_DIR}/buffer_tune_run.log" | tail -1 \
      | sed -n 's/.*smallest_input_cb_depth_at_peak=\([0-9][0-9]*\).*/\1/p'
  )"
fi

python3 "${EXPORT_PY}" \
  --input-file "${LOG_DIR}/profile_log_device.csv" \
  --rt-input-file "${LOG_DIR}/profile_log_device_rt.csv" \
  --tiles-per-core "${LATENCY_PAGES}" \
  --input-cb-depth-tiles "${INPUT_DEPTH}" \
  --reader-push-tiles 2

echo "BUFFER_TUNE rows: grep BUFFER_TUNE ${LOG_DIR}/buffer_tune_run.log"
