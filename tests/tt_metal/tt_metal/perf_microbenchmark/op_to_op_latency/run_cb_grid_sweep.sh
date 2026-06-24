#!/usr/bin/env bash
# Full input×output CB grid BW sweep (try more in/out CB combinations).
#
# Uses --buffer-tune-grid --buffer-tune-bw-only: every (input_cb, output_cb) pair
# is measured; logs BUFFER_TUNE rows with phase=cb_grid.
#
# Usage:
#   source python_env/bin/activate
#   ./run_cb_grid_sweep.sh
#
# Env:
#   CORE_COUNTS     - default "64 70 80 90 96"
#   INPUT_DEPTHS    - default "8,12,16,24,32,48,64"
#   OUTPUT_DEPTHS   - default "2,4,8,16,32,64"
#   BW_PAGES        - default 256
#   NUM_ACTIVE_CORES per run (same as each CORE_COUNTS entry)
#   BUILD           - default 1

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME

BUILD_DIR="${BUILD_DIR:-${TT_METAL_HOME}/build_Release}"
TEST_BIN="${BUILD_DIR}/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
OUT_DIR="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch2"
CSV="${OUT_DIR}/cb_grid_sweep.csv"

CORE_COUNTS="${CORE_COUNTS:-64 70 80 90 96}"
INPUT_DEPTHS="${INPUT_DEPTHS:-8,12,16,24,32,48,64}"
OUTPUT_DEPTHS="${OUTPUT_DEPTHS:-2,4,8,16,32,64}"
BW_PAGES="${BW_PAGES:-256}"
TRID_IN_FLIGHT="${TRID_IN_FLIGHT:-2}"
BUILD="${BUILD:-1}"

mkdir -p "${OUT_DIR}"
echo "num_cores,input_cb_depth,output_cb_depth,dram_pipeline_gbps,elapsed_us" > "${CSV}"

if [[ "${BUILD}" != "0" ]]; then
  cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
fi

for N in ${CORE_COUNTS}; do
  echo "======== CB grid cores=${N} ========"
  LOG="${OUT_DIR}/cb_grid_cores_${N}.log"
  "${TEST_BIN}" \
    --buffer-tune \
    --buffer-tune-grid \
    --buffer-tune-bw-only \
    --buffer-tune-input-depths "${INPUT_DEPTHS}" \
    --buffer-tune-output-depths "${OUTPUT_DEPTHS}" \
    --buffer-tune-pages-per-core "${BW_PAGES}" \
    --buffer-tune-bw-tolerance-pct 2 \
    --reader-dbuf-trid \
    --reader-trid-in-flight "${TRID_IN_FLIGHT}" \
    --reader-push-tiles 2 \
    --compute-nops 0 \
    --num-programs 1 \
    --num-active-cores "${N}" \
    2>&1 | tee "${LOG}"

  grep 'BUFFER_TUNE,phase=cb_grid,' "${LOG}" | while IFS= read -r line; do
    in_cb=$(echo "${line}" | sed -n 's/.*input_cb_depth=\([0-9][0-9]*\).*/\1/p')
    out_cb=$(echo "${line}" | sed -n 's/.*output_cb_depth=\([0-9][0-9]*\).*/\1/p')
    gbps=$(echo "${line}" | sed -n 's/.*dram_pipeline_gbps=\([0-9.]*\).*/\1/p')
    elapsed=$(echo "${line}" | sed -n 's/.*elapsed_us=\([0-9][0-9]*\).*/\1/p')
    echo "${N},${in_cb},${out_cb},${gbps},${elapsed}" >> "${CSV}"
  done

  peak=$(grep 'cb_grid peak_dram_pipeline_gbps=' "${LOG}" | tail -1 \
    | sed -n 's/.*peak_dram_pipeline_gbps=\([0-9.]*\).*/\1/p')
  in_pick=$(grep 'smallest_input_cb_depth_at_peak=' "${LOG}" | tail -1 \
    | sed -n 's/.*smallest_input_cb_depth_at_peak=\([0-9][0-9]*\).*/\1/p')
  out_pick=$(grep 'smallest_output_cb_depth_at_peak=' "${LOG}" | tail -1 \
    | sed -n 's/.*smallest_output_cb_depth_at_peak=\([0-9][0-9]*\).*/\1/p')
  echo "  peak=${peak} GB/s  picked in=${in_pick} out=${out_pick}"
done

echo
echo "CB grid CSV: ${CSV}"
echo "Peak per core (from grid summary lines in cb_grid_cores_*.log)"
