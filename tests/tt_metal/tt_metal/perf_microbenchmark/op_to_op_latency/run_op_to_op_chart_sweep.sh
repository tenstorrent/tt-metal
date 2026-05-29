#!/usr/bin/env bash
# BW-vs-latency chart sweep across core counts.
#
# For each core count N in CORE_COUNTS:
#   1. Buffer-tune (sweep input & output CB depths) to find peak DRAM BW
#      and the smallest CB depths still within TOL_PCT of peak.
#   2. Steady-state validation run (trace warmup + later prog transitions)
#      at those CB depths to get clean op2op latency and chip dispatch done/go.
#
# Produces a single chart-ready CSV:
#   chart_data.csv
#   columns: num_cores, peak_input_gbps, smallest_input_cb,
#            peak_output_gbps, smallest_output_cb,
#            op2op_us_median, chip_done_to_go_ns_median,
#            brisc_done_to_go_us_median
#
# Usage:
#   ./run_op_to_op_chart_sweep.sh
#
# Env overrides:
#   CORE_COUNTS     - space-separated list (default: "1 2 4 10 20 40 80 110")
#   NUM_RUNS        - latency runs per core count (default: 3)
#   INPUT_DEPTHS    - comma list for input sweep (default: "2,4,6,8,12,16,24,32")
#   OUTPUT_DEPTHS   - comma list for output sweep (default: "2,4,8,16")
#   BW_PAGES        - pages/core during BW sweep (default: 32)
#   LATENCY_PAGES   - pages/core during latency phase (default: 16)
#   COMPUTE_NOPS    - compute NOPs/tile during latency phase (default: 2000)
#   NUM_PROGRAMS    - programs in trace (default: 8)
#   MIN_PROG_ID     - earliest from_prog_id to keep (default: 3)
#   TOL_PCT         - BW tolerance %% (default: 2)
#   BUILD           - "0" to skip cmake build (default: build once)

set -euo pipefail

CORE_COUNTS="${CORE_COUNTS:-1 2 4 10 20 40 80 110}"
NUM_RUNS="${NUM_RUNS:-5}"
INPUT_DEPTHS="${INPUT_DEPTHS:-2,4,6,8,12,16,24,32}"
OUTPUT_DEPTHS="${OUTPUT_DEPTHS:-2,4,8,16}"
BW_PAGES="${BW_PAGES:-256}"
LATENCY_PAGES="${LATENCY_PAGES:-16}"
COMPUTE_NOPS="${COMPUTE_NOPS:-2000}"
NUM_PROGRAMS="${NUM_PROGRAMS:-8}"
MIN_PROG_ID="${MIN_PROG_ID:-3}"
TOL_PCT="${TOL_PCT:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME
export TT_METAL_DEVICE_PROFILER=1

BUILD_DIR="${BUILD_DIR:-${TT_METAL_HOME}/build_Release}"
TEST_BIN="${BUILD_DIR}/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
LOG_DIR="${TT_METAL_HOME}/generated/profiler/.logs"
CHART_DIR="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/chart_sweep"
CHART_CSV="${CHART_DIR}/chart_data.csv"

mkdir -p "${CHART_DIR}"

if [[ "${BUILD:-1}" != "0" ]]; then
  cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
fi

echo "num_cores,peak_input_gbps,smallest_input_cb,peak_output_gbps,smallest_output_cb,op2op_us_median,chip_done_to_go_ns_median,brisc_done_to_go_us_median" > "${CHART_CSV}"

for N in ${CORE_COUNTS}; do
  echo "================ cores=${N} ================"

  RUN_DIR="${CHART_DIR}/cores_${N}"
  mkdir -p "${RUN_DIR}"
  TUNE_LOG="${RUN_DIR}/buffer_tune.log"

  # ---------------- Phase 1: buffer-tune sweep ----------------
  rm -f "${LOG_DIR}/profile_log_device.csv" "${LOG_DIR}/profile_log_device_rt.csv"
  "${TEST_BIN}" \
    --buffer-tune \
    --buffer-tune-input-depths "${INPUT_DEPTHS}" \
    --buffer-tune-output-depths "${OUTPUT_DEPTHS}" \
    --buffer-tune-pages-per-core "${BW_PAGES}" \
    --buffer-tune-bw-tolerance-pct "${TOL_PCT}" \
    --reader-push-tiles 2 \
    --num-pages-per-core "${LATENCY_PAGES}" \
    --compute-nops 0 \
    --num-programs 1 \
    --num-active-cores "${N}" \
    --use-device-profiler 2>&1 | tee "${TUNE_LOG}" >/dev/null

  IN_CB=$(grep 'smallest_input_cb_depth_at_peak=' "${TUNE_LOG}" | tail -1 \
            | sed -n 's/.*smallest_input_cb_depth_at_peak=\([0-9][0-9]*\).*/\1/p')
  OUT_CB=$(grep 'smallest_output_cb_depth_at_peak=' "${TUNE_LOG}" | tail -1 \
            | sed -n 's/.*smallest_output_cb_depth_at_peak=\([0-9][0-9]*\).*/\1/p')
  PEAK_IN_BW=$(grep 'buffer_tune: peak_dram_pipeline_gbps=' "${TUNE_LOG}" | tail -1 \
            | sed -n 's/.*peak_dram_pipeline_gbps=\([0-9.]*\).*/\1/p')
  PEAK_OUT_BW=$(grep 'output sweep peak_gbps=' "${TUNE_LOG}" | tail -1 \
            | sed -n 's/.*peak_gbps=\([0-9.]*\).*/\1/p')

  IN_CB="${IN_CB:-2}"
  OUT_CB="${OUT_CB:-2}"
  PEAK_IN_BW="${PEAK_IN_BW:-0}"
  PEAK_OUT_BW="${PEAK_OUT_BW:-0}"
  echo "  picked: in_cb=${IN_CB} out_cb=${OUT_CB} peak_in_gbps=${PEAK_IN_BW} peak_out_gbps=${PEAK_OUT_BW}"

  # ---------------- Phase 2: latency at picked CBs ----------------
  CONFIG_LABEL="chart_cores${N}" \
  MIN_PROG_ID="${MIN_PROG_ID}" \
  TILES_PER_CORE="${LATENCY_PAGES}" \
  INPUT_CB_DEPTH="${IN_CB}" \
  READER_PUSH=2 \
  BUILD=0 \
  EXTRA_ARGS="--use-trace --trace-warmup-replays 2 --num-programs ${NUM_PROGRAMS} \
              --compute-nops ${COMPUTE_NOPS} --use-device-profiler --use-realtime-profiler \
              --reader-push-tiles 2 \
              --input-cb-depth-tiles ${IN_CB} --output-cb-depth-tiles ${OUT_CB} \
              --num-pages-per-core ${LATENCY_PAGES} --num-active-cores ${N}" \
  bash "${SCRIPT_DIR}/run_op_to_op_multi.sh" "${NUM_RUNS}" 2>&1 | tail -8

  # ---------------- Phase 3: aggregate this row ----------------
  python3 - "${N}" "${IN_CB}" "${OUT_CB}" "${PEAK_IN_BW}" "${PEAK_OUT_BW}" \
          "${TT_METAL_HOME}/generated/profiler/op_to_op_runs/chart_cores${N}" \
          "${CHART_CSV}" "${MIN_PROG_ID}" << 'PYEOF'
import sys, os, glob, statistics as st
import pandas as pd

ncores, in_cb, out_cb, peak_in, peak_out, base, chart_csv, min_prog = sys.argv[1:]
runs = sorted(glob.glob(os.path.join(base, "run_*")))
op2op_meds, brisc_meds, chip_ns_meds = [], [], []
for r in runs:
    f = os.path.join(r, "profile_log_device_op_to_op_complete.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    if "from_prog_id" in df.columns:
        df = df[df["from_prog_id"] >= int(min_prog)]
    if df.empty:
        continue
    op2op_meds.append(df["gap_us"].median())
    brisc_meds.append(df["brisc_done_to_go_us"].median())
    chip_ns_meds.append(df["chip_dispatch_gap_ns"].median())

def med(xs):
    return st.median(xs) if xs else float("nan")

op2op_med = med(op2op_meds)
brisc_med = med(brisc_meds)
chip_med = med(chip_ns_meds)

with open(chart_csv, "a") as fh:
    fh.write(f"{ncores},{peak_in},{in_cb},{peak_out},{out_cb},"
             f"{op2op_med:.3f},{chip_med:.1f},{brisc_med:.3f}\n")
print(f"  cores={ncores}  op2op={op2op_med:.3f}us  brisc_dg={brisc_med:.3f}us  chip_dg={chip_med:.1f}ns")
PYEOF
done

echo
echo "================ DONE ================"
echo "Chart CSV: ${CHART_CSV}"
python3 -c "
import pandas as pd
df = pd.read_csv('${CHART_CSV}')
print(df.to_string(index=False))
"
