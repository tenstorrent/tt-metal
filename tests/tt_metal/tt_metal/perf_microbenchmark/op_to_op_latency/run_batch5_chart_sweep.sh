#!/usr/bin/env bash
# Batch 5: the full shrink-CB recipe with batch-4 kernel improvements.
#   Phase 1: full input×output CB grid → find peak BW
#   Phase 2: smallest input CB, then smallest output CB still within TOL_PCT of peak
#             (sequential shrink — not min(in+out) sum like batch 4)
#   Phase 3: latency at that CB with writer flush-on-pressure + trace
#
# Output:
#   generated/profiler/op_to_op_runs/batch5/chart_data.csv
#   generated/profiler/op_to_op_runs/batch5/batch5_vs_batch1_batch4.csv
#
# Usage:
#   source python_env/bin/activate
#   ./run_batch5_chart_sweep.sh

set -uo pipefail

CORE_COUNTS="${CORE_COUNTS:-1 2 4 10 20 40 80 110}"
NUM_RUNS="${NUM_RUNS:-5}"
INPUT_DEPTHS="${INPUT_DEPTHS:-8,12,16,24,32,48,64}"
OUTPUT_DEPTHS="${OUTPUT_DEPTHS:-2,4,8,16,32,64}"
BW_PAGES="${BW_PAGES:-256}"
LATENCY_PAGES="${LATENCY_PAGES:-16}"
COMPUTE_NOPS="${COMPUTE_NOPS:-2000}"
NUM_PROGRAMS="${NUM_PROGRAMS:-8}"
MIN_PROG_ID="${MIN_PROG_ID:-3}"
TOL_PCT="${TOL_PCT:-2}"
TRID_IN_FLIGHT="${TRID_IN_FLIGHT:-2}"
BUILD="${BUILD:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME
export TT_METAL_DEVICE_PROFILER=1
# dispatch-core profiling off (workers only); official op2op uses KERNEL zones, no DISP markers needed
# export TT_METAL_DEVICE_PROFILER_DISPATCH=1

PYTHON="${TT_METAL_HOME}/python_env/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

BUILD_DIR="${BUILD_DIR:-${TT_METAL_HOME}/build_Release}"
TEST_BIN="${BUILD_DIR}/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
PICK_PY="${SCRIPT_DIR}/pick_grid_min_cb.py"
OUT_DIR="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch5"
CHART_CSV="${OUT_DIR}/chart_data.csv"
GRID_CSV="${OUT_DIR}/grid_picks.csv"

mkdir -p "${OUT_DIR}"

if [[ "${BUILD}" != "0" ]]; then
  if command -v cmake >/dev/null 2>&1; then
    cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
  else
    /usr/local/bin/cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
  fi
fi

echo "num_cores,peak_bw_gbs,peak_bw_gbps,smallest_in_cb,smallest_out_cb,op2op_us_median,dg_median_ns,dg_issue_ns,cb_label,tune_method,writer_flush_mode" > "${CHART_CSV}"
echo "num_cores,peak_bw_gbs,smallest_in_cb,smallest_out_cb,batch4_in_cb,batch4_out_cb,pick_method" > "${GRID_CSV}"

MIN_IN=$(( 2 * TRID_IN_FLIGHT ))

for N in ${CORE_COUNTS}; do
  echo "================ batch5 cores=${N} ================"
  RUN_DIR="${OUT_DIR}/cores_${N}"
  mkdir -p "${RUN_DIR}"
  TUNE_LOG="${RUN_DIR}/grid_tune.log"

  rm -f "${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv"

  "${TEST_BIN}" \
    --buffer-tune \
    --buffer-tune-grid \
    --buffer-tune-bw-only \
    --buffer-tune-input-depths "${INPUT_DEPTHS}" \
    --buffer-tune-output-depths "${OUTPUT_DEPTHS}" \
    --buffer-tune-pages-per-core "${BW_PAGES}" \
    --buffer-tune-bw-tolerance-pct "${TOL_PCT}" \
    --writer-flush-on-pressure \
    --reader-dbuf-trid \
    --reader-trid-in-flight "${TRID_IN_FLIGHT}" \
    --reader-push-tiles 2 \
    --compute-nops 0 \
    --num-programs 1 \
    --num-active-cores "${N}" \
    2>&1 | tee "${TUNE_LOG}" >/dev/null

  read -r PEAK IN_CB OUT_CB <<< "$("${PYTHON}" "${PICK_PY}" "${TUNE_LOG}" --tolerance-pct "${TOL_PCT}" --min-input-cb "${MIN_IN}")"

  # Batch 4 pick (min in+out sum) for comparison in grid_picks.csv
  B4_IN=$(grep 'smallest_input_cb_depth_at_peak=' "${TUNE_LOG}" | tail -1 \
    | sed -n 's/.*smallest_input_cb_depth_at_peak=\([0-9][0-9]*\).*/\1/p')
  B4_OUT=$(grep 'smallest_output_cb_depth_at_peak=' "${TUNE_LOG}" | tail -1 \
    | sed -n 's/.*smallest_output_cb_depth_at_peak=\([0-9][0-9]*\).*/\1/p')

  PEAK_GBPS=$(awk "BEGIN {printf \"%.1f\", ${PEAK} * 8}")
  echo "  batch5 pick (shrink): peak=${PEAK} GB/s (${PEAK_GBPS} Gbps) in=${IN_CB} out=${OUT_CB}"
  echo "  batch4 pick (sum):    in=${B4_IN:-?} out=${B4_OUT:-?}"
  echo "${N},${PEAK},${IN_CB},${OUT_CB},${B4_IN:-},${B4_OUT:-},shrink_input_then_output" >> "${GRID_CSV}"

  CONFIG_LABEL="batch5_cores${N}" \
  MIN_PROG_ID="${MIN_PROG_ID}" \
  TILES_PER_CORE="${LATENCY_PAGES}" \
  INPUT_CB_DEPTH="${IN_CB}" \
  READER_PUSH=2 \
  BUILD=0 \
  EXTRA_ARGS="--use-trace --trace-warmup-replays 2 --num-programs ${NUM_PROGRAMS} \
              --compute-nops ${COMPUTE_NOPS} --use-device-profiler --use-realtime-profiler \
              --writer-flush-on-pressure \
              --reader-dbuf-trid --reader-trid-in-flight ${TRID_IN_FLIGHT} \
              --reader-push-tiles 2 \
              --input-cb-depth-tiles ${IN_CB} --output-cb-depth-tiles ${OUT_CB} \
              --num-pages-per-core ${LATENCY_PAGES} --num-active-cores ${N}" \
  bash "${SCRIPT_DIR}/run_op_to_op_multi.sh" "${NUM_RUNS}" 2>&1 | tail -8

  "${PYTHON}" - "${N}" "${PEAK}" "${PEAK_GBPS}" "${IN_CB}" "${OUT_CB}" \
    "${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch5_cores${N}" \
    "${CHART_CSV}" "${MIN_PROG_ID}" << 'PYEOF'
import csv
import glob
import os
import statistics as st
import sys

ncores, peak, peak_gbps, in_cb, out_cb, base, chart_csv, min_prog = sys.argv[1:9]
op2op, dg_med, dg_issue = [], [], []

for r in sorted(glob.glob(os.path.join(base, "run_*"))):
    f = os.path.join(r, "profile_log_device_op_to_op_complete.csv")
    if not os.path.exists(f):
        continue
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    rows = [row for row in rows if int(float(row.get("from_prog_id", 0))) >= int(min_prog)]
    if not rows:
        continue
    op2op.append(st.median([float(row["gap_us"]) for row in rows]))
    dg_vals = [float(row["dg_median_ns"]) for row in rows if row.get("dg_median_ns") not in ("", "nan")]
    if dg_vals:
        dg_med.append(st.median(dg_vals))
    iss = [float(row["dg_issue_ns"]) for row in rows if row.get("dg_issue_ns") not in ("", "nan")]
    if iss:
        dg_issue.append(st.median(iss))


def med(xs):
    return st.median(xs) if xs else float("nan")

op2op_m, dg_m, dg_i = med(op2op), med(dg_med), med(dg_issue)
cb_label = f"in={in_cb}/out={out_cb}"
with open(chart_csv, "a") as fh:
    fh.write(
        f"{ncores},{peak},{peak_gbps},{in_cb},{out_cb},"
        f"{op2op_m:.3f},{dg_m:.1f},{dg_i:.1f},{cb_label},grid_shrink,flush_on_pressure\n"
    )
print(f"  cores={ncores}  peak={peak} GB/s  op2op={op2op_m:.3f}us  dg_median={dg_m:.0f}ns")
PYEOF
done

echo
echo "Batch 5 chart: ${CHART_CSV}"
echo "Grid picks:    ${GRID_CSV}"
"${PYTHON}" "${SCRIPT_DIR}/rebuild_batch5_compare.py" || true
