#!/usr/bin/env bash
# Batch 7: same methodology as batch 4 (grid CB tune + writer flush-on-pressure)
# but each program in the trace reads/writes a DISJOINT DRAM tile slice via
# --cross-program-dram-offset. This forces the DRAM controller to open new rows
# per program enqueue, isolating any optimistic "warm-row" cache effect that
# could shrink op-to-op gap in batch 4 where every program touched the same
# pages back-to-back.
#
# Output:
#   generated/profiler/op_to_op_runs/batch7/chart_data.csv
#   generated/profiler/op_to_op_runs/batch7/batch7_vs_batch4.csv  (rebuild script)
#
# Usage:
#   source python_env/bin/activate   # recommended for export
#   ./run_batch7_dram_offset.sh
#
# Env knobs match batch 4. Most relevant:
#   NUM_PROGRAMS -> bigger N means bigger DRAM stride covered per replay
#                   (host allocates (NUM_PROGRAMS+1) * per-program slice).

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
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-${TT_METAL_HOME}}"
export TT_METAL_DEVICE_PROFILER=1
# dispatch-core profiling off (workers only); official op2op uses KERNEL zones, no DISP markers needed
# export TT_METAL_DEVICE_PROFILER_DISPATCH=1

BUILD_DIR="${BUILD_DIR:-${TT_METAL_HOME}/build_Release}"
TEST_BIN="${BUILD_DIR}/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
OUT_DIR="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch7"
CHART_CSV="${OUT_DIR}/chart_data.csv"

mkdir -p "${OUT_DIR}"

if [[ "${BUILD}" != "0" ]]; then
  if command -v cmake >/dev/null 2>&1; then
    cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
  else
    /usr/local/bin/cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
  fi
fi

echo "num_cores,peak_bw_gbs,peak_bw_gbps,smallest_in_cb,smallest_out_cb,op2op_us_median,dg_median_ns,dg_issue_ns,cb_label,tune_method,writer_flush_mode,dram_offset" > "${CHART_CSV}"

for N in ${CORE_COUNTS}; do
  echo "================ batch7 cores=${N} ================"
  RUN_DIR="${OUT_DIR}/cores_${N}"
  mkdir -p "${RUN_DIR}"
  TUNE_LOG="${RUN_DIR}/grid_tune.log"

  rm -f "${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv"

  # Phase 1: grid BW tune. Single-program so offset does not affect tune.
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

  PEAK=$(grep 'cb_grid peak_dram_pipeline_gbps=' "${TUNE_LOG}" | tail -1 \
    | sed -n 's/.*peak_dram_pipeline_gbps=\([0-9.]*\).*/\1/p')
  IN_CB=$(grep 'smallest_input_cb_depth_at_peak=' "${TUNE_LOG}" | tail -1 \
    | sed -n 's/.*smallest_input_cb_depth_at_peak=\([0-9][0-9]*\).*/\1/p')
  OUT_CB=$(grep 'smallest_output_cb_depth_at_peak=' "${TUNE_LOG}" | tail -1 \
    | sed -n 's/.*smallest_output_cb_depth_at_peak=\([0-9][0-9]*\).*/\1/p')

  PEAK="${PEAK:-0}"
  IN_CB="${IN_CB:-8}"
  OUT_CB="${OUT_CB:-2}"
  MIN_IN=$(( 2 * TRID_IN_FLIGHT ))
  if (( IN_CB < MIN_IN )); then
    echo "  bumping in_cb ${IN_CB} -> ${MIN_IN} (mode2 N=${TRID_IN_FLIGHT})"
    IN_CB="${MIN_IN}"
  fi
  PEAK_GBPS=$(awk "BEGIN {printf \"%.1f\", ${PEAK} * 8}")
  echo "  grid pick: peak=${PEAK} GB/s (${PEAK_GBPS} Gbps) in=${IN_CB} out=${OUT_CB}"

  # Phase 2: latency w/ cross-program DRAM offset.
  CONFIG_LABEL="batch7_cores${N}" \
  MIN_PROG_ID="${MIN_PROG_ID}" \
  TILES_PER_CORE="${LATENCY_PAGES}" \
  INPUT_CB_DEPTH="${IN_CB}" \
  READER_PUSH=2 \
  BUILD=0 \
  EXTRA_ARGS="--use-trace --trace-warmup-replays 2 --num-programs ${NUM_PROGRAMS} \
              --compute-nops ${COMPUTE_NOPS} --use-device-profiler --use-realtime-profiler \
              --writer-flush-on-pressure \
              --cross-program-dram-offset \
              --reader-dbuf-trid --reader-trid-in-flight ${TRID_IN_FLIGHT} \
              --reader-push-tiles 2 \
              --input-cb-depth-tiles ${IN_CB} --output-cb-depth-tiles ${OUT_CB} \
              --num-pages-per-core ${LATENCY_PAGES} --num-active-cores ${N}" \
  bash "${SCRIPT_DIR}/run_op_to_op_multi.sh" "${NUM_RUNS}" 2>&1 | tail -8

  PYTHON="${TT_METAL_HOME}/python_env/bin/python3"
  if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
  fi

  "${PYTHON}" - "${N}" "${PEAK}" "${PEAK_GBPS}" "${IN_CB}" "${OUT_CB}" \
    "${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch7_cores${N}" \
    "${CHART_CSV}" "${MIN_PROG_ID}" << 'PYEOF'
import sys, os, glob, statistics as st

ncores, peak, peak_gbps, in_cb, out_cb, base, chart_csv, min_prog = sys.argv[1:9]
runs = sorted(glob.glob(os.path.join(base, "run_*")))
op2op, dg_med, dg_issue = [], [], []

for r in runs:
    f = os.path.join(r, "profile_log_device_op_to_op_complete.csv")
    if not os.path.exists(f):
        continue
    import csv
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    rows = [row for row in rows if int(float(row.get("from_prog_id", 0))) >= int(min_prog)]
    if not rows:
        continue
    op2op.append(float(st.median([float(row["gap_us"]) for row in rows])))
    dg_vals = [float(row["dg_median_ns"]) for row in rows if row.get("dg_median_ns") not in ("", "nan")]
    if dg_vals:
        dg_med.append(st.median(dg_vals))
    iss = [float(row["dg_issue_ns"]) for row in rows if row.get("dg_issue_ns") not in ("", "nan")]
    if iss:
        dg_issue.append(st.median(iss))

def med(xs):
    return st.median(xs) if xs else float("nan")

op2op_m = med(op2op)
dg_m = med(dg_med)
dg_i = med(dg_issue)
cb_label = f"in={in_cb}/out={out_cb}"
with open(chart_csv, "a") as fh:
    fh.write(
        f"{ncores},{peak},{peak_gbps},{in_cb},{out_cb},"
        f"{op2op_m:.3f},{dg_m:.1f},{dg_i:.1f},{cb_label},grid_cb,flush_on_pressure,cross_program\n"
    )
print(f"  cores={ncores}  peak={peak} GB/s  op2op={op2op_m:.3f}us  dg_median={dg_m:.0f}ns")
PYEOF
done

echo
echo "Batch 7 chart: ${CHART_CSV}"
"${PYTHON:-python3}" "${SCRIPT_DIR}/rebuild_batch7_compare.py" || true
