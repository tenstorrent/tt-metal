#!/usr/bin/env bash
# Batch 6: BW vs op-to-op-gap Pareto front at 80 and 110 cores.
#
# Goal: show the BW/latency tradeoff cares about — at the high-core counts
# where batch 1 (small CB, low gap) and batch 4/5 (big CB, high BW) split apart.
#
# For each (cores, in_cb, out_cb) in CONFIGS:
#   1. Measure peak BW (no profiler, multi-program) at that CB.
#   2. Measure program-gap + dg_median (profiler) at that CB.
#
# Output:
#   generated/profiler/op_to_op_runs/batch6/pareto.csv
#   columns: cores, in_cb, out_cb, peak_gbs, op2op_us_median, dg_median_ns, cb_label
#
# Usage:
#   source python_env/bin/activate
#   ./run_batch6_pareto.sh

set -uo pipefail

NUM_RUNS="${NUM_RUNS:-3}"
LATENCY_PAGES="${LATENCY_PAGES:-16}"
COMPUTE_NOPS="${COMPUTE_NOPS:-2000}"
NUM_PROGRAMS="${NUM_PROGRAMS:-8}"
MIN_PROG_ID="${MIN_PROG_ID:-3}"
TRID_IN_FLIGHT="${TRID_IN_FLIGHT:-2}"
BUILD="${BUILD:-1}"

# Pareto sweep — small -> big CBs, spanning batch-1 and batch-5 picks.
# Format: "cores in_cb out_cb"
CONFIGS_DEFAULT=(
  "80 4 2"
  "80 8 4"
  "80 12 8"
  "80 16 16"
  "80 32 32"
  "80 48 64"
  "80 64 64"
  "110 4 2"
  "110 8 4"
  "110 12 8"
  "110 16 16"
  "110 32 32"
  "110 48 64"
  "110 64 64"
)
# Comma-separated CONFIGS env var overrides default (e.g. CONFIGS="80 4 2,80 64 64").
if [[ -n "${CONFIGS:-}" ]]; then
  IFS=',' read -ra CONFIGS_ARR <<< "${CONFIGS}"
else
  CONFIGS_ARR=("${CONFIGS_DEFAULT[@]}")
fi

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
OUT_DIR="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch6"
PARETO_CSV="${OUT_DIR}/pareto.csv"

mkdir -p "${OUT_DIR}"

if [[ "${BUILD}" != "0" ]]; then
  if command -v cmake >/dev/null 2>&1; then
    cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
  else
    /usr/local/bin/cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
  fi
fi

echo "cores,in_cb,out_cb,peak_bw_gbs,peak_bw_gbps,op2op_us_median,dg_median_ns,dg_issue_ns,cb_label" > "${PARETO_CSV}"

MIN_IN=$(( 2 * TRID_IN_FLIGHT ))

for cfg in "${CONFIGS_ARR[@]}"; do
  read -r N IN_CB OUT_CB <<< "${cfg}"
  if (( IN_CB < MIN_IN )); then
    echo "  skip cores=${N} in=${IN_CB} (mode2 needs in >= ${MIN_IN}); bumping to ${MIN_IN}"
    IN_CB="${MIN_IN}"
  fi
  echo "================ batch6 cores=${N} in=${IN_CB} out=${OUT_CB} ================"
  RUN_DIR="${OUT_DIR}/cores_${N}_in${IN_CB}_out${OUT_CB}"
  mkdir -p "${RUN_DIR}"
  BW_LOG="${RUN_DIR}/bw.log"

  # Phase 1: BW via buffer-tune-bw-only at this single CB pair
  # (buffer-tune logs BUFFER_TUNE rows with dram_pipeline_gbps we can grep)
  rm -f "${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv"
  "${TEST_BIN}" \
    --buffer-tune \
    --buffer-tune-grid \
    --buffer-tune-bw-only \
    --buffer-tune-input-depths "${IN_CB}" \
    --buffer-tune-output-depths "${OUT_CB}" \
    --buffer-tune-pages-per-core 256 \
    --buffer-tune-bw-tolerance-pct 2 \
    --writer-flush-on-pressure \
    --reader-dbuf-trid --reader-trid-in-flight "${TRID_IN_FLIGHT}" \
    --reader-push-tiles 2 \
    --compute-nops 0 \
    --num-programs 1 \
    --num-active-cores "${N}" \
    2>&1 | tee "${BW_LOG}" >/dev/null

  PEAK=$(grep 'BUFFER_TUNE,phase=cb_grid' "${BW_LOG}" | tail -1 \
    | sed -n 's/.*dram_pipeline_gbps=\([0-9.]*\).*/\1/p')
  PEAK="${PEAK:-0}"
  PEAK_GBPS=$(awk "BEGIN {printf \"%.1f\", ${PEAK} * 8}")

  # Phase 2: latency at same CB
  CONFIG_LABEL="batch6_c${N}_i${IN_CB}_o${OUT_CB}" \
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
  bash "${SCRIPT_DIR}/run_op_to_op_multi.sh" "${NUM_RUNS}" 2>&1 | tail -4

  "${PYTHON}" - "${N}" "${IN_CB}" "${OUT_CB}" "${PEAK}" "${PEAK_GBPS}" \
    "${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch6_c${N}_i${IN_CB}_o${OUT_CB}" \
    "${PARETO_CSV}" "${MIN_PROG_ID}" << 'PYEOF'
import csv, glob, os, statistics as st, sys

ncores, in_cb, out_cb, peak, peak_gbps, base, out_csv, min_prog = sys.argv[1:9]
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
with open(out_csv, "a") as fh:
    fh.write(
        f"{ncores},{in_cb},{out_cb},{peak},{peak_gbps},"
        f"{op2op_m:.3f},{dg_m:.1f},{dg_i:.1f},{cb_label}\n"
    )
print(f"  cores={ncores} {cb_label}  peak={peak} GB/s  gap={op2op_m:.3f}us  dg={dg_m:.0f}ns")
PYEOF
done

echo
echo "Pareto CSV: ${PARETO_CSV}"
"${PYTHON}" - "${PARETO_CSV}" << 'PYEOF'
import csv, sys
from collections import defaultdict

with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))

by_cores = defaultdict(list)
for r in rows:
    by_cores[int(r["cores"])].append(r)

for cores in sorted(by_cores):
    points = sorted(by_cores[cores], key=lambda r: float(r["peak_bw_gbs"]))
    print(f"\n=== {cores} cores ===")
    print(f"  {'CB':<14} {'BW GB/s':>10} {'gap us':>10} {'dg ns':>8}")
    for r in points:
        print(f"  {r['cb_label']:<14} {float(r['peak_bw_gbs']):>10.2f} {float(r['op2op_us_median']):>10.3f} {float(r['dg_median_ns']):>8.0f}")
PYEOF
