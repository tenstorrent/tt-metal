#!/usr/bin/env bash
# Batch 7 Pareto: BW vs op-to-op-gap front at 80 and 110 cores, with each
# program in the trace touching a disjoint DRAM tile slice
# (--cross-program-dram-offset). Mirror of run_batch6_pareto.sh so we can
# diff batch7-vs-batch6 per CB pair and see if the ±15% swings observed in the
# batch7 main chart at 80c/110c are noise vs a real DRAM-pattern signal.
#
# For each (cores, in_cb, out_cb) in CONFIGS:
#   1. Measure peak BW at that CB (--buffer-tune-grid --buffer-tune-bw-only,
#      single-program so DRAM offset is moot for the tune phase).
#   2. Measure program-gap + dg_median at that CB with --cross-program-dram-offset
#      on, NUM_RUNS replays for variance reduction.
#
# Output:
#   generated/profiler/op_to_op_runs/batch7_pareto/pareto.csv
#   generated/profiler/op_to_op_runs/batch7_pareto/batch7_vs_batch6.csv
#
# Usage:
#   source python_env/bin/activate
#   ./run_batch7_pareto.sh

set -uo pipefail

NUM_RUNS="${NUM_RUNS:-5}"
LATENCY_PAGES="${LATENCY_PAGES:-16}"
COMPUTE_NOPS="${COMPUTE_NOPS:-2000}"
NUM_PROGRAMS="${NUM_PROGRAMS:-8}"
MIN_PROG_ID="${MIN_PROG_ID:-3}"
TRID_IN_FLIGHT="${TRID_IN_FLIGHT:-2}"
BUILD="${BUILD:-1}"

# Same Pareto grid as batch 6 so we can do per-CB diff.
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
if [[ -n "${CONFIGS:-}" ]]; then
  IFS=',' read -ra CONFIGS_ARR <<< "${CONFIGS}"
else
  CONFIGS_ARR=("${CONFIGS_DEFAULT[@]}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-${TT_METAL_HOME}}"
export TT_METAL_DEVICE_PROFILER=1
# dispatch-core profiling off (workers only); official op2op uses KERNEL zones, no DISP markers needed
# export TT_METAL_DEVICE_PROFILER_DISPATCH=1

PYTHON="${TT_METAL_HOME}/python_env/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

BUILD_DIR="${BUILD_DIR:-${TT_METAL_HOME}/build_Release}"
TEST_BIN="${BUILD_DIR}/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
OUT_DIR="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch7_pareto"
PARETO_CSV="${OUT_DIR}/pareto.csv"

mkdir -p "${OUT_DIR}"

if [[ "${BUILD}" != "0" ]]; then
  if command -v cmake >/dev/null 2>&1; then
    cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
  else
    /usr/local/bin/cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
  fi
fi

echo "cores,in_cb,out_cb,peak_bw_gbs,peak_bw_gbps,op2op_us_median,dg_median_ns,dg_issue_ns,cb_label,dram_offset" > "${PARETO_CSV}"

MIN_IN=$(( 2 * TRID_IN_FLIGHT ))

for cfg in "${CONFIGS_ARR[@]}"; do
  read -r N IN_CB OUT_CB <<< "${cfg}"
  if (( IN_CB < MIN_IN )); then
    echo "  bumping in_cb ${IN_CB} -> ${MIN_IN} (mode2 needs in >= ${MIN_IN})"
    IN_CB="${MIN_IN}"
  fi
  echo "================ batch7_pareto cores=${N} in=${IN_CB} out=${OUT_CB} ================"
  RUN_DIR="${OUT_DIR}/cores_${N}_in${IN_CB}_out${OUT_CB}"
  mkdir -p "${RUN_DIR}"
  BW_LOG="${RUN_DIR}/bw.log"

  # Phase 1: BW at this CB pair (single-program tune; offset has no effect)
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

  # Phase 2: latency at same CB, WITH cross-program DRAM offset
  CONFIG_LABEL="batch7p_c${N}_i${IN_CB}_o${OUT_CB}" \
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
  bash "${SCRIPT_DIR}/run_op_to_op_multi.sh" "${NUM_RUNS}" 2>&1 | tail -4

  "${PYTHON}" - "${N}" "${IN_CB}" "${OUT_CB}" "${PEAK}" "${PEAK_GBPS}" \
    "${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch7p_c${N}_i${IN_CB}_o${OUT_CB}" \
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
        f"{op2op_m:.3f},{dg_m:.1f},{dg_i:.1f},{cb_label},cross_program\n"
    )
print(f"  cores={ncores} {cb_label}  peak={peak} GB/s  gap={op2op_m:.3f}us  dg={dg_m:.0f}ns")
PYEOF
done

echo
echo "Pareto CSV: ${PARETO_CSV}"

# Per-core summary
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
    print(f"\n=== {cores} cores (cross-program DRAM offset) ===")
    print(f"  {'CB':<14} {'BW GB/s':>10} {'gap us':>10} {'dg ns':>8}")
    for r in points:
        print(f"  {r['cb_label']:<14} {float(r['peak_bw_gbs']):>10.2f} {float(r['op2op_us_median']):>10.3f} {float(r['dg_median_ns']):>8.0f}")
PYEOF

# Batch7-vs-batch6 diff (per cores,in_cb,out_cb)
BATCH6_CSV="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch6/pareto.csv"
DIFF_CSV="${OUT_DIR}/batch7_vs_batch6.csv"
"${PYTHON}" - "${BATCH6_CSV}" "${PARETO_CSV}" "${DIFF_CSV}" << 'PYEOF'
import csv, os, sys

b6_path, b7_path, out_path = sys.argv[1:4]

def load(p):
    if not os.path.isfile(p):
        return {}
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            key = (int(r["cores"]), int(r["in_cb"]), int(r["out_cb"]))
            out[key] = r
    return out

b6, b7 = load(b6_path), load(b7_path)
if not b7:
    print(f"No batch7 pareto data at {b7_path}")
    raise SystemExit(0)

def f(r, k, dflt="nan"):
    v = r.get(k, dflt)
    try:
        return float(v) if v not in ("", "nan", None) else float("nan")
    except (TypeError, ValueError):
        return float("nan")

fields = [
    "cores", "in_cb", "out_cb", "cb_label",
    "b6_peak_gbs", "b7_peak_gbs", "bw_delta_pct",
    "b6_op2op_us", "b7_op2op_us", "gap_delta_us", "gap_delta_pct",
    "b6_dg_ns", "b7_dg_ns", "dg_delta_ns",
]
with open(out_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for key in sorted(b7):
        r7 = b7[key]
        r6 = b6.get(key, {})
        p6, p7 = f(r6, "peak_bw_gbs"), f(r7, "peak_bw_gbs")
        o6, o7 = f(r6, "op2op_us_median"), f(r7, "op2op_us_median")
        d6, d7 = f(r6, "dg_median_ns"), f(r7, "dg_median_ns")
        w.writerow({
            "cores": key[0], "in_cb": key[1], "out_cb": key[2],
            "cb_label": r7.get("cb_label", ""),
            "b6_peak_gbs": f"{p6:.2f}" if p6 == p6 else "",
            "b7_peak_gbs": f"{p7:.2f}" if p7 == p7 else "",
            "bw_delta_pct": f"{(p7-p6)/p6*100:+.1f}" if p6 == p6 and p7 == p7 and p6 != 0 else "",
            "b6_op2op_us": f"{o6:.3f}" if o6 == o6 else "",
            "b7_op2op_us": f"{o7:.3f}" if o7 == o7 else "",
            "gap_delta_us": f"{o7-o6:+.3f}" if o6 == o6 and o7 == o7 else "",
            "gap_delta_pct": f"{(o7-o6)/o6*100:+.1f}" if o6 == o6 and o7 == o7 and o6 != 0 else "",
            "b6_dg_ns": f"{d6:.0f}" if d6 == d6 else "",
            "b7_dg_ns": f"{d7:.0f}" if d7 == d7 else "",
            "dg_delta_ns": f"{d7-d6:+.0f}" if d6 == d6 and d7 == d7 else "",
        })
print(f"\nWrote {out_path}\n")
with open(out_path) as fh:
    print(fh.read())
PYEOF
