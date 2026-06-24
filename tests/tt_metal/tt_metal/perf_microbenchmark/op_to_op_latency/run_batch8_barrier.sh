#!/usr/bin/env bash
# Batch 8: writer end-of-kernel barrier sweep — directly tests how much HW
# barrier-elimination would buy us ('s "optimizations around issuing
# barriers" proposal).
#
# Three modes per core count (everything else identical to batch 4):
#   end_barrier_mode=0: noc_async_write_barrier()  -- DEFAULT, wait for DRAM ACK
#   end_barrier_mode=1: noc_async_writes_flushed() -- only flush local L1
#   end_barrier_mode=2: nothing                    -- simulates HW giving it free
#
# CB sizes are taken from batch 4's grid pick per core count (apples-to-apples
# vs batch 4); only the barrier knob changes.
#
# Output:
#   generated/profiler/op_to_op_runs/batch8/chart_data.csv
#   generated/profiler/op_to_op_runs/batch8/batch8_vs_batch4.csv
#
# Usage:
#   source python_env/bin/activate
#   ./run_batch8_barrier.sh

set -uo pipefail

CORE_COUNTS="${CORE_COUNTS:-1 2 4 10 20 40 80 110}"
NUM_RUNS="${NUM_RUNS:-5}"
LATENCY_PAGES="${LATENCY_PAGES:-16}"
COMPUTE_NOPS="${COMPUTE_NOPS:-2000}"
NUM_PROGRAMS="${NUM_PROGRAMS:-8}"
MIN_PROG_ID="${MIN_PROG_ID:-3}"
TRID_IN_FLIGHT="${TRID_IN_FLIGHT:-2}"
BUILD="${BUILD:-1}"

# CB sizes per core-count, taken from batch4 grid picks.
# Format: cores in_cb out_cb
declare -A CB_FOR_CORES=(
  [1]="12 2"
  [2]="12 2"
  [4]="12 4"
  [10]="16 8"
  [20]="8 4"
  [40]="64 64"
  [80]="64 32"
  [110]="64 16"
)

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
OUT_DIR="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch8"
CHART_CSV="${OUT_DIR}/chart_data.csv"

mkdir -p "${OUT_DIR}"

if [[ "${BUILD}" != "0" ]]; then
  cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
fi

echo "num_cores,end_barrier_mode,barrier_label,in_cb,out_cb,op2op_us_median,dg_median_ns,dg_issue_ns,cb_label" > "${CHART_CSV}"

for N in ${CORE_COUNTS}; do
  read -r IN_CB OUT_CB <<< "${CB_FOR_CORES[$N]:-16 8}"
  for MODE in 0 1 2; do
    case "${MODE}" in
      0) LABEL="barrier";;
      1) LABEL="flushed";;
      2) LABEL="none";;
    esac
    echo "================ batch8 cores=${N} mode=${MODE}(${LABEL}) cb=${IN_CB}/${OUT_CB} ================"

    CONFIG_LABEL="batch8_c${N}_m${MODE}" \
    MIN_PROG_ID="${MIN_PROG_ID}" \
    TILES_PER_CORE="${LATENCY_PAGES}" \
    INPUT_CB_DEPTH="${IN_CB}" \
    READER_PUSH=2 \
    BUILD=0 \
    EXTRA_ARGS="--use-trace --trace-warmup-replays 2 --num-programs ${NUM_PROGRAMS} \
                --compute-nops ${COMPUTE_NOPS} --use-device-profiler --use-realtime-profiler \
                --writer-flush-on-pressure \
                --writer-end-barrier-mode ${MODE} \
                --reader-dbuf-trid --reader-trid-in-flight ${TRID_IN_FLIGHT} \
                --reader-push-tiles 2 \
                --input-cb-depth-tiles ${IN_CB} --output-cb-depth-tiles ${OUT_CB} \
                --num-pages-per-core ${LATENCY_PAGES} --num-active-cores ${N}" \
    bash "${SCRIPT_DIR}/run_op_to_op_multi.sh" "${NUM_RUNS}" 2>&1 | tail -4

    "${PYTHON}" - "${N}" "${MODE}" "${LABEL}" "${IN_CB}" "${OUT_CB}" \
      "${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch8_c${N}_m${MODE}" \
      "${CHART_CSV}" "${MIN_PROG_ID}" << 'PYEOF'
import csv, glob, os, statistics as st, sys

ncores, mode, label, in_cb, out_cb, base, out_csv, min_prog = sys.argv[1:9]
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
        f"{ncores},{mode},{label},{in_cb},{out_cb},"
        f"{op2op_m:.3f},{dg_m:.1f},{dg_i:.1f},{cb_label}\n"
    )
print(f"  cores={ncores} mode={mode}({label})  gap={op2op_m:.3f}us  dg={dg_m:.0f}ns")
PYEOF
  done
done

echo
echo "Batch 8 chart: ${CHART_CSV}"

# Build batch8_vs_batch4 delta table
BATCH4_CSV="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch4/chart_data.csv"
DIFF_CSV="${OUT_DIR}/batch8_vs_batch4.csv"
"${PYTHON}" - "${BATCH4_CSV}" "${CHART_CSV}" "${DIFF_CSV}" << 'PYEOF'
import csv, os, sys
from collections import defaultdict

b4_path, b8_path, out_path = sys.argv[1:4]

def load_b4(p):
    out = {}
    if not os.path.isfile(p):
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            out[int(r["num_cores"])] = r
    return out

def load_b8(p):
    out = defaultdict(dict)
    if not os.path.isfile(p):
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            out[int(r["num_cores"])][int(r["end_barrier_mode"])] = r
    return out

def fl(r, k):
    try:
        v = r.get(k, "")
        return float(v) if v not in ("", "nan", None) else float("nan")
    except (TypeError, ValueError, AttributeError):
        return float("nan")

b4 = load_b4(b4_path)
b8 = load_b8(b8_path)

fields = [
    "cores", "cb",
    "b4_gap_us", "b4_dg_ns",
    "b8m0_gap_us", "b8m0_dg_ns",
    "b8m1_gap_us", "b8m1_dg_ns", "m1_vs_m0_gap_us", "m1_vs_m0_gap_pct",
    "b8m2_gap_us", "b8m2_dg_ns", "m2_vs_m0_gap_us", "m2_vs_m0_gap_pct",
]
with open(out_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for cores in sorted(b8):
        modes = b8[cores]
        r0 = modes.get(0, {}); r1 = modes.get(1, {}); r2 = modes.get(2, {})
        rb4 = b4.get(cores, {})
        g0 = fl(r0, "op2op_us_median"); g1 = fl(r1, "op2op_us_median"); g2 = fl(r2, "op2op_us_median")
        d0 = fl(r0, "dg_median_ns"); d1 = fl(r1, "dg_median_ns"); d2 = fl(r2, "dg_median_ns")
        gb4 = fl(rb4, "op2op_us_median"); db4 = fl(rb4, "dg_median_ns")
        cb = r0.get("cb_label", "")
        w.writerow({
            "cores": cores, "cb": cb,
            "b4_gap_us": f"{gb4:.3f}" if gb4 == gb4 else "",
            "b4_dg_ns": f"{db4:.0f}" if db4 == db4 else "",
            "b8m0_gap_us": f"{g0:.3f}" if g0 == g0 else "",
            "b8m0_dg_ns": f"{d0:.0f}" if d0 == d0 else "",
            "b8m1_gap_us": f"{g1:.3f}" if g1 == g1 else "",
            "b8m1_dg_ns": f"{d1:.0f}" if d1 == d1 else "",
            "m1_vs_m0_gap_us": f"{g1-g0:+.3f}" if g0 == g0 and g1 == g1 else "",
            "m1_vs_m0_gap_pct": f"{(g1-g0)/g0*100:+.1f}" if g0 == g0 and g1 == g1 and g0 != 0 else "",
            "b8m2_gap_us": f"{g2:.3f}" if g2 == g2 else "",
            "b8m2_dg_ns": f"{d2:.0f}" if d2 == d2 else "",
            "m2_vs_m0_gap_us": f"{g2-g0:+.3f}" if g0 == g0 and g2 == g2 else "",
            "m2_vs_m0_gap_pct": f"{(g2-g0)/g0*100:+.1f}" if g0 == g0 and g2 == g2 and g0 != 0 else "",
        })

print(f"\nWrote {out_path}\n")
with open(out_path) as fh:
    print(fh.read())
PYEOF
