#!/usr/bin/env bash
# Batch 8b: writer end-of-kernel barrier sweep, but pinned to BATCH 1 CBs
# (the latency-optimal / baseline CBs) instead of batch 4's BW-optimal CBs.
# Apples-to-apples vs batch 1, which is the right comparison for a pure
# latency experiment.
#
# Three modes per core count (everything else identical to batch 4/8):
#   end_barrier_mode=0: noc_async_write_barrier()  -- DEFAULT, wait for DRAM ACK
#   end_barrier_mode=1: noc_async_writes_flushed() -- only flush local L1
#   end_barrier_mode=2: nothing                    -- simulates HW giving it free
#
# Output:
#   generated/profiler/op_to_op_runs/batch8b/chart_data.csv
#   generated/profiler/op_to_op_runs/batch8b/batch8b_vs_batch1.csv

set -uo pipefail

CORE_COUNTS="${CORE_COUNTS:-1 2 4 10 20 40 80 110}"
NUM_RUNS="${NUM_RUNS:-5}"
LATENCY_PAGES="${LATENCY_PAGES:-16}"
COMPUTE_NOPS="${COMPUTE_NOPS:-2000}"
NUM_PROGRAMS="${NUM_PROGRAMS:-8}"
MIN_PROG_ID="${MIN_PROG_ID:-3}"
TRID_IN_FLIGHT="${TRID_IN_FLIGHT:-2}"
BUILD="${BUILD:-0}"

# CB sizes per core-count, taken from batch 1 picks (latency-optimal).
# Format: cores in_cb out_cb
declare -A CB_FOR_CORES=(
  [1]="4 2"
  [2]="4 2"
  [4]="6 2"
  [10]="4 4"
  [20]="4 2"
  [40]="16 16"
  [80]="12 16"
  [110]="4 2"
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
OUT_DIR="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch8b"
CHART_CSV="${OUT_DIR}/chart_data.csv"

mkdir -p "${OUT_DIR}"

if [[ "${BUILD}" != "0" ]]; then
  cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
fi

echo "num_cores,end_barrier_mode,barrier_label,in_cb,out_cb,op2op_us_median,dg_median_ns,dg_issue_ns,cb_label" > "${CHART_CSV}"

for N in ${CORE_COUNTS}; do
  read -r IN_CB OUT_CB <<< "${CB_FOR_CORES[$N]:-4 2}"
  for MODE in 0 1 2; do
    case "${MODE}" in
      0) LABEL="barrier";;
      1) LABEL="flushed";;
      2) LABEL="none";;
    esac
    echo "================ batch8b cores=${N} mode=${MODE}(${LABEL}) cb=${IN_CB}/${OUT_CB} ================"

    CONFIG_LABEL="batch8b_c${N}_m${MODE}" \
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
      "${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch8b_c${N}_m${MODE}" \
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
echo "Batch 8b chart: ${CHART_CSV}"

# Build batch8b_vs_batch1 delta table (apples-to-apples vs B1, the latency baseline)
BATCH1_CANDIDATES=(
  "${TT_METAL_HOME}/generated/profiler/op_to_op_runs/chart_sweep/dg_v4/chart_data.csv"
  "${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch1/chart_data.csv"
  "${TT_METAL_HOME}/generated/profiler/op_to_op_runs/paul_chart_sweep/chart_data.csv"
)
B1_REF=""
for cand in "${BATCH1_CANDIDATES[@]}"; do
  if [[ -f "${cand}" ]]; then B1_REF="${cand}"; break; fi
done

DIFF_CSV="${OUT_DIR}/batch8b_vs_batch1.csv"
"${PYTHON}" - "${B1_REF}" "${CHART_CSV}" "${DIFF_CSV}" << 'PYEOF'
import csv, os, sys
from collections import defaultdict

b1_path, b8b_path, out_path = sys.argv[1:4]

def load_b1(p):
    out = {}
    if not p or not os.path.isfile(p):
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            # batch1 chart_data.csv uses "num_cores" / "op2op_us_median" / "dg_median_ns"
            key = r.get("num_cores") or r.get("cores")
            if key is None: continue
            try:
                out[int(float(key))] = r
            except ValueError:
                continue
    return out

def load_b8b(p):
    out = defaultdict(dict)
    if not os.path.isfile(p):
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            out[int(r["num_cores"])][int(r["end_barrier_mode"])] = r
    return out

def fl(r, *keys):
    for k in keys:
        v = r.get(k, "")
        if v in ("", "nan", None): continue
        try: return float(v)
        except (TypeError, ValueError): continue
    return float("nan")

b1 = load_b1(b1_path)
b8b = load_b8b(b8b_path)

fields = [
    "cores", "cb",
    "b1_gap_us", "b1_dg_ns",
    "b8bm0_gap_us", "b8bm0_dg_ns",
    "b8bm1_gap_us", "b8bm1_dg_ns", "m1_vs_m0_gap_us", "m1_vs_m0_gap_pct",
    "b8bm2_gap_us", "b8bm2_dg_ns", "m2_vs_m0_gap_us", "m2_vs_m0_gap_pct",
]
with open(out_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for cores in sorted(b8b):
        modes = b8b[cores]
        r0 = modes.get(0, {}); r1 = modes.get(1, {}); r2 = modes.get(2, {})
        rb1 = b1.get(cores, {})
        g0 = fl(r0, "op2op_us_median"); g1 = fl(r1, "op2op_us_median"); g2 = fl(r2, "op2op_us_median")
        d0 = fl(r0, "dg_median_ns"); d1 = fl(r1, "dg_median_ns"); d2 = fl(r2, "dg_median_ns")
        gb1 = fl(rb1, "op2op_us_median", "gap_us")
        db1 = fl(rb1, "dg_median_ns", "chip_done_to_go_ns_median")
        cb = r0.get("cb_label", "")
        w.writerow({
            "cores": cores, "cb": cb,
            "b1_gap_us": f"{gb1:.3f}" if gb1 == gb1 else "",
            "b1_dg_ns": f"{db1:.0f}" if db1 == db1 else "",
            "b8bm0_gap_us": f"{g0:.3f}" if g0 == g0 else "",
            "b8bm0_dg_ns": f"{d0:.0f}" if d0 == d0 else "",
            "b8bm1_gap_us": f"{g1:.3f}" if g1 == g1 else "",
            "b8bm1_dg_ns": f"{d1:.0f}" if d1 == d1 else "",
            "m1_vs_m0_gap_us": f"{g1-g0:+.3f}" if g0 == g0 and g1 == g1 else "",
            "m1_vs_m0_gap_pct": f"{(g1-g0)/g0*100:+.1f}" if g0 == g0 and g1 == g1 and g0 != 0 else "",
            "b8bm2_gap_us": f"{g2:.3f}" if g2 == g2 else "",
            "b8bm2_dg_ns": f"{d2:.0f}" if d2 == d2 else "",
            "m2_vs_m0_gap_us": f"{g2-g0:+.3f}" if g0 == g0 and g2 == g2 else "",
            "m2_vs_m0_gap_pct": f"{(g2-g0)/g0*100:+.1f}" if g0 == g0 and g2 == g2 and g0 != 0 else "",
        })

print(f"\nWrote {out_path}\n")
with open(out_path) as fh:
    print(fh.read())
PYEOF
