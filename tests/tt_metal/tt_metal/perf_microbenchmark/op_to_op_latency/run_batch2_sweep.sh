#!/usr/bin/env bash
# batch-2 sweep (post-chart follow-ups):
#   1. Read-only BW sanity vs NoC table (strip writer DRAM writes)
#   2. Zero-compute chart (compute_nops=0) with denser core counts + larger CB sweep
#   3. Aggregates into batch2_summary.csv for sharing
#
# Env overrides:
#   READONLY_CORES   - default "1 10 20 40 80 110"
#   ZERO_CORES       - default "1 2 4 8 10 16 20 32 40 56 64 80 96 110"
#   INPUT_DEPTHS     - default "8,12,16,24,32,48,64" (mode2 min depth 8)
#   OUTPUT_DEPTHS    - default "2,4,8,16,32,64"
#   BW_PAGES         - default 256
#   NUM_RUNS         - default 3
#   RUN_READONLY     - default 1
#   RUN_ZERO_CHART   - default 1
#   BUILD            - default 1

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME
export TT_METAL_DEVICE_PROFILER=1
# dispatch-core profiling off (workers only); official op2op uses KERNEL zones, no DISP markers needed
# export TT_METAL_DEVICE_PROFILER_DISPATCH=1

BUILD_DIR="${BUILD_DIR:-${TT_METAL_HOME}/build_Release}"
TEST_BIN="${BUILD_DIR}/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
OUT_DIR="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch2"
SUMMARY="${OUT_DIR}/batch2_summary.csv"
READONLY_CSV="${OUT_DIR}/read_only_bw.csv"
ZERO_CHART="${OUT_DIR}/zero_compute_chart_data.csv"

READONLY_CORES="${READONLY_CORES:-1 10 20 40 80 110}"
ZERO_CORES="${ZERO_CORES:-1 2 4 8 10 16 20 32 40 56 64 80 96 110}"
INPUT_DEPTHS="${INPUT_DEPTHS:-8,12,16,24,32,48,64}"
OUTPUT_DEPTHS="${OUTPUT_DEPTHS:-2,4,8,16,32,64}"
BW_PAGES="${BW_PAGES:-256}"
NUM_RUNS="${NUM_RUNS:-3}"
RUN_READONLY="${RUN_READONLY:-1}"
RUN_ZERO_CHART="${RUN_ZERO_CHART:-1}"
TRID_IN_FLIGHT="${TRID_IN_FLIGHT:-2}"

mkdir -p "${OUT_DIR}"

if [[ "${BUILD:-1}" != "0" ]]; then
  if command -v cmake >/dev/null 2>&1; then
    cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
  else
    /usr/local/bin/cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j
  fi
fi

echo "section,cores,peak_bw_gbs,peak_bw_gbps,per_core_gbps,smallest_in_cb,smallest_out_cb,pack_to_unpack_us,dg_median_ns,dg_last_ns,cb_label,notes" > "${SUMMARY}"

# ---------------- Part 1: read-only BW ----------------
if [[ "${RUN_READONLY}" == "1" ]]; then
  echo "num_cores,peak_bw_gbs,peak_bw_gbps,per_core_gbps,smallest_in_cb,smallest_out_cb" > "${READONLY_CSV}"
  for N in ${READONLY_CORES}; do
    echo "======== read-only BW cores=${N} ========"
    LOG="${OUT_DIR}/read_only_cores_${N}.log"
    rm -f "${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv"
    "${TEST_BIN}" \
      --read-only \
      --buffer-tune \
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

    PEAK=$(grep 'buffer_tune: peak_dram_pipeline_gbps=' "${LOG}" | tail -1 | sed -n 's/.*peak_dram_pipeline_gbps=\([0-9.]*\).*/\1/p')
    IN_CB=$(grep 'smallest_input_cb_depth_at_peak=' "${LOG}" | tail -1 | sed -n 's/.*smallest_input_cb_depth_at_peak=\([0-9][0-9]*\).*/\1/p')
    OUT_CB=$(grep 'smallest_output_cb_depth_at_peak=' "${LOG}" | tail -1 | sed -n 's/.*smallest_output_cb_depth_at_peak=\([0-9][0-9]*\).*/\1/p')
    PEAK="${PEAK:-0}"
    IN_CB="${IN_CB:-0}"
    OUT_CB="${OUT_CB:-0}"

    python3 - "${N}" "${PEAK}" "${IN_CB}" "${OUT_CB}" "${READONLY_CSV}" "${SUMMARY}" << 'PY'
import sys
ncores, peak, in_cb, out_cb, ro_csv, summary = sys.argv[1:]
peak = float(peak)
nc = int(ncores)
per_core_gbps = (peak * 8.0 / nc) if nc > 0 else 0.0
with open(ro_csv, "a") as f:
    f.write(f"{ncores},{peak:.4f},{peak*8:.4f},{per_core_gbps:.4f},{in_cb},{out_cb}\n")
with open(summary, "a") as f:
    f.write(f"read_only_bw,{ncores},{peak:.4f},{peak*8:.4f},{per_core_gbps:.4f},"
            f"{in_cb},{out_cb},,,,in={in_cb}/out={out_cb},"
            f"read-only pipeline; BW=GB/s in logs x8=Gbps\n")
print(f"  read-only cores={ncores}: {peak:.2f} GB/s = {peak*8:.2f} Gbps aggregate, "
      f"{per_core_gbps:.2f} Gbps/core")
PY
  done
fi

# ---------------- Part 2: zero-compute chart sweep ----------------
if [[ "${RUN_ZERO_CHART}" == "1" ]]; then
  echo "======== zero-compute chart sweep ========"
  CORE_COUNTS="${ZERO_CORES}" \
  NUM_RUNS="${NUM_RUNS}" \
  INPUT_DEPTHS="${INPUT_DEPTHS}" \
  OUTPUT_DEPTHS="${OUTPUT_DEPTHS}" \
  BW_PAGES="${BW_PAGES}" \
  COMPUTE_NOPS=0 \
  CHART_TAG=batch2_zero_compute \
  TRID_IN_FLIGHT="${TRID_IN_FLIGHT}" \
  USE_DBUF_TRID=1 \
  BUILD=0 \
  bash "${SCRIPT_DIR}/run_op_to_op_chart_sweep.sh" 2>&1 | tee "${OUT_DIR}/zero_compute_sweep.log"

  ZERO_SRC="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/chart_sweep/batch2_zero_compute/chart_data.csv"
  if [[ -f "${ZERO_SRC}" ]]; then
    cp "${ZERO_SRC}" "${ZERO_CHART}"
    python3 - "${ZERO_SRC}" "${SUMMARY}" << 'PY'
import sys
import pandas as pd

src, summary = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
for _, r in df.iterrows():
    nc = int(r["num_cores"])
    peak = float(r["peak_input_gbps"])
    per_core = peak * 8.0 / nc if nc else 0.0
    cb = r.get("cb_label", "")
    with open(summary, "a") as f:
        f.write(
            f"zero_compute,{nc},{peak:.4f},{peak*8:.4f},{per_core:.4f},"
            f"{r['smallest_input_cb']},{r['smallest_output_cb']},"
            f"{r['op2op_us_median']:.3f},{r['dg_median_ns']:.1f},{r['dg_last_ns']:.1f},"
            f"{cb},compute_nops=0; gap=pack->unpack not math-to-math\n"
        )
print(df.to_string(index=False))
PY
  fi
fi

echo
echo "================ BATCH 2 DONE ================"
echo "Summary:     ${SUMMARY}"
echo "Read-only:   ${READONLY_CSV}"
echo "Zero chart:  ${ZERO_CHART}"
echo
column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
