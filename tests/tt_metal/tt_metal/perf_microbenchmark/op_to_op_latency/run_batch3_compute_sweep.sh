#!/usr/bin/env bash
# Batch 3: compute_nops sweep at fixed CB configs (isolate compute vs batch 1/2).
#
# Fixed CB per core count (from batch-2 zero-compute picks; held constant across all NOPs).
#   cores=1   in=12  out=2
#   cores=10  in=12  out=8
#   cores=40  in=64  out=64
#   cores=80  in=64  out=64
#   cores=110 in=12  out=64
#
# Usage:
#   source python_env/bin/activate
#   export TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_DISPATCH=1
#   ./run_batch3_compute_sweep.sh
#
# Env:
#   CORE_COUNTS        default "1 10 40 80 110"
#   COMPUTE_NOPS_LIST  default "0 100 500 2000 5000"
#   NUM_RUNS           default 3
#   BUILD              default 1

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME
export TT_METAL_DEVICE_PROFILER=1
# dispatch-core profiling off (workers only); official op2op uses KERNEL zones, no DISP markers needed
# export TT_METAL_DEVICE_PROFILER_DISPATCH=1

# Prefer project venv (system /opt/venv pandas is broken).
if [[ -x "${TT_METAL_HOME}/python_env/bin/python3" ]]; then
  export PATH="${TT_METAL_HOME}/python_env/bin:${PATH}"
fi

CORE_COUNTS="${CORE_COUNTS:-1 10 40 80 110}"
COMPUTE_NOPS_LIST="${COMPUTE_NOPS_LIST:-0 100 500 2000 5000}"
NUM_RUNS="${NUM_RUNS:-3}"
NUM_PROGRAMS="${NUM_PROGRAMS:-8}"
MIN_PROG_ID="${MIN_PROG_ID:-3}"
LATENCY_PAGES="${LATENCY_PAGES:-16}"
TRID_IN_FLIGHT="${TRID_IN_FLIGHT:-2}"
BUILD="${BUILD:-1}"

OUT_DIR="${TT_METAL_HOME}/generated/profiler/op_to_op_runs/batch3"
mkdir -p "${OUT_DIR}"

cb_in()  { case "$1" in 1) echo 12;; 10) echo 12;; 40) echo 64;; 80) echo 64;; 110) echo 12;; *) echo 12;; esac; }
cb_out() { case "$1" in 1) echo 2;; 10) echo 8;; 40) echo 64;; 80) echo 64;; 110) echo 64;; *) echo 2;; esac; }

echo "Batch 3 compute sweep: cores=[${CORE_COUNTS}] nops=[${COMPUTE_NOPS_LIST}] runs=${NUM_RUNS}"
echo "Output: ${OUT_DIR}/batch3_compute_sweep.csv"
echo | tee "${OUT_DIR}/batch3_sweep.log"

for N in ${CORE_COUNTS}; do
  IN_CB="$(cb_in "${N}")"
  OUT_CB="$(cb_out "${N}")"
  for NOPS in ${COMPUTE_NOPS_LIST}; do
    echo "======== cores=${N} compute_nops=${NOPS} in=${IN_CB} out=${OUT_CB} ========" | tee -a "${OUT_DIR}/batch3_sweep.log"
    CONFIG_LABEL="batch3_cores${N}_nops${NOPS}_n2" \
    MIN_PROG_ID="${MIN_PROG_ID}" \
    TILES_PER_CORE="${LATENCY_PAGES}" \
    INPUT_CB_DEPTH="${IN_CB}" \
    READER_PUSH=2 \
    BUILD="${BUILD}" \
    EXTRA_ARGS="--use-trace --trace-warmup-replays 2 --num-programs ${NUM_PROGRAMS} \
                --compute-nops ${NOPS} --use-device-profiler --use-realtime-profiler \
                --reader-push-tiles 2 --reader-dbuf-trid --reader-trid-in-flight ${TRID_IN_FLIGHT} \
                --input-cb-depth-tiles ${IN_CB} --output-cb-depth-tiles ${OUT_CB} \
                --num-pages-per-core ${LATENCY_PAGES} --num-active-cores ${N}" \
    bash "${SCRIPT_DIR}/run_op_to_op_multi.sh" "${NUM_RUNS}" 2>&1 | tee -a "${OUT_DIR}/batch3_sweep.log" | tail -6
    BUILD=0
  done
done

python3 "${SCRIPT_DIR}/rebuild_batch3_results.py" 2>&1 | tee -a "${OUT_DIR}/batch3_sweep.log"

echo
echo "================ BATCH 3 DONE ================"
echo "CSV: ${OUT_DIR}/batch3_compute_sweep.csv"
