#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Sweep trid_in_flight (reads per per-trid batch = N) with CB DEPTH HELD FIXED (in_cb=64),
# to isolate N from CB depth. Tests: (a) does larger N raise per-core BW (-> fewer cores to
# saturate)?  (b) does larger N defer math start (bigger first batch -> later first push)?
# good NoC config (default), row, nops=0 (read+write bound). in_cb=64 stays >= 2*N for N<=32.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_DEVICE_PROFILER=1
# dispatch-core profiling off (workers only); official op2op uses KERNEL zones, no DISP markers needed
# export TT_METAL_DEVICE_PROFILER_DISPATCH=1
BIN="${TT_METAL_HOME}/build_RelWithDebInfo/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
PY="${TT_METAL_HOME}/python_env/bin/python3"
DEC="${SCRIPT_DIR}/decompose_latency_bw.py"
DEV_CSV="${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv"
OUTDIR="${OUTDIR:-${TT_METAL_HOME}/generated/profiler/op_to_op_runs/stage_a}"
OUT="${OUTDIR}/trid_sweep.csv"
mkdir -p "${OUTDIR}"

IN_CB="${IN_CB:-64}"          # fixed CB depth (>= 2*max N) to decouple from N
OUT_CB="${OUT_CB:-8}"
PAGES="${PAGES:-1024}"
CORES_LIST="${CORES_LIST:-1 8 32}"
TRID_LIST="${TRID_LIST:-2 4 8 16 32}"

rm -f "${OUT}"
for C in ${CORES_LIST}; do
  for NF in ${TRID_LIST}; do
    rm -f "${DEV_CSV}"
    "${BIN}" --reader-dbuf-trid --reader-trid-in-flight "${NF}" --reader-push-tiles 2 \
      --input-cb-depth-tiles "${IN_CB}" --output-cb-depth-tiles "${OUT_CB}" --writer-end-barrier-mode 0 \
      --lean-compute --skip-output-validation \
      --num-pages-per-core "${PAGES}" --num-programs 2 --compute-nops 0 \
      --use-trace --trace-warmup-replays 1 --num-active-cores "${C}" --use-device-profiler \
      >/tmp/trid_run.log 2>&1
    if grep -q PASSED /tmp/trid_run.log; then
      "${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${C}" --csv-out "${OUT}" \
        --label "cores=${C};trid=${NF};in_cb=${IN_CB}" 2>/dev/null \
        | grep -E "agg_total_gbps|read_fill_head_us|official_op2op_min_us|m2m_bubble_us"
      echo "  cores=${C} trid=${NF} done"
    else echo "  cores=${C} trid=${NF} FAILED: $(grep -oE 'TT_FATAL: .*' /tmp/trid_run.log | head -1)"; fi
  done
done
echo "TRID_SWEEP_COMPLETE"
