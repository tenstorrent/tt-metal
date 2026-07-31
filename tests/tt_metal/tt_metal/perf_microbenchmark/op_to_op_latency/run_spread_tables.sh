#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Two matched-parameter tables for one spreadsheet tab (good NoC config, row layout):
#   Table 1 spread_vs_cores.csv : BW + done-spread vs core count (nops=0). Spread grows.
#   Table 2 spread_vs_nops.csv  : writer spread vs NOPs at fixed cores. Spread drops back.
# Identical pages/core, input/output CB across both so they agree at (cores=FIX, nops=0).
# writer_spread_pct = spread / per-core op duration (work-size independent).
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
mkdir -p "${OUTDIR}"

PAGES="${PAGES:-512}"          # fixed per-core work (so spread is comparable across runs)
IN_CB="${IN_CB:-32}"
OUT_CB="${OUT_CB:-16}"         # held deep+fixed so output buffering isn't a confound
N="${N:-8}"
CORES_LIST="${CORES_LIST:-1 2 4 8 16 24 32 40 48 56}"
NOPS_LIST="${NOPS_LIST:-0 64 128 256 512 1024 2048 4096}"
FIX_CORES="${FIX_CORES:-56}"

run() { # cores nops
  rm -f "${DEV_CSV}"
  "${BIN}" --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 \
    --input-cb-depth-tiles "${IN_CB}" --output-cb-depth-tiles "${OUT_CB}" --writer-end-barrier-mode 0 \
    --lean-compute --skip-output-validation \
    --num-pages-per-core "${PAGES}" --num-programs 2 --compute-nops "$2" \
    --use-trace --trace-warmup-replays 1 --num-active-cores "$1" --use-device-profiler >/tmp/st_run.log 2>&1
  grep -q PASSED /tmp/st_run.log
}

T1="${OUTDIR}/spread_vs_cores.csv"; rm -f "${T1}"
echo "==== Table 1: BW + done-spread vs cores (nops=0, row) ===="
for C in ${CORES_LIST}; do
  if run "${C}" 0; then
    "${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${C}" --csv-out "${T1}" \
      --label "cores=${C};nops=0" 2>/dev/null \
      | grep -E "agg_total_gbps|writer_done_spread_us|writer_spread_pct|starvation_ratio"
    echo "  T1 cores=${C} done"
  else echo "  T1 cores=${C} FAILED: $(grep -oE 'TT_FATAL: .*' /tmp/st_run.log | head -1)"; fi
done

T2="${OUTDIR}/spread_vs_nops.csv"; rm -f "${T2}"
echo "==== Table 2: writer spread vs NOPs (cores=${FIX_CORES}, row) ===="
for NQ in ${NOPS_LIST}; do
  if run "${FIX_CORES}" "${NQ}"; then
    "${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${FIX_CORES}" --csv-out "${T2}" \
      --label "cores=${FIX_CORES};nops=${NQ}" 2>/dev/null \
      | grep -E "agg_total_gbps|writer_done_spread_us|writer_spread_pct|starvation_ratio"
    echo "  T2 nops=${NQ} done"
  else echo "  T2 nops=${NQ} FAILED"; fi
done
echo "SPREAD_TABLES_COMPLETE"
