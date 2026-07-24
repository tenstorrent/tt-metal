#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Canonical sweep 2/6: aggregate BW + done-skew + starvation vs core count.
# Full read+write op by default; set READ_ONLY=1 for the read-only variant (writer just pops
# the CB, no DRAM writes -> isolates read BW and read-side completion skew).
# Good NoCs (reader=NOC0, writer=NOC1), row-major fill, lean compute, trace warmup.
# Per core count picks the input-CB depth maximizing agg_read (read-only) / agg_total (full-op).
# Records: cores,in_cb,agg_read_gbps,agg_write_gbps,agg_total_gbps,
#          reader_done_spread_us,writer_done_spread_us,starvation_ratio.
#
# Env: READ_ONLY (0/1), CORES, INPUT_DEPTHS, N (trid), PAGES, OUT, OUTDIR.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)}"
export TT_METAL_DEVICE_PROFILER=1
cd "${TT_METAL_HOME}" || exit 1  # binary detects root from cwd, not env -> must run from repo root
BIN="${TT_METAL_HOME}/build_RelWithDebInfo/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
PY="${TT_METAL_HOME}/python_env/bin/python3"
DEC="${SCRIPT_DIR}/decompose_latency_bw.py"
DEV_CSV="${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv"
OUTDIR="${OUTDIR:-${TT_METAL_HOME}/generated/profiler/op_to_op_runs/stage_a}"
READ_ONLY="${READ_ONLY:-0}"
DEFAULT_OUT="bw_vs_cores.csv"; [ "${READ_ONLY}" = "1" ] && DEFAULT_OUT="bw_vs_cores_readonly.csv"
OUT="${OUT:-${OUTDIR}/${DEFAULT_OUT}}"
mkdir -p "${OUTDIR}"

INPUT_DEPTHS="${INPUT_DEPTHS:-16 32}"
N="${N:-8}"
CORES="${CORES:-1 2 4 8 16 24 32 40 48 56}"
# Fixed-TOTAL (data-parallel) work model, matching run_stage_a_sweep.sh: a real op shards a
# fixed-size tensor across the grid, so pages_per_core = TOTAL_PAGES / cores (recomputed per core
# count below). This keeps execution duration realistic instead of weak-scaling total work with
# cores. NOTE: at high core counts the small shard pays a ~5% pipeline fill/drain tax on aggregate
# BW (56c: ~205 vs ~215 GB/s at a large shard) -- that IS the realistic sharded-op number.
# Set PAGES=<n> to force the legacy fixed-per-core model instead.
TOTAL_PAGES="${TOTAL_PAGES:-3584}"
MIN_PAGES_PER_CORE="${MIN_PAGES_PER_CORE:-8}"
PAGES_OVERRIDE="${PAGES:-}"
# read-only skips output validation natively; full-op uses dummy strided reads -> skip it too.
# Rank in-CB tuning on overall_bw_gbps (total bytes / kernel envelope) -- the honest end-to-end
# throughput incl. the straggler tail -- NOT the union-span agg_* (steady/peak rate).
if [ "${READ_ONLY}" = "1" ]; then MODE_FLAGS="--read-only"; DIRN=1; else MODE_FLAGS="--skip-output-validation"; DIRN=2; fi
RANK="overall_bw_gbps"

echo "cores,pages,in_cb,overall_bw_gbps,agg_read_gbps,agg_write_gbps,agg_total_gbps,kernel_envelope_us,reader_done_spread_us,writer_done_spread_us,starvation_ratio" > "${OUT}"

for cores in ${CORES}; do
  if [ -n "${PAGES_OVERRIDE}" ]; then PAGES="${PAGES_OVERRIDE}"; else
    PAGES=$(( TOTAL_PAGES / cores )); (( PAGES < MIN_PAGES_PER_CORE )) && PAGES=${MIN_PAGES_PER_CORE}; fi
  best=0; best_line=""
  for d in ${INPUT_DEPTHS}; do
    rm -f "${DEV_CSV}"
    "${BIN}" ${MODE_FLAGS} --lean-compute --reader-noc 0 --writer-noc 1 \
      --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 \
      --input-cb-depth-tiles "${d}" --output-cb-depth-tiles 2 \
      --num-pages-per-core "${PAGES}" --num-programs 2 --compute-nops 0 \
      --use-trace --trace-warmup-replays 1 --num-active-cores "${cores}" --use-device-profiler \
      >/tmp/bwvc_run.log 2>&1
    grep -q PASSED /tmp/bwvc_run.log || continue
    read -r ob rb wb tb env rds wds st rank < <("${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${cores}" --directions "${DIRN}" 2>/dev/null | awk -v rk="${RANK}" '
      /overall_bw_gbps/{o=$2} /agg_read_gbps/{r=$2} /agg_write_gbps/{w=$2} /agg_total_gbps/{t=$2}
      /kernel_envelope_us/{e=$2} /reader_done_spread_us/{rd=$2} /writer_done_spread_us/{wd=$2} /starvation_ratio/{s=$2}
      $1==rk{rv=$2} END{print o, r, w, t, e, rd, wd, s, rv}')
    awk -v a="${rank:-0}" -v b="${best}" 'BEGIN{exit !(a>b)}' && {
      best="${rank}"; best_line="${cores},${PAGES},${d},${ob},${rb},${wb},${tb},${env},${rds},${wds},${st}"; }
  done
  [ -n "${best_line}" ] && { echo "${best_line}" >> "${OUT}"; echo "${best_line}"; } \
    || echo "cores=${cores} FAILED (no PASSED run)"
done
echo "BW_VS_CORES_COMPLETE (read_only=${READ_ONLY}) -> ${OUT}"
