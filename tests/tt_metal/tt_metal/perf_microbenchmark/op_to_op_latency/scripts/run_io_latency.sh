#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Canonical sweep 4/6: two-sided I/O latency decomposition.
#  (A) READ side -- read-fill latency vs read txns in flight (N) at cores {8,56}:
#        read_fill_head_us = first read issued (READ_BEFORE_BARRIER) -> first math (UNPACK tile0).
#        Larger N => larger first in-flight batch must complete before the first CB push, so the
#        read-fill head grows with N. Output CB held fixed. Always full reads.
#  (B) WRITE side -- write metrics vs OUTPUT CB depth at cores {8,56}, for each read-bytes regime
#      in READ_BYTES_LIST (default "0 32"):
#        rb=0  -> full reads = READ-BOUND baseline. Output CB barely matters (the read side
#                throttles, the CB never fills): agg ~flat, write_drain ~flat in out_cb.
#        rb=32 -> cheap reads (NoC-read 32B, push a full page; payload is dummy) = OUTPUT-BOUND.
#                The writer is the bottleneck, the CB fills at slow cores, so output CB depth NOW
#                matters: agg_write rises with depth and the output-side STARVATION grows
#                (starvation_ratio + writer_done_spread climb with out_cb). NB: with cheap reads
#                agg_read/agg_total are meaningless (decompose assumes full-page reads) -- read
#                agg_write + the write-side skew metrics instead.
#        write_drain_us = LAST write issued -> write barrier complete; ~flat in out_cb in BOTH
#                regimes (the per-core final drain is small). write_tail = whole write phase.
#        Input CB + N held fixed.
#  Good NoCs (reader=NOC0/writer=NOC1), lean compute (nops=0), trace. in_cb=64 (>= 2*N for N<=32).
# Records CSVs:
#   read_lat_vs_n.csv               : cores,pages,N,read_fill_head_us,m2m_bubble_us,official_op2op_min_us,agg_total_gbps
#   write_lat_vs_outcb.csv          : (rb=0)  cores,pages,out_cb,read_bytes,agg_write_gbps,write_drain_us,writer_done_spread_us,write_tail_us,starvation_ratio
#   write_lat_vs_outcb_cheapread.csv: (rb>0)  same columns, output-bound regime
#
# Env: CORES_LIST, N_LIST, OUTCB_LIST, READ_BYTES_LIST, IN_CB, OUT_CB_FIXED, N_FIXED, TOTAL_PAGES,
#      MIN_PAGES_PER_CORE, PAGES (override -> fixed-per-core), OUTDIR.
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
mkdir -p "${OUTDIR}"

CORES_LIST="${CORES_LIST:-8 56}"
N_LIST="${N_LIST:-2 4 8 16 32}"
OUTCB_LIST="${OUTCB_LIST:-2 4 8 16 32}"
READ_BYTES_LIST="${READ_BYTES_LIST:-0 32}"   # 0 = full read (read-bound); 32 = cheap read (output-bound)
IN_CB="${IN_CB:-64}"
OUT_CB_FIXED="${OUT_CB_FIXED:-8}"   # output CB during the read-side (N) sweep
N_FIXED="${N_FIXED:-8}"             # trid during the write-side (out_cb) sweep
# Fixed-TOTAL (data-parallel) work model, matching run_stage_a_sweep.sh: PAGES = TOTAL_PAGES /
# cores, set per core count by set_pages() below. Set PAGES=<n> to force fixed-per-core.
TOTAL_PAGES="${TOTAL_PAGES:-3584}"
MIN_PAGES_PER_CORE="${MIN_PAGES_PER_CORE:-8}"
PAGES_OVERRIDE="${PAGES:-}"
PAGES=1024  # placeholder; set_pages() overwrites per core count
set_pages() { # cores
  if [ -n "${PAGES_OVERRIDE}" ]; then PAGES="${PAGES_OVERRIDE}"; else
    PAGES=$(( TOTAL_PAGES / $1 )); (( PAGES < MIN_PAGES_PER_CORE )) && PAGES=${MIN_PAGES_PER_CORE}; fi
}

run() { # N in_cb out_cb cores read_bytes -> runs full op; leaves DEV_CSV
  rm -f "${DEV_CSV}"
  "${BIN}" --skip-output-validation --lean-compute --reader-noc 0 --writer-noc 1 \
    --reader-dbuf-trid --reader-trid-in-flight "$1" --reader-push-tiles 2 \
    --input-cb-depth-tiles "$2" --output-cb-depth-tiles "$3" --reader-read-bytes "${5:-0}" \
    --num-pages-per-core "${PAGES}" --num-programs 4 --compute-nops 0 \
    --use-trace --trace-warmup-replays 1 --num-active-cores "$4" --use-device-profiler \
    >/tmp/iolat_run.log 2>&1
  grep -q PASSED /tmp/iolat_run.log
}
# prints: read_fill m2m op2op write_drain write_tail agg_total agg_write writer_spread starvation
fields() {
  "${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "$1" 2>/dev/null | awk '
    $1=="read_fill_head_us"{rf=$2} $1=="m2m_bubble_us"{m=$2} $1=="official_op2op_min_us"{o=$2}
    $1=="write_drain_us"{wd=$2} $1=="write_tail_us"{wt=$2} $1=="agg_total_gbps"{at=$2}
    $1=="agg_write_gbps"{aw=$2} $1=="writer_done_spread_us"{ws=$2} $1=="starvation_ratio"{s=$2}
    END{print rf, m, o, wd, wt, at, aw, ws, s}'
}

# (A) read-fill latency vs N -- always full reads.
A="${OUTDIR}/read_lat_vs_n.csv"
echo "cores,pages,N,read_fill_head_us,m2m_bubble_us,official_op2op_min_us,agg_total_gbps" > "${A}"
echo "==== (A) read-fill latency vs N (in_cb=${IN_CB}, out_cb=${OUT_CB_FIXED}) ===="
for C in ${CORES_LIST}; do
  set_pages "${C}"
  for N in ${N_LIST}; do
    [ "${IN_CB}" -lt $((2 * N)) ] && { echo "cores=${C} N=${N} SKIP (in_cb<2N)"; continue; }
    if run "${N}" "${IN_CB}" "${OUT_CB_FIXED}" "${C}" 0; then
      read -r rf m o wd wt at aw ws s < <(fields "${C}")
      echo "${C},${PAGES},${N},${rf},${m},${o},${at}" | tee -a "${A}"
    else echo "cores=${C} N=${N} FAILED"; fi
  done
done

# (B) write metrics vs output CB depth -- once per read-bytes regime.
echo "==== (B) write metrics vs output CB depth (in_cb=${IN_CB}, N=${N_FIXED}) ===="
for RB in ${READ_BYTES_LIST}; do
  if [ "${RB}" -gt 0 ]; then B="${OUTDIR}/write_lat_vs_outcb_cheapread.csv"; tag="cheap-read ${RB}B (output-bound)";
  else                       B="${OUTDIR}/write_lat_vs_outcb.csv";          tag="full read (read-bound)"; fi
  echo "  -- regime: ${tag} -> ${B}"
  echo "cores,pages,out_cb,read_bytes,agg_write_gbps,write_drain_us,writer_done_spread_us,write_tail_us,starvation_ratio" > "${B}"
  for C in ${CORES_LIST}; do
    set_pages "${C}"
    for OC in ${OUTCB_LIST}; do
      if run "${N_FIXED}" "${IN_CB}" "${OC}" "${C}" "${RB}"; then
        read -r rf m o wd wt at aw ws s < <(fields "${C}")
        echo "${C},${PAGES},${OC},${RB},${aw},${wd},${ws},${wt},${s}" | tee -a "${B}"
      else echo "cores=${C} out_cb=${OC} rb=${RB} FAILED"; fi
    done
  done
done
echo "IO_LATENCY_COMPLETE -> ${A} , write_lat_vs_outcb[_cheapread].csv"
