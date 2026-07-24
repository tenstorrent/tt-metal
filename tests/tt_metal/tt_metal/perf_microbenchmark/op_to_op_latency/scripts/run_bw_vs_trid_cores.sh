#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Canonical sweep 5/6: aggregate BW vs read txns in flight (N) for various core counts.
# READ-ONLY (writer just pops the CB; no DRAM writes) so this isolates the read path and the
# read BW tops out at the full ~210 GB/s read ceiling (not ~100, which is the read half of a
# read+write op where the two share DRAM). Good NoCs, lean compute, trace. CB held FIXED+DEEP so
# N is the only variable (decoupled from the cb_reserve_back run-ahead axis). Shows the knee N
# (smallest N that saturates BW) and whether it shifts with core count (BW-unsaturated at low
# core counts vs saturated at high). For the latency-vs-N x cb cross, use run_lat_vs_trid_cb.sh.
# Records: cores,N,actual_gbps,fair_gbps,starvation_ratio.
#   actual_gbps = aggregate read BW (union-span based, dragged down by the slow-core tail).
#   fair_gbps   = cores * per-core-median read BW = the "no-skew" projection (what aggregate
#                 would be if every core ran at the median rate). actual < fair => skew tax.
#
# Env: CORES_LIST, N_LIST, IN_CB, OUT_CB, PAGES, OUT, OUTDIR.
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
OUT="${OUT:-${OUTDIR}/bw_vs_trid_cores.csv}"
mkdir -p "${OUTDIR}"

IN_CB="${IN_CB:-64}"          # fixed + deep (>= 2*max N) to decouple from N
OUT_CB="${OUT_CB:-8}"
CORES_LIST="${CORES_LIST:-1 8 16 32 56}"
N_LIST="${N_LIST:-2 3 4 6 8 12 16 24 32}"
# Fixed-TOTAL (data-parallel) work model, matching run_stage_a_sweep.sh: pages_per_core =
# TOTAL_PAGES / cores (recomputed per core count). NOTE: small shards at high core count pay a
# ~5% pipeline fill/drain tax on aggregate BW -- the realistic sharded-op number. Set PAGES=<n>
# to force the legacy fixed-per-core model.
TOTAL_PAGES="${TOTAL_PAGES:-3584}"
MIN_PAGES_PER_CORE="${MIN_PAGES_PER_CORE:-8}"
PAGES_OVERRIDE="${PAGES:-}"

echo "cores,pages,N,actual_gbps,fair_gbps,starvation_ratio" > "${OUT}"
for C in ${CORES_LIST}; do
  if [ -n "${PAGES_OVERRIDE}" ]; then PAGES="${PAGES_OVERRIDE}"; else
    PAGES=$(( TOTAL_PAGES / C )); (( PAGES < MIN_PAGES_PER_CORE )) && PAGES=${MIN_PAGES_PER_CORE}; fi
  for N in ${N_LIST}; do
    if [ "${IN_CB}" -lt $((2 * N)) ]; then continue; fi
    rm -f "${DEV_CSV}"
    "${BIN}" --read-only --lean-compute --reader-noc 0 --writer-noc 1 \
      --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 \
      --input-cb-depth-tiles "${IN_CB}" --output-cb-depth-tiles "${OUT_CB}" \
      --num-pages-per-core "${PAGES}" --num-programs 2 --compute-nops 0 \
      --use-trace --trace-warmup-replays 1 --num-active-cores "${C}" --use-device-profiler \
      >/tmp/bwtc_run.log 2>&1
    grep -q PASSED /tmp/bwtc_run.log || { echo "cores=${C} N=${N} FAILED"; continue; }
    # actual = aggregate read BW; fair = cores * per-core-median read BW (no-skew projection)
    line=$("${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${C}" 2>/dev/null | awk -v c="${C}" -v n="${N}" -v pg="${PAGES}" '
      /agg_read_gbps/{r=$2} /per_core_gbps_median/{p=$2} /starvation_ratio/{s=$2}
      END{printf "%s,%s,%s,%.3f,%.3f,%s", c, pg, n, r, c*p, s}')
    echo "${line}" >> "${OUT}"; echo "${line}"
  done
done
echo "BW_VS_TRID_CORES_COMPLETE -> ${OUT}"
