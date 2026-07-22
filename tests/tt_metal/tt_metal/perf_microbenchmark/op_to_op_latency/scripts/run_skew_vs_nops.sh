#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Canonical sweep 3/6: writer done-skew vs compute load (NOPs/tile).
# Full read+write op at a fixed core count (default 56), good NoCs, lean compute, trace.
# Sweeps --compute-nops: as compute grows it paces the consumer, changing how writes pile up
# at the DRAM tail -> shows whether adding math tightens or worsens the writer completion skew.
# Records: cores,nops,in_cb,agg_total_gbps,agg_write_gbps,writer_done_spread_us,writer_spread_pct,starvation_ratio.
#
# Env: CORES (single value), NOPS_LIST, IN_CB, N (trid), PAGES, OUT, OUTDIR.
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
OUT="${OUT:-${OUTDIR}/skew_vs_nops.csv}"
mkdir -p "${OUTDIR}"

CORES="${CORES:-56}"
NOPS_LIST="${NOPS_LIST:-0 32 64 128 256 512 1024 2048}"
IN_CB="${IN_CB:-32}"
N="${N:-8}"
PAGES="${PAGES:-1024}"

echo "cores,nops,in_cb,agg_total_gbps,agg_write_gbps,writer_done_spread_us,writer_spread_pct,starvation_ratio" > "${OUT}"
for nops in ${NOPS_LIST}; do
  rm -f "${DEV_CSV}"
  "${BIN}" --skip-output-validation --lean-compute --reader-noc 0 --writer-noc 1 \
    --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 \
    --input-cb-depth-tiles "${IN_CB}" --output-cb-depth-tiles 2 \
    --num-pages-per-core "${PAGES}" --num-programs 2 --compute-nops "${nops}" \
    --use-trace --trace-warmup-replays 1 --num-active-cores "${CORES}" --use-device-profiler \
    >/tmp/skewnops_run.log 2>&1
  grep -q PASSED /tmp/skewnops_run.log || { echo "cores=${CORES} nops=${nops} FAILED"; continue; }
  read -r tb wb wds wsp st < <("${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${CORES}" 2>/dev/null | awk '
    /agg_total_gbps/{t=$2} /agg_write_gbps/{w=$2}
    /writer_done_spread_us/{wd=$2} /writer_spread_pct/{ws=$2} /starvation_ratio/{s=$2}
    END{print t, w, wd, ws, s}')
  line="${CORES},${nops},${IN_CB},${tb},${wb},${wds},${wsp},${st}"
  echo "${line}" >> "${OUT}"; echo "${line}"
done
echo "SKEW_VS_NOPS_COMPLETE -> ${OUT}"
