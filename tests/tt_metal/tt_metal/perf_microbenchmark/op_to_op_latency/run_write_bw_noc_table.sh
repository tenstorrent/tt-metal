#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Write-BW vs spatial-extent x writer-NoC table (full pipeline, reader pinned to its best
# NoC = NOC0 so reads don't bottleneck; isolates the writer NoC's effect on write BW).
# 4 series: {rows, cols} x {writer NOC0, writer NOC1}.
# Per point, picks the best output-CB depth (writer pipelining). Emits one CSV.
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
OUT="${OUTDIR}/write_bw_noc_table.csv"
mkdir -p "${OUTDIR}"

WIDTH="${WIDTH:-8}"
HEIGHT="${HEIGHT:-7}"
OUTPUT_DEPTHS="${OUTPUT_DEPTHS:-2 8 16}"
IN_CB="${IN_CB:-32}"        # reads on NOC0 aren't the bottleneck; fixed input depth
N="${N:-8}"
PAGES="${PAGES:-1024}"

# best write BW over output depths -> echoes "bw out_cb"
# CONSTRAINT: reader and writer must be on DIFFERENT NoCs, so reader = the other NoC.
# This means the two writer-NoC series are the two valid pipeline configs:
#   writer=NOC1 (reader=NOC0, fast reads) and writer=NOC0 (reader=NOC1, read-capped).
measure() { # layout_flag wnoc cores
  local lf="$1" wnoc="$2" cores="$3" rnoc=$((1 - wnoc)) best=0 besto=0
  for o in ${OUTPUT_DEPTHS}; do
    rm -f "${DEV_CSV}"
    "${BIN}" --lean-compute --skip-output-validation ${lf} \
      --reader-noc "${rnoc}" --writer-noc "${wnoc}" \
      --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 \
      --input-cb-depth-tiles "${IN_CB}" --output-cb-depth-tiles "${o}" --writer-end-barrier-mode 0 \
      --num-pages-per-core "${PAGES}" --num-programs 2 --compute-nops 0 \
      --use-trace --trace-warmup-replays 1 --num-active-cores "${cores}" --use-device-profiler \
      >/tmp/wbnoc_run.log 2>&1
    grep -q PASSED /tmp/wbnoc_run.log || continue
    local bw
    bw=$("${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${cores}" 2>/dev/null | awk '/agg_write_gbps/{print $2}')
    awk -v a="${bw:-0}" -v b="${best}" 'BEGIN{exit !(a>b)}' && { best="${bw}"; besto="${o}"; }
  done
  echo "${best} ${besto}"
}

echo "orientation,writer_noc,units,cores,out_cb,write_gbps" > "${OUT}"
for wnoc in 0 1; do
  for u in $(seq 1 "${HEIGHT}"); do
    cores=$((u * WIDTH))
    read -r bw o < <(measure "" "${wnoc}" "${cores}")
    echo "rows,noc${wnoc},${u},${cores},${o},${bw}" >> "${OUT}"
    echo "rows writer=noc${wnoc} ${u} row(s) cores=${cores} -> ${bw} GB/s (out_cb=${o})"
  done
  for u in $(seq 1 "${WIDTH}"); do
    cores=$((u * HEIGHT))
    read -r bw o < <(measure "--core-layout-col" "${wnoc}" "${cores}")
    echo "cols,noc${wnoc},${u},${cores},${o},${bw}" >> "${OUT}"
    echo "cols writer=noc${wnoc} ${u} col(s) cores=${cores} -> ${bw} GB/s (out_cb=${o})"
  done
done
echo "WRITE_BW_NOC_TABLE_COMPLETE -> ${OUT}"
