#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Read-BW vs spatial-extent x NoC table (read-only, isolates the read NoC path).
# 4 series: {rows, cols} x {default reader NoC (NOC1), swapped (NOC0)}.
#   rows: N full rows of cores (N*WIDTH), grown row-major.
#   cols: N full columns (N*HEIGHT), grown column-major.
# Per point, picks the best input-CB depth. Emits one CSV: orientation,noc,units,cores,in_cb,read_gbps.
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
OUT="${OUTDIR}/read_bw_noc_table.csv"
mkdir -p "${OUTDIR}"

WIDTH="${WIDTH:-8}"          # usable grid width (cores per row)
HEIGHT="${HEIGHT:-7}"        # usable grid height (cores per column)
INPUT_DEPTHS="${INPUT_DEPTHS:-16 32}"
N="${N:-8}"                  # trid_in_flight
PAGES="${PAGES:-1024}"

echo "orientation,noc,units,cores,in_cb,read_gbps" > "${OUT}"

# best read BW over input depths -> echoes "bw in_cb"
measure() { # layout_flag extra_flags cores
  local lf="$1" extra="$2" cores="$3" best=0 bestd=0
  for d in ${INPUT_DEPTHS}; do
    rm -f "${DEV_CSV}"
    "${BIN}" --read-only --lean-compute ${lf} ${extra} \
      --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 \
      --input-cb-depth-tiles "${d}" --output-cb-depth-tiles 2 \
      --num-pages-per-core "${PAGES}" --num-programs 2 --compute-nops 0 \
      --use-trace --trace-warmup-replays 1 --num-active-cores "${cores}" --use-device-profiler \
      >/tmp/rbnoc_run.log 2>&1
    grep -q PASSED /tmp/rbnoc_run.log || continue
    local bw
    bw=$("${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${cores}" 2>/dev/null | awk '/agg_read_gbps/{print $2}')
    awk -v a="${bw:-0}" -v b="${best}" 'BEGIN{exit !(a>b)}' && { best="${bw}"; bestd="${d}"; }
  done
  echo "${best} ${bestd}"
}

for rnoc in 0 1; do
  noc="noc${rnoc}"; extra="--reader-noc ${rnoc}"
  for u in $(seq 1 "${HEIGHT}"); do          # rows: 1..HEIGHT rows of WIDTH cores
    cores=$((u * WIDTH))
    read -r bw d < <(measure "" "${extra}" "${cores}")
    echo "rows,${noc},${u},${cores},${d},${bw}" >> "${OUT}"
    echo "rows reader=${noc} ${u} row(s) cores=${cores} -> ${bw} GB/s (in_cb=${d})"
  done
  for u in $(seq 1 "${WIDTH}"); do            # cols: 1..WIDTH columns of HEIGHT cores
    cores=$((u * HEIGHT))
    read -r bw d < <(measure "--core-layout-col" "${extra}" "${cores}")
    echo "cols,${noc},${u},${cores},${d},${bw}" >> "${OUT}"
    echo "cols reader=${noc} ${u} col(s) cores=${cores} -> ${bw} GB/s (in_cb=${d})"
  done
done
echo "READ_BW_NOC_TABLE_COMPLETE -> ${OUT}"
