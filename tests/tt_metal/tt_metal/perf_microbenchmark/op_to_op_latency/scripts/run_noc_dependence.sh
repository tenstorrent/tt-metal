#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Regather item #1: NoC BW dependence -- read-only aggregate read BW vs spatial extent,
# at 2/4/6/8 rows and 2/4/6/8 columns of cores, on the reader's NoC = NOC0 and NOC1.
# 4 series: {rows, cols} x {NOC0, NOC1}.
#   grid is 8 wide (x) x 7 tall (y):
#     row  = 8 cores (full x-span), grown row-major   -> u rows  = u*8 cores
#     col  = 7 cores (full y-span), grown column-major -> u cols  = u*7 cores
#   rows: only 7 rows exist, so 8 rows is impossible; the rows series uses {2,4,6}
#         plus 7 (=56 cores, full grid) as the all-core saturation point so it reaches
#         the same 56-core endpoint as cols=8 (=56). cols: {2,4,6,8} all valid.
#   read-only => writer just pops the CB (no DRAM writes), so ONLY the reader NoC matters.
#   reader on NOC0: --reader-noc 0 --writer-noc 1 ; on NOC1: --reader-noc 1 --writer-noc 0.
# Per point picks the best input-CB depth. Emits one CSV: orientation,noc,units,cores,in_cb,read_gbps.
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
OUT="${OUT:-${OUTDIR}/read_bw_noc_2468.csv}"
mkdir -p "${OUTDIR}"

WIDTH="${WIDTH:-8}"          # cores per row (x-span)
HEIGHT="${HEIGHT:-7}"        # cores per column (y-span)
INPUT_DEPTHS="${INPUT_DEPTHS:-16 32}"
N="${N:-8}"                  # reader trids in flight (mode 2)
PAGES="${PAGES:-1024}"
ROW_UNITS="${ROW_UNITS:-2 4 6 7}"   # 7 = full grid (8 rows impossible on a 7-tall grid)
COL_UNITS="${COL_UNITS:-2 4 6 8}"

echo "orientation,noc,units,cores,in_cb,read_gbps" > "${OUT}"

# best read BW over input depths -> echoes "bw in_cb"
measure() { # layout_flag noc_flags cores
  local lf="$1" noc="$2" cores="$3" best=0 bestd=0
  for d in ${INPUT_DEPTHS}; do
    rm -f "${DEV_CSV}"
    "${BIN}" --read-only --lean-compute ${lf} ${noc} \
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
  if [ "${rnoc}" -eq 0 ]; then noc="noc0"; nf="--reader-noc 0 --writer-noc 1";
  else                          noc="noc1"; nf="--reader-noc 1 --writer-noc 0"; fi
  for u in ${ROW_UNITS}; do                  # rows: u full rows of WIDTH cores
    cores=$((u * WIDTH)); (( cores > WIDTH*HEIGHT )) && cores=$((WIDTH*HEIGHT))
    read -r bw d < <(measure "" "${nf}" "${cores}")
    echo "rows,${noc},${u},${cores},${d},${bw}" >> "${OUT}"
    echo "rows reader=${noc} ${u} row(s) cores=${cores} -> ${bw} GB/s (in_cb=${d})"
  done
  for u in ${COL_UNITS}; do                  # cols: u full columns of HEIGHT cores
    cores=$((u * HEIGHT)); (( cores > WIDTH*HEIGHT )) && cores=$((WIDTH*HEIGHT))
    read -r bw d < <(measure "--core-layout-col" "${nf}" "${cores}")
    echo "cols,${noc},${u},${cores},${d},${bw}" >> "${OUT}"
    echo "cols reader=${noc} ${u} col(s) cores=${cores} -> ${bw} GB/s (in_cb=${d})"
  done
done
echo "READ_BW_NOC_2468_COMPLETE -> ${OUT}"
