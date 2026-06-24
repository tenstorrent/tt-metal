#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Fewer-vs-more-cores: hold TOTAL work fixed (an "op" of TOTAL tiles), spread it across N
# cores (pages/core = TOTAL/N), and see whether aggregate DRAM BW saturates (so adding
# cores stops helping) while latency / starvation grow. Good NoC config is the default now
# (reader=NOC0, writer=NOC1), row-major placement (saturates with the fewest cores).
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
LAYOUT="${LAYOUT:-row}"                        # row | col core allocation order
REVERSE="${REVERSE:-0}"                         # 1 = reverse fill order (last cores first)
LF=""; [ "${LAYOUT}" = "col" ] && LF="--core-layout-col"
tag="${LAYOUT}"; [ "${REVERSE}" = "1" ] && { LF="${LF} --core-reverse"; tag="${LAYOUT}_rev"; }
OUT="${OUTDIR}/cores_sweep_fixed_total_${tag}.csv"
mkdir -p "${OUTDIR}"

TOTAL="${TOTAL:-16384}"                       # fixed total tiles (the op size)
CORES="${CORES:-1 2 4 8 16 24 32 40 48 56}"
IN_CB="${IN_CB:-32}"
OUT_CB="${OUT_CB:-8}"
N="${N:-8}"

rm -f "${OUT}"
for C in ${CORES}; do
  pages=$(( (TOTAL + C / 2) / C ))            # round TOTAL/C
  [ "${pages}" -lt 1 ] && pages=1
  total=$(( pages * C ))
  rm -f "${DEV_CSV}"
  "${BIN}" --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 ${LF} \
    --input-cb-depth-tiles "${IN_CB}" --output-cb-depth-tiles "${OUT_CB}" --writer-end-barrier-mode 0 \
    --lean-compute --skip-output-validation \
    --num-pages-per-core "${pages}" --num-programs 2 --compute-nops 0 \
    --use-trace --trace-warmup-replays 1 --num-active-cores "${C}" --use-device-profiler \
    >/tmp/cs_run.log 2>&1
  if ! grep -q PASSED /tmp/cs_run.log; then
    echo "cores=${C} FAILED: $(grep -oE 'TT_FATAL: .*' /tmp/cs_run.log | head -1)"; continue
  fi
  "${PY}" "${DEC}" --pages-per-core "${pages}" --num-cores "${C}" --csv-out "${OUT}" \
    --label "cores=${C};pages_per_core=${pages};total_tiles=${total}" 2>/dev/null \
    | grep -E "agg_total_gbps|official_op2op_min_us|m2m_bubble_us|starvation_ratio"
  echo "cores=${C} pages/core=${pages} total=${total} -> appended"
done
echo "CORES_SWEEP_COMPLETE -> ${OUT}"
