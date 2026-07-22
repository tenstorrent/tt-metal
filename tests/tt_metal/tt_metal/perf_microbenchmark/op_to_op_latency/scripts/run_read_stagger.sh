#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Deliberate read-stagger experiment. Holds cores / write volume / layout / out_cb / flush FIXED
# and dials a per-core read-start delay (core i spins i*STAGGER after go) to INDUCE read skew.
# Tests the hypothesis: does staggering reads (so write bursts de-correlate) lower the write-barrier
# congestion? Records the MEASURED read skew (event_timeline) alongside write_drain / write_drain_max
# / writer_done_spread so we plot the write metrics vs the induced read skew (the independent var),
# with everything else constant. If write_drain_max FALLS as read skew rises -> hypothesis confirmed.
#
# Env: CORES, PAGES, OUT_CB, N, IN_CB, STAGGERS (per-core spin counts), OUT, OUTDIR.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)}"
export TT_METAL_DEVICE_PROFILER=1
cd "${TT_METAL_HOME}" || exit 1
BIN="${TT_METAL_HOME}/build_RelWithDebInfo/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
PY="${TT_METAL_HOME}/python_env/bin/python3"
DEC="${SCRIPT_DIR}/decompose_latency_bw.py"
ET="${SCRIPT_DIR}/event_timeline.py"
DEV_CSV="${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv"
OUTDIR="${OUTDIR:-${TT_METAL_HOME}/generated/profiler/op_to_op_runs/stage_a}"
OUT="${OUT:-${OUTDIR}/read_stagger.csv}"
mkdir -p "${OUTDIR}"

CORES="${CORES:-56}"
PAGES="${PAGES:-256}"     # fixed per-core write volume
OUT_CB="${OUT_CB:-32}"    # fixed flush cadence
N="${N:-4}"
IN_CB="${IN_CB:-16}"
NOPS="${NOPS:-0}"         # keep compute light so reads aren't gated (isolate the stagger effect)
# Calibration: ~0.5us induced last-core read skew per unit stagger at 56c (loop is ~9cyc/iter,
# last core = 55*stagger iters). Steps below span induced read skew ~0 -> ~30us.
STAGGERS="${STAGGERS:-0 1 2 4 8 16 32 64}"  # per-core spin count (core i spins i*this)

echo "stagger,cores,read_start_skew_us,read_compl_skew_us,write_drain_us,write_drain_max_us,writer_done_spread_us,agg_total_gbps" > "${OUT}"
for S in ${STAGGERS}; do
  rm -f "${DEV_CSV}"
  "${BIN}" --skip-output-validation --lean-compute --reader-noc 0 --writer-noc 1 \
    --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 --reader-stagger-cycles "${S}" \
    --input-cb-depth-tiles "${IN_CB}" --output-cb-depth-tiles "${OUT_CB}" --writer-end-barrier-mode 0 \
    --num-pages-per-core "${PAGES}" --num-programs 4 --compute-nops "${NOPS}" \
    --use-trace --trace-warmup-replays 1 --num-active-cores "${CORES}" --use-device-profiler \
    >/tmp/stagger_run.log 2>&1
  grep -q PASSED /tmp/stagger_run.log || { echo "stagger=${S} FAILED: $(grep -oE 'TT_FATAL: .*' /tmp/stagger_run.log | head -1)"; continue; }
  # measured read skew from event_timeline summary lines (units: cyc ~= ns at 1GHz -> /1000 = us):
  #   "read skew (first->last core first-read-return) = N cyc"  -> start skew
  #   "read INTER-core skew (first->last core reads-complete) = N cyc" -> completion skew
  read -r rs rc < <("${PY}" "${ET}" --input-file "${DEV_CSV}" 2>/dev/null | awk '
    /read skew \(first->last/   {v=$0; gsub(/[^0-9]/,"",v); s=v}
    /read INTER-core skew/       {v=$0; gsub(/[^0-9]/,"",v); c=v}
    END{print (s+0)/1000.0, (c+0)/1000.0}')
  read -r wd wdm ws at < <("${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${CORES}" 2>/dev/null | awk '
    $1=="write_drain_us"{d=$2} $1=="write_drain_max_us"{m=$2} $1=="writer_done_spread_us"{s=$2} $1=="agg_total_gbps"{a=$2}
    END{print d, m, s, a}')
  line="${S},${CORES},${rs},${rc},${wd},${wdm},${ws},${at}"
  echo "${line}" >> "${OUT}"; echo "${line}"
done
echo "READ_STAGGER_COMPLETE -> ${OUT}"
