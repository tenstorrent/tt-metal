#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Output-CB depth -> math-to-math, evaluated at the BALANCED knee NOPs (not nops=0), since
# the out_cb drain-tail effect depends on compute pacing. Per core count: find the knee NOPs
# (largest where read BW still >= 95% of peak), then sweep out_cb at that NOPs. N=4, in_cb=64.
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
OUT="${OUTDIR}/m2m_vs_outcb_balanced.csv"
mkdir -p "${OUTDIR}"

N="${N:-4}"
READER_NOC="${READER_NOC:-0}"   # good: reader=0/writer=1 ; swapped (write-contended): reader=1/writer=0
WRITER_NOC="${WRITER_NOC:-1}"
IN_CB="${IN_CB:-64}"
PAGES="${PAGES:-1024}"
CORES_LIST="${CORES_LIST:-8 16 56}"
OUTPUT_DEPTHS="${OUTPUT_DEPTHS:-2 4 8 16 32 64}"
NOP_SWEEP="${NOP_SWEEP:-0 32 64 128 256 512 1024 2048}"
KNEE_TOL="${KNEE_TOL:-0.95}"

run() { # cores out_cb nops
  rm -f "${DEV_CSV}"
  "${BIN}" --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 \
    --input-cb-depth-tiles "${IN_CB}" --output-cb-depth-tiles "$2" --writer-end-barrier-mode 0 \
    --reader-noc "${READER_NOC}" --writer-noc "${WRITER_NOC}" \
    --lean-compute --skip-output-validation \
    --num-pages-per-core "${PAGES}" --num-programs 2 --compute-nops "$3" \
    --use-trace --trace-warmup-replays 1 --num-active-cores "$1" --use-device-profiler >/tmp/ocb_run.log 2>&1
  grep -q PASSED /tmp/ocb_run.log
}
field() { "${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "$1" 2>/dev/null | awk -v f="$2" '$1==f{print $2}'; }

rm -f "${OUT}"
for C in ${CORES_LIST}; do
  echo "==== cores=${C}: knee search (N=${N}, out_cb=8) ===="
  run "${C}" 8 0 || { echo "  nops=0 FAILED"; continue; }
  peak=$(field "${C}" agg_read_gbps); knee=0
  for nq in ${NOP_SWEEP}; do
    run "${C}" 8 "${nq}" || continue
    bw=$(field "${C}" agg_read_gbps)
    echo "  nops=${nq}: agg_read=${bw}"
    if awk -v a="${bw:-0}" -v p="${peak}" -v t="${KNEE_TOL}" 'BEGIN{exit !(a>=p*t)}'; then knee="${nq}"; else break; fi
  done
  echo "  -> balanced knee nops=${knee} (peak_read=${peak})"
  echo "==== cores=${C}: output-CB sweep at balanced nops=${knee} ===="
  for o in ${OUTPUT_DEPTHS}; do
    if run "${C}" "${o}" "${knee}"; then
      "${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${C}" --csv-out "${OUT}" \
        --label "cores=${C};nops=${knee};out_cb=${o}" 2>/dev/null \
        | grep -E "agg_total_gbps|official_op2op_min_us|m2m_bubble_us|reader_to_writer_us"
      echo "  cores=${C} out_cb=${o} done"
    else echo "  cores=${C} out_cb=${o} FAILED"; fi
  done
done
echo "OUTCB_BALANCED_COMPLETE"
