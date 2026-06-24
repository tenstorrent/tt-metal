#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# CB-depth -> latency tradeoff at a fixed (unsaturated) config. Deeper buffers buy a bit
# of BW but cost drain latency. Two sweeps at cores=8, balanced nops, row, good NoC:
#   cb_input_latency.csv  : sweep input CB depth (reader buffer), output fixed
#   cb_output_latency.csv : sweep output CB depth (writer buffer), input fixed
# Metrics per depth: agg_total BW, op-to-op, math-to-math, reader_to_writer, write_tail,
# op_duration, writer_spread_pct.
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
mkdir -p "${OUTDIR}"

CORES="${CORES:-8}"
NOPS="${NOPS:-64}"            # balanced knee for 8 cores
N="${N:-8}"
PAGES="${PAGES:-1024}"

run() { # in_cb out_cb -> runs; leaves DEV_CSV
  rm -f "${DEV_CSV}"
  "${BIN}" --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 \
    --input-cb-depth-tiles "$1" --output-cb-depth-tiles "$2" --writer-end-barrier-mode 0 \
    --lean-compute --skip-output-validation \
    --num-pages-per-core "${PAGES}" --num-programs 2 --compute-nops "${NOPS}" \
    --use-trace --trace-warmup-replays 1 --num-active-cores "${CORES}" --use-device-profiler \
    >/tmp/cbl_run.log 2>&1
  grep -q PASSED /tmp/cbl_run.log
}

A="${OUTDIR}/cb_input_latency.csv"; rm -f "${A}"
echo "==== input CB depth sweep (out_cb=8, cores=${CORES}, nops=${NOPS}) ===="
for d in 16 32 64 128 256; do
  if run "${d}" 8; then
    "${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${CORES}" --csv-out "${A}" \
      --label "sweep=input;in_cb=${d};out_cb=8" 2>/dev/null \
      | grep -E "agg_total_gbps|official_op2op_min_us|m2m_bubble_us|reader_to_writer_us|write_tail_us|op_duration_us"
    echo "  in_cb=${d} done"
  else echo "  in_cb=${d} FAILED"; fi
done

B="${OUTDIR}/cb_output_latency.csv"; rm -f "${B}"
echo "==== output CB depth sweep (in_cb=64, cores=${CORES}, nops=${NOPS}) ===="
for d in 2 4 8 16 32 64; do
  if run 64 "${d}"; then
    "${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${CORES}" --csv-out "${B}" \
      --label "sweep=output;in_cb=64;out_cb=${d}" 2>/dev/null \
      | grep -E "agg_total_gbps|official_op2op_min_us|m2m_bubble_us|reader_to_writer_us|write_tail_us|op_duration_us"
    echo "  out_cb=${d} done"
  else echo "  out_cb=${d} FAILED"; fi
done
echo "CB_LATENCY_COMPLETE"
