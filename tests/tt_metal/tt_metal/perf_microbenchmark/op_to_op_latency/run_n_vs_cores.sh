#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Read-N latency-hiding vs core count. Isolates N (reader trid_in_flight = outstanding NoC
# read txns) as the ONLY latency-hiding knob: input CB held FIXED+DEEP (in_cb=128) so the
# cb_reserve_back run-ahead/stall axis is constant across N, and READ-ONLY so there is no
# writer DRAM contention. Headline metric is per-core read BW; knee N = smallest N reaching
# ~peak per-core BW. Question: does knee N depend on core count (BW unsaturated vs saturated)?
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
OUT="${OUT:-${OUTDIR}/n_vs_cores.csv}"
mkdir -p "${OUTDIR}"

IN_CB="${IN_CB:-128}"          # fixed + deep: never the limiter, holds reserve-stall axis constant
OUT_CB="${OUT_CB:-8}"          # just needs to drain (writer pops even in read-only)
PAGES="${PAGES:-2048}"         # long read span -> stable per-core BW
CORES_LIST="${CORES_LIST:-1 2 4 8 16 32 56}"
N_LIST="${N_LIST:-1 2 3 4 6 8 12 16 24 32}"

# decompose drops per_core_* / agg_read from its CSV and agg_total is NaN in read-only, so
# we capture the read fields from its stdout and write our own CSV here.
field() { awk -v f="$1" '$1==f{print $2}' <<<"$2"; }

echo "cores,N,in_cb,agg_read_gbps,per_core_read_med,per_core_read_min,starvation_ratio" > "${OUT}"
for C in ${CORES_LIST}; do
  for N in ${N_LIST}; do
    rm -f "${DEV_CSV}"
    "${BIN}" --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 \
      --input-cb-depth-tiles "${IN_CB}" --output-cb-depth-tiles "${OUT_CB}" \
      --read-only --skip-output-validation --lean-compute \
      --num-pages-per-core "${PAGES}" --num-programs 2 --compute-nops 0 \
      --use-trace --trace-warmup-replays 1 --num-active-cores "${C}" --use-device-profiler \
      >/tmp/nvc_run.log 2>&1
    if grep -q PASSED /tmp/nvc_run.log; then
      out=$("${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${C}" 2>/dev/null)
      ar=$(field agg_read_gbps "${out}"); pcm=$(field per_core_gbps_median "${out}")
      pcmin=$(field per_core_gbps_min "${out}"); st=$(field starvation_ratio "${out}")
      echo "${C},${N},${IN_CB},${ar},${pcm},${pcmin},${st}" >> "${OUT}"
      echo "  cores=${C} N=${N}: agg_read=${ar} per_core_med=${pcm} starv=${st}"
    else echo "  cores=${C} N=${N} FAILED: $(grep -oE 'TT_FATAL: .*' /tmp/nvc_run.log | head -1)"; fi
  done
done
echo "N_VS_CORES_COMPLETE"
