#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Stage A of the op-to-op latency/BW campaign: balanced-compute decomposition sweep.
# For each layout (row, column) x core count:
#   1. tune input CB depth for peak aggregate read BW (nops=0, read-bound),
#   2. derive balanced compute: nops so per-tile compute (~COPY_NS + nops) ~= per-tile
#      read time at the measured per-core BW,
#   3. final full-pipeline run, decompose into BW + latency + starvation columns.
# Emits one CSV per layout (chart-ready). Resumable: rows already present are skipped.
# Reader trid_in_flight is locked at 8 (re-tune showed it flat 4..32).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)}"
export TT_METAL_DEVICE_PROFILER=1
cd "${TT_METAL_HOME}" || exit 1  # binary detects root from cwd, not env -> must run from repo root
# dispatch-core profiling off (workers only); official op2op uses KERNEL zones, no DISP markers needed
# export TT_METAL_DEVICE_PROFILER_DISPATCH=1

BIN="${TT_METAL_HOME}/build_RelWithDebInfo/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency"
PY="${TT_METAL_HOME}/python_env/bin/python3"
DEC="${SCRIPT_DIR}/decompose_latency_bw.py"
DEV_CSV="${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv"
OUTDIR="${OUTDIR:-${TT_METAL_HOME}/generated/profiler/op_to_op_runs/stage_a}"
mkdir -p "${OUTDIR}"

CORES="${CORES:-1 2 4 8 16 32 56}"          # coarse first; densify (e.g. add 48) later
LAYOUTS="${LAYOUTS:-row col}"
INPUT_DEPTHS="${INPUT_DEPTHS:-16 32 64}"     # input CB depth candidates (>= 2*trid_in_flight)
OUTPUT_DEPTHS="${OUTPUT_DEPTHS:-2 4 8 16 32}"  # output CB candidates (tuned in balanced regime)
OUT_CB_BASE="${OUT_CB_BASE:-4}"              # output CB used during the input-depth tune
N="${N:-4}"                                  # trid_in_flight (4 saturates BW at all core counts)
# Final run is repeated per end-barrier mode so the writer-drain/barrier term of math-to-math
# can be attributed: 0=noc_async_write_barrier (full ack), 1=writes_flushed, 2=none (relaxed).
# writer-CB/barrier contribution = m2m(mode 0) - m2m(mode 2). Tuning runs at TUNE_BARRIER.
BARRIER_MODES="${BARRIER_MODES:-0 1 2}"
TUNE_BARRIER="${TUNE_BARRIER:-0}"
# Fixed-TOTAL (data-parallel) work model: a real op shards a fixed-size tensor across the grid,
# so pages_per_core = TOTAL_PAGES / cores. This keeps execution duration bounded as cores grow
# (instead of weak-scaling, where fixed pages/core makes the high-core write phase ~1ms and
# swamps the op-to-op / skew signal). TOTAL_PAGES=3584 -> 56c gets 64 pages (~67us of work,
# calibrated), large enough that compute (NOPs) actually back-pressures the reader, and divides
# evenly across {1,2,4,8,16,32,56}. PAGES is recomputed per core count.
TOTAL_PAGES="${TOTAL_PAGES:-3584}"
MIN_PAGES_PER_CORE="${MIN_PAGES_PER_CORE:-8}"  # floor so the trid pipeline (in_cb >= 2*N) stays valid
# Cap input-CB depth at pages_per_core / IN_CB_CAP_FRAC so the reader CANNOT stage the whole shard
# and run ahead of compute. Without this the peak-BW in_cb tune just picks a CB big enough to hold
# the shard, the reader never stalls on compute, and the NOP knee degenerates to max. /4 keeps the
# reader >=~75% compute-coupled (calibrated: in_cb=16 at 64 pages gates read BW ~8% under load).
IN_CB_CAP_FRAC="${IN_CB_CAP_FRAC:-4}"
PAGES=1024                                     # placeholder; set to TOTAL_PAGES/cores in the core loop
TUNE_PROGS="${TUNE_PROGS:-2}"
FINAL_PROGS="${FINAL_PROGS:-4}"
BW_TOL="${BW_TOL:-0.98}"                     # smallest depth within this frac of peak BW
# Balance = NOP knee: largest NOPs/tile where read BW still sits at its read-bound peak.
# Beyond the knee compute gates reads (BW drops). Early-exit once BW falls below KNEE_TOL*peak.
NOP_SWEEP="${NOP_SWEEP:-0 32 64 128 256 512 1024 1536}"
KNEE_TOL="${KNEE_TOL:-0.95}"

run_test() { # layout_flag cores in_cb out_cb nops nprogs [barrier_mode=0]
  local lf="$1" cores="$2" in_cb="$3" out_cb="$4" nops="$5" nprogs="$6" barrier="${7:-0}"
  rm -f "${DEV_CSV}"
  "${BIN}" --reader-dbuf-trid --reader-trid-in-flight "${N}" --reader-push-tiles 2 \
    --input-cb-depth-tiles "${in_cb}" --output-cb-depth-tiles "${out_cb}" \
    --writer-end-barrier-mode "${barrier}" --lean-compute --skip-output-validation ${lf} \
    --num-pages-per-core "${PAGES}" --num-programs "${nprogs}" \
    --use-trace --trace-warmup-replays 1 --compute-nops "${nops}" \
    --num-active-cores "${cores}" --use-device-profiler >/tmp/stage_a_run.log 2>&1
  grep -q "PASSED" /tmp/stage_a_run.log
}

decomp_field() { # field-name cores  (reads current DEV_CSV, prints value)
  "${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "$2" 2>/dev/null \
    | awk -v f="$1" '$1==f{print $2}'
}

for layout in ${LAYOUTS}; do
  lf=""; [ "${layout}" = "col" ] && lf="--core-layout-col"
  OUT="${OUTDIR}/stage_a_${layout}.csv"
  for C in ${CORES}; do
    # Fixed-total work: this core count's per-core shard (floored so the trid pipeline stays valid).
    PAGES=$(( TOTAL_PAGES / C ))
    (( PAGES < MIN_PAGES_PER_CORE )) && PAGES=${MIN_PAGES_PER_CORE}
    # Capped input-CB candidates: only depths small enough that the reader can't hold the whole
    # shard (in_cb <= PAGES/IN_CB_CAP_FRAC), so compute can back-pressure it. Fall back to the
    # smallest candidate if the shard is too small for any to qualify.
    IN_DEPTHS_CAPPED=""
    for d in ${INPUT_DEPTHS}; do (( d * IN_CB_CAP_FRAC <= PAGES )) && IN_DEPTHS_CAPPED="${IN_DEPTHS_CAPPED} ${d}"; done
    [ -z "${IN_DEPTHS_CAPPED}" ] && IN_DEPTHS_CAPPED="$(set -- ${INPUT_DEPTHS}; echo "$1")"
    echo "==== ${layout} cores=${C}: pages/core=${PAGES} (total~$(( PAGES * C )) of ${TOTAL_PAGES}); in_cb candidates=[${IN_DEPTHS_CAPPED} ] ===="
    # Skip this (layout,cores) only if every barrier-mode row is already present.
    if [ -f "${OUT}" ]; then
      have_all=1
      for m in ${BARRIER_MODES}; do grep -q "^${layout},${C},${m}," "${OUT}" || have_all=0; done
      if [ "${have_all}" = "1" ]; then echo "[skip] ${layout} cores=${C} (all modes present)"; continue; fi
    fi
    # --- Stage 1: tune input CB depth at nops=0 (read-bound), max agg_read ---
    echo "==== ${layout} cores=${C}: tuning input CB depth (nops=0) ===="
    best_bw=0; declare -A BW_AT; declare -A PCMED_AT
    for d in ${IN_DEPTHS_CAPPED}; do
      if ! run_test "${lf}" "${C}" "${d}" "${OUT_CB_BASE}" 0 "${TUNE_PROGS}"; then
        echo "  in_cb=${d}: FAILED ($(grep -oE 'TT_FATAL: .*' /tmp/stage_a_run.log | head -1))"; continue
      fi
      bw=$(decomp_field agg_read_gbps "${C}"); pcm=$(decomp_field per_core_gbps_median "${C}")
      BW_AT[$d]="${bw:-0}"; PCMED_AT[$d]="${pcm:-0}"
      echo "  in_cb=${d}: agg_read=${bw} GB/s  per_core_med=${pcm}"
      awk -v a="${bw:-0}" -v b="${best_bw}" 'BEGIN{exit !(a>b)}' && best_bw="${bw}"
    done
    [ "${best_bw}" = "0" ] && { echo "  no successful input tune; skipping"; unset BW_AT PCMED_AT; continue; }
    best_in=""
    for d in ${IN_DEPTHS_CAPPED}; do
      [ -z "${BW_AT[$d]:-}" ] && continue
      if awk -v a="${BW_AT[$d]}" -v p="${best_bw}" -v t="${BW_TOL}" 'BEGIN{exit !(a>=p*t)}'; then best_in="${d}"; break; fi
    done
    peak_read="${BW_AT[$best_in]}"
    echo "  -> in_cb=${best_in} (read-bound peak agg_read=${peak_read})"

    # --- Balance via NOP knee: largest NOPs where agg_read still >= KNEE_TOL*peak ---
    echo "==== ${layout} cores=${C}: NOP-knee search (peak_read=${peak_read}) ===="
    nops=0
    for nq in ${NOP_SWEEP}; do
      if ! run_test "${lf}" "${C}" "${best_in}" "${OUT_CB_BASE}" "${nq}" "${TUNE_PROGS}"; then
        echo "  nops=${nq}: FAILED"; continue
      fi
      bw=$(decomp_field agg_read_gbps "${C}")
      echo "  nops=${nq}: agg_read=${bw}"
      if awk -v a="${bw:-0}" -v p="${peak_read}" -v t="${KNEE_TOL}" 'BEGIN{exit !(a>=p*t)}'; then
        nops="${nq}"
      else
        break  # past the knee: compute is gating reads
      fi
    done
    echo "  -> balanced (knee) nops=${nops}"

    # --- Stage 2: tune output CB depth in the BALANCED regime, max agg_total ---
    # (Output starvation only appears once compute is real; at high cores partial read
    #  starvation backs up the output CB and stalls compute, so deeper output can help.)
    echo "==== ${layout} cores=${C}: tuning output CB depth (balanced nops=${nops}) ===="
    best_tot=0; declare -A TOT_AT
    for o in ${OUTPUT_DEPTHS}; do
      if ! run_test "${lf}" "${C}" "${best_in}" "${o}" "${nops}" "${TUNE_PROGS}"; then
        echo "  out_cb=${o}: FAILED ($(grep -oE 'TT_FATAL: .*' /tmp/stage_a_run.log | head -1))"; continue
      fi
      tot=$(decomp_field agg_total_gbps "${C}")
      TOT_AT[$o]="${tot:-0}"
      echo "  out_cb=${o}: agg_total=${tot} GB/s"
      awk -v a="${tot:-0}" -v b="${best_tot}" 'BEGIN{exit !(a>b)}' && best_tot="${tot}"
    done
    best_out="${OUT_CB_BASE}"
    if [ "${best_tot}" != "0" ]; then
      for o in ${OUTPUT_DEPTHS}; do
        [ -z "${TOT_AT[$o]:-}" ] && continue
        if awk -v a="${TOT_AT[$o]}" -v p="${best_tot}" -v t="${BW_TOL}" 'BEGIN{exit !(a>=p*t)}'; then best_out="${o}"; break; fi
      done
    fi
    echo "  -> out_cb=${best_out} (agg_total peak=${best_tot})"

    echo "==== ${layout} cores=${C}: final balanced runs (in=${best_in} out=${best_out} nops=${nops}) barriers=[${BARRIER_MODES}] ===="
    for m in ${BARRIER_MODES}; do
      if [ -f "${OUT}" ] && grep -q "^${layout},${C},${m}," "${OUT}"; then echo "  [skip] barrier=${m}"; continue; fi
      if run_test "${lf}" "${C}" "${best_in}" "${best_out}" "${nops}" "${FINAL_PROGS}" "${m}"; then
        "${PY}" "${DEC}" --pages-per-core "${PAGES}" --num-cores "${C}" --csv-out "${OUT}" \
          --label "layout=${layout};cores=${C};barrier=${m};in_cb=${best_in};out_cb=${best_out};nops=${nops};peak_read_gbps=${peak_read}" \
          2>/dev/null | grep -E "agg_total_gbps|m2m_bubble_us|m2m_skew_env_us|finish_skew_us|start_skew_us|read_fill_head_us"
        # per-config event timeline (go -> ... -> go received; delta + cumulative cyc/us) from the
        # same device log this run just produced. Appends one block per config to a per-layout CSV.
        "${PY}" "${SCRIPT_DIR}/event_timeline.py" --input-file "${DEV_CSV}" \
          --csv-out "${OUTDIR}/stage_a_timeline_${layout}.csv" \
          --label "layout=${layout};cores=${C};barrier=${m};in_cb=${best_in};out_cb=${best_out};nops=${nops}" \
          >/dev/null 2>&1 || echo "  [timeline skipped: <2 steady ops] ${layout} cores=${C} barrier=${m}"
        echo "  appended ${layout} cores=${C} barrier=${m} -> ${OUT} (+ timeline)"
      else
        echo "  FINAL FAILED ${layout} cores=${C} barrier=${m}: $(grep -oE 'TT_FATAL: .*' /tmp/stage_a_run.log | head -1)"
      fi
    done
    unset BW_AT PCMED_AT TOT_AT
  done
  echo; echo "==== ${layout} done: ${OUT} ===="; column -t -s, "${OUT}" 2>/dev/null || cat "${OUT}"
done
echo "STAGE_A_SWEEP_COMPLETE"
