#!/usr/bin/env bash
# Profile the MLA random-weight pytest (test_mla_profile.py) WITH or WITHOUT the H2D stream service,
# over PREFILL_MLA_ITERS warm runs (+1 compile run, which is SKIPPED by the parser via the
# mla_run_{it}_start/_end signposts). Produces:
#   - per-RUN table: kernel_ms + op2op (raw signed AND positive-only) for each warm run (compile skipped)
#   - mla_test_<mode>_log.new   per-op breakdown of the last warm run (same format as 9L_*_log.new)
#   - tracy_mlatest_<mode>.tracy   raw tracy capture for manual inspection (Tracy GUI)
#
# Usage: profile_mla_test.sh <noservice|service> [synced|pipelined] [tag]
#   synced    (default): re-push input + ttnn.synchronize_device() after every forward (runs isolated;
#                        each run's first-op op2op carries the inter-run sync/host-upload boundary).
#   pipelined          : input pushed ONCE, NO synchronize between forwards -> back-to-back issue,
#                        dispatcher runs ahead (true steady-state op2op, no boundary gap).
# Env overrides: PREFILL_MLA_ITERS (default 10), PREFILL_MLA_SEQ_LEN (5120),
#                PSC (profiler program-support count, default 3000), PROFDIR (default /dev/shm/ttprofmla).
#
# Direct (no-script) equivalent:
#   TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=3000 TT_METAL_PROFILER_DIR=/dev/shm/ttprofmla \
#   KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized PREFILL_MLA_ITERS=10 PREFILL_MLA_SEQ_LEN=5120 \
#   python -m tracy -r -p --disable-device-data-push-to-tracy -m pytest -svv \
#     'models/demos/deepseek_v3_d_p/tests/test_mla_profile.py::test_kimi_mla_profile[blackhole-kimi-service-pipelined-mesh-8x4]'
set -u
MODE="${1:?usage: profile_mla_test.sh noservice|service [synced|pipelined] [tag]}"
case "$MODE" in noservice|service) :;; *) echo "MODE must be noservice|service"; exit 2;; esac
RUNMODE="${2:-synced}"
case "$RUNMODE" in synced|pipelined) :;; *) echo "RUNMODE must be synced|pipelined"; exit 2;; esac
TAG="${3:-mlatest_${MODE}_${RUNMODE}}"
NODEID="blackhole-kimi-${MODE}-${RUNMODE}-mesh-8x4"
REPO=/home/ppopovic/tt-metal
INV=$REPO/kimi_perf_overnight
PROFDIR="${PROFDIR:-/dev/shm/ttprofmla}"; mkdir -p "$PROFDIR"
LOGD="$PROFDIR/.logs"; REPORTS="$PROFDIR/reports"
LOG="$INV/profile_${TAG}.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

reap(){
  me=$$
  for p in $(pgrep -x python; pgrep -x python3); do
    [ "$p" = "$me" ] && continue
    if tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | grep -q "test_mla_profile"; then kill -KILL "$p" 2>/dev/null; fi
  done
  while :; do
    alive=0
    for p in $(pgrep -x python; pgrep -x python3); do
      tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | grep -q "test_mla_profile" && alive=1
    done
    [ "$alive" = 0 ] && break; sleep 3
  done
  rm -f /dev/shm/tt_prefill_layer_acks_ds_prefill /dev/shm/tt_d2h_* /dev/shm/tt_h2d_stream_service_ds_prefill* 2>/dev/null; sleep 2
}
purge_raw(){ rm -f "$LOGD/profile_log_device.csv" "$LOGD/tracy_ops_times.csv" "$LOGD/tracy_profile_log_host.tracy" 2>/dev/null; }

PSC="${PSC:-3000}"
log "######## profile MLA pytest MODE=$MODE tag=$TAG START; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free ########"
reap; purge_raw

( while true; do f=$(df --output=avail "$REPO"|tail -1);
    if [ "$f" -lt 838860 ]; then echo "[disk-guard] <0.8G ($((f/1024))M) KILL"|tee -a "$LOG";
      for p in $(pgrep -x python; pgrep -x python3); do tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | grep -q test_mla_profile && kill -KILL "$p" 2>/dev/null; done; break; fi
    echo "[disk-watch] $((f/1024))M free" >> "$LOG"; sleep 4; done ) & GUARD=$!

( cd "$REPO" && \
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=$PSC TT_METAL_PROFILER_DIR=$PROFDIR \
  KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized \
  TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
  TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden \
  TT_DS_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
  TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden \
  PREFILL_MLA_ITERS="${PREFILL_MLA_ITERS:-10}" \
  PREFILL_MLA_SEQ_LEN="${PREFILL_MLA_SEQ_LEN:-5120}" \
  python -m tracy -r -p --disable-device-data-push-to-tracy -m pytest -svv \
    "models/demos/deepseek_v3_d_p/tests/test_mla_profile.py::test_kimi_mla_profile[${NODEID}]" \
) > "$INV/profile_${TAG}.runner.log" 2>&1
RC=$?
log "$TAG exit=$RC ; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free"
kill "$GUARD" 2>/dev/null

echo "  marker-drop warnings: $(grep -c 'markers were dropped' "$INV/profile_${TAG}.runner.log")" | tee -a "$LOG"
if [ -f "$LOGD/tracy_profile_log_host.tracy" ]; then
  cp -f "$LOGD/tracy_profile_log_host.tracy" "$INV/tracy_${TAG}.tracy"; log "saved raw tracy -> tracy_${TAG}.tracy"
else
  log "WARN no tracy_profile_log_host.tracy in $LOGD"
fi
latest=$(ls -dt "$REPORTS"/*/ 2>/dev/null | head -1)
rep=$(ls "$latest"ops_perf_results_*.csv 2>/dev/null | head -1)
if [ -n "$rep" ] && [ -f "$rep" ]; then
  cp -f "$rep" "$INV/ops_${TAG}.csv"; log "saved ops_${TAG}.csv ($(wc -l < "$INV/ops_${TAG}.csv") rows)"
  echo "------------------------------------------------------------" | tee -a "$LOG"
  python3 "$INV/parse_runs.py" "$INV/ops_${TAG}.csv" "MLA $MODE/$RUNMODE (random weights, seq=${PREFILL_MLA_SEQ_LEN:-5120})" \
    "$INV/mla_test_${MODE}_${RUNMODE}_log.new" 2>&1 | tee -a "$LOG"
  echo "------------------------------------------------------------" | tee -a "$LOG"
else
  log "WARN no ops_perf_results CSV for '$TAG' (rc=$RC) — check profile_${TAG}.runner.log"
fi
[ -n "$latest" ] && rm -rf "$latest" 2>/dev/null
purge_raw; reap
log "######## $TAG DONE; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free ########"
