#!/usr/bin/env bash
# Profile ALL 61 layers (warm, large chunk kv=51200) for op2op + kernel per layer on the slowest device.
# One mode per invocation: $1 = "noservice" | "service".  Disk-guarded (kills run if free < 3G).
# Drops the Tracy GUI push (--disable-device-data-push-to-tracy) to bound the host .tracy file on this
# 99%-full box; keeps -r -p so ops_perf_results_*.csv is still generated.
set -u
MODE="${1:?usage: profile_61L.sh noservice|service}"
REPO=/home/ppopovic/tt-metal
INV=$REPO/kimi_perf_overnight
TRACE=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok
LOGD="$REPO/generated/profiler/.logs"
REPORTS="$REPO/generated/profiler/reports"
LOG="$INV/profile_61L_$MODE.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
reap(){ me=$$; for p in $(pgrep -f "runners.prefill_runner" 2>/dev/null); do [ "$p" = "$me" ] && continue; c=$(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null); case "$c" in *"python "*) kill -KILL "$p" 2>/dev/null;; esac; done; while pgrep -f "runners.prefill_runner" >/dev/null 2>&1; do sleep 3; done; rm -f /dev/shm/tt_prefill_layer_acks_ds_prefill /dev/shm/tt_d2h_* 2>/dev/null; sleep 3; }
purge_raw(){ rm -f "$LOGD/profile_log_device.csv" "$LOGD/tracy_ops_times.csv" "$LOGD/tracy_profile_log_host.tracy" 2>/dev/null; }

SVC=""; [ "$MODE" = "service" ] && SVC="PREFILL_FORCE_BUILD_SERVICE=1"
log "######## 61-LAYER profile MODE=$MODE START; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free ########"
reap; purge_raw

# disk guard: kill the runner if free space drops below 3G
( while true; do
    free_kb=$(df --output=avail "$REPO" | tail -1)
    if [ "$free_kb" -lt 3145728 ]; then
      echo "[disk-guard] free < 3G ($((free_kb/1024))M) -> KILLING run" | tee -a "$LOG"
      for p in $(pgrep -f "runners.prefill_runner"); do kill -KILL "$p" 2>/dev/null; done
      break
    fi
    sleep 8
  done ) &
GUARD=$!

( cd "$REPO" && env -u DEEPSEEK_V3_HF_MODEL -u TT_DS_PREFILL_TTNN_CACHE -u TT_DS_PREFILL_HOST_REF_CACHE \
    -u PREFILL_REQUEST_LOOP_PCC -u PREFILL_STANDALONE \
    TT_METAL_DEVICE_PROFILER=1 \
    PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=61 \
    PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
    PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill \
    PREFILL_STANDALONE_CHUNKED=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=1 \
    PREFILL_STANDALONE_CHUNKED_ITERS=2 PREFILL_PROFILE_KV=51200 $SVC \
    PREFILL_STANDALONE_CHUNKED_SLOT=0 PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1 \
    DEEPSEEK_PREFILL_TRACE_DIR="$TRACE" \
    python -m tracy -r -p --disable-device-data-push-to-tracy -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner \
) > "$INV/profile_61L_$MODE.runner.log" 2>&1
log "61L $MODE exit=$? ; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free"
kill "$GUARD" 2>/dev/null

latest=$(ls -dt "$REPORTS"/*/ 2>/dev/null | head -1)
rep=$(ls "$latest"ops_perf_results_*.csv 2>/dev/null | head -1)
if [ -n "$rep" ] && [ -f "$rep" ]; then
  cp -f "$rep" "$INV/ops_61L_$MODE.csv"
  log "saved ops_61L_$MODE.csv ($(wc -l < "$INV/ops_61L_$MODE.csv") rows)"
  python3 "$INV/parse_NL.py" "$INV/ops_61L_$MODE.csv" "61L $MODE warm kv=51200" "$INV/ops_61L_$MODE.perlayer.log" 2>&1 | tee -a "$LOG"
else
  log "WARN no ops_perf_results for '$MODE'"
fi
[ -n "$latest" ] && rm -rf "$latest" 2>/dev/null
purge_raw; reap
log "######## 61L $MODE DONE; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free ########"
