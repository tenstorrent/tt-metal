#!/usr/bin/env bash
# Profile N layers (warm, kv=51200) with TT_METAL_PROFILER_MID_RUN_DUMP to drain the profiler buffer and
# avoid the deep-layer marker drop. Usage: profile_NL.sh <NLAYERS> <noservice|service> [tag]
# Disk-guarded (<3G kill). Parses per-layer op2op via parse_NL.py.
set -u
NL="${1:?usage: profile_NL.sh NLAYERS noservice|service [tag]}"
MODE="${2:?need noservice|service}"
TAG="${3:-${NL}L_$MODE}"
REPO=/home/ppopovic/tt-metal
INV=$REPO/kimi_perf_overnight
TRACE=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok
PROFDIR="${PROFDIR:-/dev/shm/ttprof}"; mkdir -p "$PROFDIR"
LOGD="$PROFDIR/.logs"; REPORTS="$PROFDIR/reports"
LOG="$INV/profile_${TAG}.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
reap(){ me=$$; for p in $(pgrep -f "runners.prefill_runner" 2>/dev/null); do [ "$p" = "$me" ] && continue; c=$(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null); case "$c" in *"python "*) kill -KILL "$p" 2>/dev/null;; esac; done; while pgrep -f "runners.prefill_runner" >/dev/null 2>&1; do sleep 3; done; rm -f /dev/shm/tt_prefill_layer_acks_ds_prefill /dev/shm/tt_d2h_* 2>/dev/null; sleep 3; }
purge_raw(){ rm -f "$LOGD/profile_log_device.csv" "$LOGD/tracy_ops_times.csv" "$LOGD/tracy_profile_log_host.tracy" 2>/dev/null; }

SVC=""; [ "$MODE" = "service" ] && SVC="PREFILL_FORCE_BUILD_SERVICE=1"
ITERS="${ITERS:-2}"; PROF="${PROF:-1}"
PROFENV="TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=${PSC:-3000} TT_METAL_PROFILER_DIR=$PROFDIR"; WRAP="python -m tracy -r -p --disable-device-data-push-to-tracy -m"
if [ "$PROF" = "0" ]; then PROFENV=""; WRAP="python -m"; fi
log "######## profile NLAYERS=$NL MODE=$MODE tag=$TAG START; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free ########"
reap; purge_raw

READ_EVERY="${READ_EVERY:-3}"
( while true; do f=$(df --output=avail "$REPO"|tail -1); big=$(ls -S "$LOGD" 2>/dev/null|head -1); echo "[disk-watch] $((f/1024))M free; biggest=$big $(du -h "$LOGD/$big" 2>/dev/null|awk '{print $1}')" >> "$LOG"; if [ "$f" -lt 838860 ]; then echo "[disk-guard] <0.8G ($((f/1024))M) KILL"|tee -a "$LOG"; for p in $(pgrep -f runners.prefill_runner); do kill -KILL "$p" 2>/dev/null; done; break; fi; sleep 2; done ) & GUARD=$!

( cd "$REPO" && env -u DEEPSEEK_V3_HF_MODEL -u TT_DS_PREFILL_TTNN_CACHE -u TT_DS_PREFILL_HOST_REF_CACHE \
    -u PREFILL_REQUEST_LOOP_PCC -u PREFILL_STANDALONE \
    $PROFENV PREFILL_PROFILE_READ_EVERY=$READ_EVERY \
    PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=$NL \
    PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
    PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill \
    PREFILL_STANDALONE_CHUNKED=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=1 \
    PREFILL_STANDALONE_CHUNKED_ITERS=$ITERS PREFILL_PROFILE_KV=51200 $SVC \
    PREFILL_STANDALONE_CHUNKED_SLOT=0 PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1 \
    DEEPSEEK_PREFILL_TRACE_DIR="$TRACE" \
    $WRAP models.demos.deepseek_v3_d_p.tt.runners.prefill_runner \
) > "$INV/profile_${TAG}.runner.log" 2>&1
log "$TAG exit=$? ; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free"
kill "$GUARD" 2>/dev/null

echo "  marker-drop warnings: $(grep -c 'markers were dropped' "$INV/profile_${TAG}.runner.log")" | tee -a "$LOG"
latest=$(ls -dt "$REPORTS"/*/ 2>/dev/null | head -1)
rep=$(ls "$latest"ops_perf_results_*.csv 2>/dev/null | head -1)
if [ -n "$rep" ] && [ -f "$rep" ]; then
  cp -f "$rep" "$INV/ops_${TAG}.csv"; log "saved ops_${TAG}.csv ($(wc -l < "$INV/ops_${TAG}.csv") rows)"
  python3 "$INV/parse_NL.py" "$INV/ops_${TAG}.csv" "$TAG warm kv=51200" "$INV/ops_${TAG}.perlayer.log" 2>&1 | tee -a "$LOG"
else
  log "WARN no ops_perf_results for '$TAG'"
fi
[ -n "$latest" ] && rm -rf "$latest" 2>/dev/null
purge_raw; reap
log "######## $TAG DONE; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free ########"
