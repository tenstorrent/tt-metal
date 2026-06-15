#!/usr/bin/env bash
# Profile the DISPATCH CORES to attribute the ~370us op-to-op gap. 1 layer, WARM (ITERS=2), last-chunk
# shape (kv=51200). --profile-dispatch-cores populates DISPATCH TOTAL CQ CMD OP TIME + DISPATCH GO SEND
# WAIT TIME per op. Saves the named report as ops_dispatch.csv; purges multi-GB raw logs after.
set -u
REPO=/home/ppopovic/tt-metal
INV=/home/ppopovic/tt-metal/kimi_perf_overnight
TRACE=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok
LOGD="$REPO/generated/profiler/.logs"; REPORTS="$REPO/generated/profiler/reports"
LOG="$INV/tracy_dispatch.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
purge(){ rm -f "$LOGD"/{profile_log_device.csv,tracy_ops_times.csv,tracy_profile_log_host.tracy} 2>/dev/null; }
reap(){ pids=$(pgrep -f runners.prefill_runner||true); [ -n "$pids" ]&&kill -KILL $pids 2>/dev/null; while pgrep -f runners.prefill_runner>/dev/null 2>&1; do sleep 3; done; rm -f /dev/shm/tt_prefill_layer_acks_ds_prefill /dev/shm/tt_h2d_stream_service_ds_prefill* 2>/dev/null; sleep 3; }

log "######## dispatch-core profiling start (1 layer, warm, kv=51200) ########"
reap; purge; rm -rf "$REPORTS"/*/ 2>/dev/null
( cd "$REPO" && env -u DEEPSEEK_V3_HF_MODEL -u TT_DS_PREFILL_TTNN_CACHE -u TT_DS_PREFILL_HOST_REF_CACHE \
    -u PREFILL_REQUEST_LOOP_PCC -u PREFILL_STANDALONE \
    TT_METAL_DEVICE_PROFILER=1 \
    PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=1 \
    PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
    PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill \
    PREFILL_STANDALONE_CHUNKED=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=1 \
    PREFILL_STANDALONE_CHUNKED_ITERS=2 PREFILL_PROFILE_KV=51200 \
    PREFILL_STANDALONE_CHUNKED_SLOT=0 PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1 \
    DEEPSEEK_PREFILL_TRACE_DIR="$TRACE" \
    python -m tracy -r -p --profile-dispatch-cores -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner \
) > "$INV/tracy_dispatch.runner.log" 2>&1
log "exit=$?"
latest=$(ls -dt "$REPORTS"/*/ 2>/dev/null | head -1); rep=$(ls "$latest"ops_perf_results_*.csv 2>/dev/null|head -1)
[ -n "$rep" ]&&{ cp -f "$rep" "$INV/ops_dispatch.csv"; log "saved ops_dispatch.csv ($(wc -l <"$INV/ops_dispatch.csv") rows)"; }||log "WARN no report"
[ -n "$latest" ]&&rm -rf "$latest" 2>/dev/null
purge; reap
log "######## done; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free ########"
