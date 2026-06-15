#!/usr/bin/env bash
# Tracy profile of ONE layer's 1st chunk (kv_actual=0, logical_n=5120) vs LAST chunk (kv_actual=51200,
# logical_n=56320), WARM (ITERS=2: pass1 compiles, pass2 measured). No H2D service. Deletes the multi-GB
# tracy raw logs immediately after extracting the small per-op device report (disk is near-full).
set -u
REPO=/home/ppopovic/tt-metal
INV=/home/ppopovic/kimi_perf_overnight
TRACE=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok
LOGD="$REPO/generated/profiler/.logs"
DREP="$LOGD/cpp_device_perf_report.csv"
ODATA="$LOGD/tracy_ops_data.csv"
LOG="$INV/tracy_chunks.log"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
reap(){
  for p in "runners.prefill_runner"; do pids=$(pgrep -f "$p" || true); [ -n "$pids" ] && kill -KILL $pids 2>/dev/null; done
  while pgrep -f "runners.prefill_runner" >/dev/null 2>&1; do sleep 3; done
  rm -f /dev/shm/tt_prefill_layer_acks_ds_prefill /dev/shm/tt_h2d_stream_service_ds_prefill* 2>/dev/null
  sleep 4
}
purge_raw(){ rm -f "$LOGD/profile_log_device.csv" "$LOGD/tracy_ops_times.csv" "$LOGD/tracy_profile_log_host.tracy" 2>/dev/null; }

run_case(){ # name  kv_actual
  local name="$1" kv="$2"
  log "=== chunk profile '$name' (kv_actual=$kv) START ==="
  reap; purge_raw; rm -f "$DREP" "$ODATA"
  ( cd "$REPO" && env -u DEEPSEEK_V3_HF_MODEL -u TT_DS_PREFILL_TTNN_CACHE -u TT_DS_PREFILL_HOST_REF_CACHE \
      -u PREFILL_REQUEST_LOOP_PCC -u PREFILL_STANDALONE \
      TT_METAL_DEVICE_PROFILER=1 \
      PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=1 \
      PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
      PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill \
      PREFILL_STANDALONE_CHUNKED=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=1 \
      PREFILL_STANDALONE_CHUNKED_ITERS=2 PREFILL_PROFILE_KV=$kv \
      PREFILL_STANDALONE_CHUNKED_SLOT=0 PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1 \
      DEEPSEEK_PREFILL_TRACE_DIR="$TRACE" \
      python -m tracy -r -p -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner \
  ) > "$INV/tracy_chunk_$name.runner.log" 2>&1
  log "chunk profile '$name' exit=$?"
  [ -f "$DREP" ] && { cp -f "$DREP" "$INV/devperf_chunk_$name.csv"; log "saved devperf_chunk_$name.csv ($(wc -l < "$INV/devperf_chunk_$name.csv") rows)"; } || log "WARN no device report for '$name'"
  [ -f "$ODATA" ] && cp -f "$ODATA" "$INV/tracy_chunk_$name.opsdata.csv"
  purge_raw  # free the multi-GB raw logs right away
  reap
  log "=== chunk profile '$name' DONE; disk: $(df -h "$REPO" | tail -1 | awk '{print $4" free"}') ==="
}

log "######## chunk profiling start (1 layer, warm) ########"
run_case first 0
run_case last 51200
log "######## chunk profiling complete ########"
