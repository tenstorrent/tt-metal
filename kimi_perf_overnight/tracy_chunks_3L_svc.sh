#!/usr/bin/env bash
# Tracy profile of the FIRST 3 LAYERS (layer0 dense + layers1,2 MoE for Kimi), 1st chunk (kv=0) vs last
# chunk (kv=51200), WARM (ITERS=2). No H2D service. Captures the NAMED ops_perf_results report (OP CODE +
# DEVICE ID) per case and purges the multi-GB raw logs immediately (disk near-full).
set -u
REPO=/home/ppopovic/tt-metal
INV=/home/ppopovic/kimi_perf_overnight
TRACE=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok
LOGD="$REPO/generated/profiler/.logs"
REPORTS="$REPO/generated/profiler/reports"
LOG="$INV/tracy_chunks_3Lsvc.log"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
reap(){
  pids=$(pgrep -f "runners.prefill_runner" || true); [ -n "$pids" ] && kill -KILL $pids 2>/dev/null
  while pgrep -f "runners.prefill_runner" >/dev/null 2>&1; do sleep 3; done
  rm -f /dev/shm/tt_prefill_layer_acks_ds_prefill /dev/shm/tt_h2d_stream_service_ds_prefill* 2>/dev/null
  sleep 4
}
purge_raw(){ rm -f "$LOGD/profile_log_device.csv" "$LOGD/tracy_ops_times.csv" "$LOGD/tracy_profile_log_host.tracy" 2>/dev/null; }

run_case(){ # name kv
  local name="$1" kv="$2"
  log "=== 3Lsvc profile '$name' (kv=$kv) START; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free ==="
  reap; purge_raw
  ( cd "$REPO" && env -u DEEPSEEK_V3_HF_MODEL -u TT_DS_PREFILL_TTNN_CACHE -u TT_DS_PREFILL_HOST_REF_CACHE \
      -u PREFILL_REQUEST_LOOP_PCC -u PREFILL_STANDALONE \
      TT_METAL_DEVICE_PROFILER=1 \
      PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=3 \
      PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
      PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill \
      PREFILL_STANDALONE_CHUNKED=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=1 \
      PREFILL_STANDALONE_CHUNKED_ITERS=2 PREFILL_PROFILE_KV=$kv PREFILL_FORCE_BUILD_SERVICE=1 \
      PREFILL_STANDALONE_CHUNKED_SLOT=0 PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1 \
      DEEPSEEK_PREFILL_TRACE_DIR="$TRACE" \
      python -m tracy -r -p -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner \
  ) > "$INV/tracy_3Lsvc_$name.runner.log" 2>&1
  log "3Lsvc '$name' exit=$?"
  # grab the newest named ops report
  latest=$(ls -dt "$REPORTS"/*/ 2>/dev/null | head -1)
  rep=$(ls "$latest"ops_perf_results_*.csv 2>/dev/null | head -1)
  if [ -n "$rep" ] && [ -f "$rep" ]; then cp -f "$rep" "$INV/ops_3Lsvc_$name.csv"; log "saved ops_3Lsvc_$name.csv ($(wc -l < "$INV/ops_3Lsvc_$name.csv") rows) from $latest"; else log "WARN no ops_perf_results for '$name'"; fi
  # free the report dir + raw logs right away to bound disk on this near-full box
  [ -n "$latest" ] && rm -rf "$latest" 2>/dev/null
  purge_raw; reap
  log "=== 3Lsvc '$name' DONE; disk $(df -h "$REPO"|tail -1|awk '{print $4}') free ==="
}

log "######## 3-layer WITH-SERVICE chunk profiling start (NUM_LAYERS=3, warm) ########"
run_case first 0
run_case last 51200
log "######## 3-layer WITH-SERVICE chunk profiling complete ########"
