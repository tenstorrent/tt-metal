#!/usr/bin/env bash
# Tracy profile of the H2D-service slowdown at NUM_LAYERS=1 (fast). Runs the standalone-chunked runner
# under tracy twice — WITHOUT the service and WITH it (PREFILL_FORCE_BUILD_SERVICE=1) — and saves each
# tracy_ops_data.csv. Diff per-op DEVICE time vs inter-op dispatch GAPS to pin the service's cost.
set -u
REPO=/home/ppopovic/tt-metal
INV=/home/ppopovic/kimi_perf_overnight
TRACE=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok
CSV="$REPO/generated/profiler/.logs/tracy_ops_data.csv"
LOG="$INV/tracy_profile.log"
NCHUNKS=3

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

reap(){
  for p in "runners.prefill_runner" "runners.prefill_h2d_producer"; do
    pids=$(pgrep -f "$p" || true); [ -n "$pids" ] && kill -KILL $pids 2>/dev/null
  done
  while pgrep -f "runners.prefill_runner|runners.prefill_h2d_producer" >/dev/null 2>&1; do sleep 3; done
  rm -f /dev/shm/tt_prefill_layer_acks_ds_prefill /dev/shm/tt_h2d_stream_service_ds_prefill* 2>/dev/null
  sleep 5
}

run_case(){ # name  extra_env
  local name="$1" extra="$2"
  log "=== tracy case '$name' (extra='$extra') START ==="
  reap
  rm -f "$CSV"
  ( cd "$REPO" && env -u DEEPSEEK_V3_HF_MODEL -u TT_DS_PREFILL_TTNN_CACHE -u TT_DS_PREFILL_HOST_REF_CACHE \
      -u PREFILL_REQUEST_LOOP_PCC -u PREFILL_STANDALONE \
      TT_METAL_DEVICE_PROFILER=1 \
      PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=1 \
      PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
      PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill \
      PREFILL_STANDALONE_CHUNKED=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=$NCHUNKS \
      PREFILL_STANDALONE_CHUNKED_ITERS=3 \
      PREFILL_STANDALONE_CHUNKED_SLOT=0 PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1 \
      PREFILL_PREFILL_SYNC=1 \
      DEEPSEEK_PREFILL_TRACE_DIR="$TRACE" \
      $extra \
      python -m tracy -r -p -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner \
  ) > "$INV/tracy_$name.runner.log" 2>&1
  log "tracy case '$name' exit=$?"
  if [ -f "$CSV" ]; then
    cp -f "$CSV" "$INV/tracy_$name.csv"
    log "saved $INV/tracy_$name.csv ($(wc -l < "$INV/tracy_$name.csv") rows)"
  else
    log "WARN: no tracy CSV produced for '$name' (check tracy_$name.runner.log)"
  fi
  # The per-op device perf report (DEVICE FW DURATION + OP TO OP LATENCY) — the artifact we actually need.
  DREP="$REPO/generated/profiler/.logs/cpp_device_perf_report.csv"
  if [ -f "$DREP" ]; then
    cp -f "$DREP" "$INV/devperf_$name.csv"
    log "saved $INV/devperf_$name.csv ($(wc -l < "$INV/devperf_$name.csv") rows)"
  else
    log "WARN: no cpp_device_perf_report.csv for '$name'"
  fi
  reap
  log "=== tracy case '$name' DONE ==="
}

log "######## tracy profiling start (NUM_LAYERS=1, NCHUNKS=$NCHUNKS) ########"
run_case noservice ""
run_case service "PREFILL_FORCE_BUILD_SERVICE=1"
log "######## tracy profiling complete ########"
