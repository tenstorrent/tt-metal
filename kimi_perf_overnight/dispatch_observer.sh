#!/usr/bin/env bash
# OBSERVER-EFFECT test: is the ~370us/op gap real or caused by TT_METAL_DEVICE_PROFILER?
# Run the SAME 1-layer warm chunk (PROFILE_KV=51200, ITERS=6) as PLAIN python (no tracy) twice:
#   OFF = device profiler disabled (production-like)   ON = TT_METAL_DEVICE_PROFILER=1
# Compare warm-pass wall ms ([profile-kv-timing]). If ON >> OFF, the gaps are observer effect.
set -u
REPO=/home/ppopovic/tt-metal
INV=/home/ppopovic/tt-metal/kimi_perf_overnight
TRACE=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320/vllm-kimi-k26-b783c42e-56321tok
LOG="$INV/dispatch_observer.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
reap(){ pids=$(pgrep -f runners.prefill_runner||true); [ -n "$pids" ]&&kill -KILL $pids 2>/dev/null; while pgrep -f runners.prefill_runner>/dev/null 2>&1; do sleep 3; done; rm -f /dev/shm/tt_prefill_layer_acks_ds_prefill 2>/dev/null; sleep 3; }

run(){ # name  profiler_env
  local name="$1" penv="$2"
  log "=== observer '$name' START ($penv) ==="; reap
  ( cd "$REPO" && env -u DEEPSEEK_V3_HF_MODEL -u TT_DS_PREFILL_TTNN_CACHE -u TT_DS_PREFILL_HOST_REF_CACHE \
      -u PREFILL_REQUEST_LOOP_PCC -u PREFILL_STANDALONE -u TT_METAL_DEVICE_PROFILER \
      $penv \
      PREFILL_MODEL_VARIANT=kimi_k2_6 PREFILL_SP=8 PREFILL_TP=4 PREFILL_NUM_LAYERS=1 \
      PREFILL_MAX_SEQ_LEN=61440 PREFILL_CHUNK_SIZE=5120 PREFILL_NUM_USERS=1 \
      PREFILL_IS_BALANCED=0 PREFILL_H2D_SERVICE_ID=ds_prefill \
      PREFILL_STANDALONE_CHUNKED=1 PREFILL_STANDALONE_CHUNKED_NCHUNKS=1 \
      PREFILL_STANDALONE_CHUNKED_ITERS=6 PREFILL_PROFILE_KV=51200 \
      PREFILL_STANDALONE_CHUNKED_SLOT=0 PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1 \
      DEEPSEEK_PREFILL_TRACE_DIR="$TRACE" \
      python -m models.demos.deepseek_v3_d_p.tt.runners.prefill_runner \
  ) > "$INV/observer_$name.log" 2>&1
  log "observer '$name' exit=$?"
  echo "  per-pass wall ms ($name):"; grep "profile-kv-timing" "$INV/observer_$name.log" | sed -E 's/.*pass /pass /'
  reap; log "=== observer '$name' DONE ==="
}
log "######## observer-effect test (1 layer, kv=51200, 6 passes) ########"
run OFF ""
run ON  "TT_METAL_DEVICE_PROFILER=1"
log "######## done ########"
