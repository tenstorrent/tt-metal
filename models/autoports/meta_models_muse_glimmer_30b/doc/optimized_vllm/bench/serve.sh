#!/usr/bin/env bash
# The optimized-vLLM stage's canonical server invocation.
#
# It is character-for-character the vLLM-integration stage's TT config -- same
# mesh, same fabric, same l1_small_size, same trace_region_size, same
# trace_mode, same max_model_len, same max_num_seqs, same greedy benchmark
# temperature -- because the before/after comparison this stage exists to make is
# only meaningful if the harness does not move.  The single thing that differs
# between the arms is the code under test, selected by
# MUSE_GLIMMER_VLLM_PREFILL_TRACE (unset/0 = the vLLM-integration behaviour,
# 1 = this stage's traced serving prefill).
#
#   serve.sh hold                 # launch and hold open (iterate from another shell)
#   serve.sh full                 # launch + sampling(full) + qualitative + benchmark + shutdown
#   serve.sh smoke                # launch + sampling(smoke) + shutdown
#   serve.sh checks <stage,...>   # attach stages to an already-running server
#
# Env knobs: MAX_NUM_SEQS, MAX_MODEL_LEN, PORT, SAMPLING_PROFILE, OUT_DIR,
#   EXTRA_SERVER_ARGS, MUSE_GLIMMER_VLLM_PREFILL_TRACE,
#   MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS.
set -euo pipefail

REPO=/home/ttuser/dev/muse-glimmer/tt-metal
MODEL_DIR=models/autoports/meta_models_muse_glimmer_30b
HF_MODEL=meta-models/Muse-Glimmer-30B
MESH=P300x2
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
PORT=${PORT:-8000}
TT_CONFIG='{"trace_region_size": 400000000, "fabric_config": "FABRIC_1D_RING", "fabric_packet_payload_bytes": 8192, "l1_small_size": 6144, "trace_mode": "decode_only"}'
OUT_DIR=${OUT_DIR:-$REPO/$MODEL_DIR/readiness_vllm}
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}

cd "$REPO"
mkdir -p "$OUT_DIR"
mode=${1:-full}

case "$mode" in
  hold)
    exec python -m models.common.readiness_check.run_vllm_server \
      --stages serve \
      --model-dir "$MODEL_DIR" --hf-model "$HF_MODEL" --mesh-device "$MESH" \
      --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" --port "$PORT" \
      --server-timeout 2400 --output-dir "$OUT_DIR" \
      ${EXTRA_SERVER_ARGS:+--additional-server-args=$EXTRA_SERVER_ARGS} \
      --tt-config "$TT_CONFIG"
    ;;
  full|smoke)
    profile=full; [ "$mode" = smoke ] && profile=smoke
    exec python -m models.common.readiness_check.run_vllm_server \
      --model-dir "$MODEL_DIR" --hf-model "$HF_MODEL" --mesh-device "$MESH" \
      --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" --port "$PORT" \
      --sampling-profile "$profile" --server-timeout 2400 --output-dir "$OUT_DIR" \
      ${EXTRA_SERVER_ARGS:+--additional-server-args=$EXTRA_SERVER_ARGS} \
      --tt-config "$TT_CONFIG"
    ;;
  checks)
    stages=${2:?"usage: serve.sh checks <serve-less stage list>"}
    exec python -m models.common.readiness_check.run_vllm_server \
      --stages "$stages" \
      --server-url "http://localhost:$PORT" \
      --model-dir "$MODEL_DIR" --hf-model "$HF_MODEL" --output-dir "$OUT_DIR" \
      --max-num-seqs "$MAX_NUM_SEQS" --sampling-profile "${SAMPLING_PROFILE:-full}"
    ;;
  *)
    echo "usage: serve.sh {hold|full|smoke|checks <stages>}" >&2
    exit 2
    ;;
esac
