#!/usr/bin/env bash
# The stage's canonical vLLM server invocation, in one place so the work log, the
# README and every rerun quote the same command.
#
#   serve.sh hold                 # launch and hold open (for iterating from another shell)
#   serve.sh full                 # launch + sampling(full) + qualitative + benchmark + shutdown
#   serve.sh smoke                # launch + sampling(smoke) + shutdown, for the inner loop
#   serve.sh checks <stage,...>   # attach the named stages to an already-running server
#
# Env knobs: MAX_NUM_SEQS, MAX_MODEL_LEN, PORT, SAMPLING_PROFILE, and
#   OUT_DIR            artifact directory (default <model_dir>/readiness_vllm).  Point a
#                      variant run at a different directory so it cannot overwrite the
#                      stage's committed evidence.
#   EXTRA_SERVER_ARGS  passed through as --additional-server-args, e.g.
#                      EXTRA_SERVER_ARGS=--async-scheduling bash serve.sh hold
#                      Passed in the --flag=value form on purpose: argparse treats a
#                      separate value that begins with '--' as another option and
#                      fails with "expected one argument".
#
# TT config, and why each key is here rather than defaulted:
#   trace_region_size 400000000   the model decode trace plus the sampler's over a
#                                 52-layer stack (tt/generator.py DEFAULT_TRACE_REGION_SIZE)
#   fabric_config FABRIC_1D_RING  the 1x4 ring the collectives were measured on
#                                 (doc/context_contract.json device.ccl_topology)
#   fabric_packet_payload_bytes   8192, the router payload the decode collective was
#                                 tuned at; without it serving opens a different fabric
#                                 from the one every earlier stage measured
#   l1_small_size 6144            holds the per-program CCL global semaphores.  A margin
#                                 choice, not pass/fail: 32768 and 8192 fail, but 7168,
#                                 6144 and 4096 all pass; 6144 clears 24 distinct CCL
#                                 programs with 1,152 B of margin (context_contract.json
#                                 device.l1_small_note).  Carried from the decoder stage (device.l1_small_note)
#   trace_mode decode_only        decode is traced; prefill is eager on purpose -- this
#                                 port's prefill graph is keyed by padded prompt length,
#                                 so there is no single prefill graph to capture
#   sample_on_device_mode all     enforced by the runner; stated here for the record
set -euo pipefail

REPO=/home/ttuser/dev/muse-glimmer/tt-metal
MODEL_DIR=models/autoports/meta_models_muse_glimmer_30b
HF_MODEL=meta-models/Muse-Glimmer-30B
MESH=P300x2
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
PORT=${PORT:-8000}
TT_CONFIG='{"trace_region_size": 400000000, "fabric_config": "FABRIC_1D_RING", "fabric_packet_payload_bytes": 8192, "l1_small_size": 6144, "trace_mode": "decode_only"}'
LOGS=$REPO/$MODEL_DIR/doc/vllm_integration/logs
OUT_DIR=${OUT_DIR:-$REPO/$MODEL_DIR/readiness_vllm}
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}

cd "$REPO"
mkdir -p "$LOGS"
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
