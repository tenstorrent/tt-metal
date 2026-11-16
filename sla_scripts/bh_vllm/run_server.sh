#!/bin/bash
# vLLM server for Llama-3.3-70B, TP8/DP4 (models/ttt_compat) on one Galaxy.
# Needs tt-metal python_env with vLLM installed (vllm-tt-plugin docs/install-vllm-tt.sh).
# Set TT_METAL_DIR, HF_MODEL, TT_CACHE_PATH, OUT to match this host.
: "${TT_METAL_DIR:?}" "${HF_MODEL:?}" "${TT_CACHE_PATH:?}" "${OUT:?}" "${PLUGIN_DIR:?}"
cd "$TT_METAL_DIR" && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export MESH_DEVICE=${MESH_DEVICE:-TG} TT_LLAMA_TEXT_VER=tt_transformers
export VLLM_RPC_TIMEOUT=900000 TORCHDYNAMO_DISABLE=1 TT_DP_DISCOVERY_JOIN_TIMEOUT_S=${TT_DP_DISCOVERY_JOIN_TIMEOUT_S:-60}
mkdir -p "$OUT"
python "$PLUGIN_DIR/examples/server_example_tt.py" --model "$HF_MODEL" --served-model-name llama70b \
  --data_parallel_size 4 --max_num_seqs 32 --max-model-len 36864 --block-size 64 \
  --max-num-batched-tokens 8192 --enable-prompt-tokens-details --port 8000 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 200000000}}' \
  "$@" > "$OUT/server.log" 2>&1
