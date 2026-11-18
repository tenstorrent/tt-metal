#!/bin/bash
# vLLM server for Llama-3.3-70B as ONE TP32 model (models/demos/llama3_70b_galaxy) on one Blackhole Galaxy.
# Required env: TT_METAL_DIR (this repo, built, python_env with vLLM), HF_MODEL, TT_CACHE_PATH, PLUGIN_DIR (vllm-tt-plugin), OUT.
# Model switches: `source env.sh` first. Optional: MAX_BATCHED_TOKENS (4096), TRACE_REGION (280 MB; 220 MB is too small with
# 4K chunks), WORKER_L1, FABRIC. Extra args go to vLLM. Log: $OUT/server.log. Stop it with Ctrl-C / SIGINT and let it exit:
# a hard kill leaves the chips hung until a reset.
: "${TT_METAL_DIR:?}" "${HF_MODEL:?}" "${TT_CACHE_PATH:?}" "${PLUGIN_DIR:?}" "${OUT:?}"
cd "$TT_METAL_DIR" && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD MESH_DEVICE=${MESH_DEVICE:-TG} TT_LLAMA_TEXT_VER=llama3_70b_galaxy VLLM_RPC_TIMEOUT=900000
TT_CFG=$(printf '{"tt": {"dispatch_core_axis": "col", "sample_on_device_mode": "all", "fabric_config": "%s", "worker_l1_size": %s, "trace_region_size": %s}}' \
  "${FABRIC:-FABRIC_1D_RING}" "${WORKER_L1:-1345000}" "${TRACE_REGION:-280000000}")
mkdir -p "$OUT"
exec python "$PLUGIN_DIR/examples/server_example_tt.py" --model "$HF_MODEL" --served-model-name llama70b \
  --data_parallel_size 4 --max_num_seqs 8 --no-async-scheduling --port 8000 \
  --enable-prompt-tokens-details --enable-chunked-prefill --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-4096}" \
  --additional-config "$TT_CFG" "$@" > "$OUT/server.log" 2>&1
