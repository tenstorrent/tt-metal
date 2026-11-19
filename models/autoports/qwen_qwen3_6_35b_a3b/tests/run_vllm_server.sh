#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
source "$repo/python_env/bin/activate"
source "$repo/../lane-env.sh"
export VLLM_TT_PLUGIN_ROOT="$repo/../vllm-tt-plugin"
export TT_MODEL_CLASS_OVERRIDES='TTQwen3_5MoeForConditionalGeneration=models.autoports.qwen_qwen3_6_35b_a3b.tt.generator_vllm:QwenForCausalLM'
export VLLM_SERVER_DEV_MODE=1 OMP_NUM_THREADS=8
unset TT_METAL_DEVICE_PROFILER TT_METAL_WATCHER
output_dir=$1
max_sequences=$2
token_budget=$3
shift 3
async_flag=--async-scheduling
if [[ ${QWEN_VLLM_ASYNC:-1} == 0 ]]; then
    async_flag=--no-async-scheduling
fi
# Repeat the same capacity with the canonical CLI spelling: readiness uses the
# underscore alias, while benchmark evidence validation checks --max-num-seqs.
timeout "${QWEN_VLLM_SERVER_LIFETIME_SECONDS:-28800}" python -m readiness_check.run_vllm_server \
    --model-dir "$MODEL_DIR" --hf-model "$HF_MODEL" --mesh-device P300x2 \
    --max-num-seqs "$max_sequences" --max-model-len 262144 --block-size 32 \
    --server-timeout 1800 --port 8000 --sampling-profile full \
    --tt-config '{"trace_region_size":805306368,"fabric_config":"FABRIC_1D","trace_mode":"decode_only"}' \
    --additional-server-args "--max-num-seqs $max_sequences $async_flag --max-num-batched-tokens $token_budget --max-logprobs -1 --shutdown-timeout 10 --revision $HF_REVISION --tokenizer-revision $HF_REVISION --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only" \
    --output-dir "$output_dir" "$@"
