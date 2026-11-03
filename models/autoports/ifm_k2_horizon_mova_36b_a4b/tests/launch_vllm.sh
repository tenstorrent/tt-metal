#!/usr/bin/env bash
set -euo pipefail
source python_env/bin/activate
source ../lane-env.sh
export VLLM_TT_PLUGIN_ROOT="$RUN_ROOT/vllm-tt-plugin"
export TT_MODEL_CLASS_OVERRIDES='TTK2HorizonForCausalLM=models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.generator_vllm:K2HorizonForCausalLM'
export K2_VLLM_HOST_COMPAT="${K2_VLLM_HOST_COMPAT:-1}"
exec python -m readiness_check.run_vllm_server --stages serve --model-dir "$MODEL_DIR" --hf-model "$HF_MODEL" --mesh-device P300x2 --max-num-seqs "${K2_VLLM_BATCH:-4}" --max-model-len 524288 --sampling-profile smoke --server-timeout 1800 --tt-config '{"trace_region_size":0,"fabric_config":"FABRIC_1D_RING","trace_mode":"all"}' --additional-server-args "--trust-remote-code --revision $HF_REVISION --tokenizer-revision $HF_REVISION --max-num-batched-tokens 256" "$@"
