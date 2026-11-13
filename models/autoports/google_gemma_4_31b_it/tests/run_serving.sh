#!/usr/bin/env bash
set -euo pipefail
# Run from this execution checkout; all runtime sources and caches stay local.
: "${TT_MODEL_BRINGUP_ROOT:?Select the enabled model-bringup package}"
: "${TT_AUTODEBUG_ROOT:?Select the enabled AutoDebug package}"
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export TT_MODEL_CLASS_OVERRIDES='TTGemma4ForConditionalGeneration=models.autoports.google_gemma_4_31b_it.tt.generator_vllm:Gemma4ForCausalLM,TTGemma4ForCausalLM=models.autoports.google_gemma_4_31b_it.tt.generator_vllm:Gemma4ForCausalLM'
export TT_METAL_CACHE="$PWD/bringup/artifacts/tt-metal-cache"
export XDG_CACHE_HOME="$PWD/bringup/artifacts/xdg-cache"
export VLLM_SERVER_DEV_MODE=1
HF_SNAPSHOT=$(python_env/bin/python models/autoports/google_gemma_4_31b_it/tools/serving_paths.py snapshot)
export VLLM_DEBUG_LOG_API_SERVER_RESPONSE=false
exec timeout 7200 python_env/bin/python -m readiness_check.run_vllm_server \
  --model-dir models/autoports/google_gemma_4_31b_it \
  --hf-model "$HF_SNAPSHOT" --mesh-device P300x2 \
  --max-model-len 262144 --max-num-seqs 1 --block-size 32 \
  --server-timeout 1800 --sampling-profile smoke \
  --benchmark-prompt-len 4096 --benchmark-output-len 128 \
  --tt-config '{"trace_region_size":134217728,"fabric_config":"FABRIC_1D_RING"}' \
  --additional-server-args "--revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --tokenizer-revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --language-model-only --middleware models.autoports.google_gemma_4_31b_it.tt.response_logging.ResponseLoggingMiddleware --async-scheduling --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --override-generation-config '{\"top_k\":32}' --served-model-name google/gemma-4-31B-it $HF_SNAPSHOT" \
  "$@"
