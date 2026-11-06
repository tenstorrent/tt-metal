#!/usr/bin/env bash
set -euo pipefail
export TT_MODEL_BRINGUP_ROOT=/home/mvasiljevic/.codex-gemma4-v3/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23
export TT_AUTODEBUG_ROOT=/home/mvasiljevic/.codex-gemma4-v3/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.9
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export VLLM_TT_PLUGIN_ROOT=/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09
export TT_MODEL_CLASS_OVERRIDES='TTGemma4ForConditionalGeneration=models.autoports.google_gemma_4_31b_it.tt.generator_vllm:Gemma4ForCausalLM,TTGemma4ForCausalLM=models.autoports.google_gemma_4_31b_it.tt.generator_vllm:Gemma4ForCausalLM'
export HF_HUB_CACHE="$PWD/bringup/artifacts/cache-env-validation/weights/hub"
export TT_METAL_CACHE="$PWD/bringup/artifacts/cache-env-validation/tt-metal-jit"
export XDG_CACHE_HOME="$PWD/bringup/artifacts/xdg-cache"
export VLLM_SERVER_DEV_MODE=1
CACHE_TEST_SNAPSHOT=/home/bringup/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475
timeout 900 python_env/bin/python -m readiness_check.run_vllm_server \
  --model-dir models/autoports/google_gemma_4_31b_it \
  --hf-model "$CACHE_TEST_SNAPSHOT" --mesh-device P300x2 \
  --max-model-len 262144 --max-num-seqs 8 --block-size 32 \
  --server-timeout 600 --sampling-profile smoke --stages serve \
  --benchmark-prompt-len 4096 --benchmark-output-len 128 \
  --output-dir models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/ci-startup-analysis/local-server \
  --tt-config '{"trace_region_size":134217728,"fabric_config":"FABRIC_1D_RING"}' \
  --additional-server-args "--revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --tokenizer-revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --language-model-only --async-scheduling --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --override-generation-config '{\"top_k\":32}' --served-model-name google/gemma-4-31B-it $CACHE_TEST_SNAPSHOT"
