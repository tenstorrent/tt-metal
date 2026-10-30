#!/usr/bin/env bash
# Run inside qwen38-functional-build; pass shared readiness runner arguments.
set -euo pipefail
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache
export XDG_CACHE_HOME=/workspace/.cache PYTHONPATH=/workspace
export TT_MODEL_BRINGUP_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21
export TT_AUTODEBUG_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
eval "$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")"
export VLLM_TT_PLUGIN_ROOT=/workspace/bringup/dependencies/vllm-tt-plugin
export TT_MODEL_CLASS_OVERRIDES='TTQwen3_5ForConditionalGeneration=models.autoports.qwen_qwen3_8_27b.tt.generator_vllm:Qwen38ForCausalLM'
export TT_METAL_TRACE_ALLOC_TRACKING=${TT_METAL_TRACE_ALLOC_TRACKING:-1}
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_DEVICE_PROFILER TT_METAL_WATCHER TT_METAL_WATCHER_DEBUG_DELAY
exec python -m readiness_check.run_vllm_server \
  --model-dir models/autoports/qwen_qwen3_8_27b --hf-model /workspace/.cache/qwen38-full \
  --mesh-device P300x2 --max-model-len 262144 --block-size 4096 \
  --server-timeout 1800 --benchmark-prompt-len 4096 \
  --tt-config '{"trace_region_size":134217728,"fabric_config":"FABRIC_1D_RING","trace_mode":"decode_only"}' \
  --additional-server-args='--language-model-only --mamba-cache-mode align --mamba-block-size 4096 --max-num-batched-tokens 8192 --max-logprobs -1 --async-scheduling' \
  "$@"
