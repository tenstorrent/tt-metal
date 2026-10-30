#!/usr/bin/env bash
# Container-side environment for the Stage 11 owned server.
set -euo pipefail
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache
export XDG_CACHE_HOME=/workspace/.cache PYTHONPATH=/workspace
export TT_MODEL_BRINGUP_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21
export TT_AUTODEBUG_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
eval "$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")"
export VLLM_TT_PLUGIN_ROOT=/workspace/bringup/dependencies/vllm-tt-plugin
export TT_MODEL_CLASS_OVERRIDES='TTQwen3_5ForConditionalGeneration=models.autoports.qwen_qwen3_8_27b.tt.generator_vllm:Qwen38ForCausalLM'
export MESH_DEVICE=P300x2 HF_MODEL=/workspace/.cache/qwen38-full VLLM_RPC_TIMEOUT=1800000
export TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_DEVICE_PROFILER TT_METAL_WATCHER TT_METAL_WATCHER_DEBUG_DELAY
unset QWEN38_HOST_SAMPLING QWEN38_SERVING_LAYERS QWEN38_PRECISION_CONFIG QWEN38_SERVING_EVENTS
if [[ -n "${QWEN38_AGENTIC_EVENTS:-}" ]]; then
    export QWEN38_SERVING_EVENTS="$QWEN38_AGENTIC_EVENTS"
fi
exec /opt/venv/bin/python -m vllm.entrypoints.openai.api_server "$@"
