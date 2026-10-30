#!/usr/bin/env bash
set -euo pipefail
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache PYTHONPATH=/workspace
export TT_MODEL_BRINGUP_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21
export TT_AUTODEBUG_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
eval "$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")"
export QWEN_OPT_POLICY='{}' TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
for layer in 0 3; do
 for width in 3 5; do
  export QWEN_SUBBLOCK_WIDTH=$width
  label=final_subblock_width${width}_l${layer}
  python -m models.autoports.qwen_qwen3_8_27b.tests.probe_prefill_subblocks --output models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder/${label}.json --layer "$layer" --seq 4096 --recorded --timing --trace-prefill > models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder/${label}.log 2>&1
 done
done
