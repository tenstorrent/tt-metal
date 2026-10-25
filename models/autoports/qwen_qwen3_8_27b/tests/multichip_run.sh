#!/usr/bin/env bash
set -euo pipefail
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache PYTHONPATH=/workspace
export TT_MODEL_BRINGUP_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21
export TT_AUTODEBUG_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
eval "$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")"
export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
label=$1; shift
base=models/autoports/qwen_qwen3_8_27b/doc/multichip_decoder
python -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder --output "$base/$label.json" "$@"
