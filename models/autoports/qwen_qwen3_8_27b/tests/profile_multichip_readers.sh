#!/usr/bin/env bash
set -euo pipefail
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache PYTHONPATH=/workspace
export TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
export TT_MODEL_BRINGUP_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21
export TT_AUTODEBUG_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
eval "$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")"
unset TT_METAL_WATCHER
base=models/autoports/qwen_qwen3_8_27b
export QWEN_OPT_POLICY="${QWEN_OPT_POLICY:-$(cat "$base/doc/optimized_multichip_decoder/candidate_policy.json")}"
for layer in 0 3; do
 label=readers_${1:-g80}_l${layer}
 python -m tracy -r -p -v --no-web-server -o "$base/doc/optimized_multichip_decoder/tracy/$label" -m models.autoports.qwen_qwen3_8_27b.tests.multichip_reader_microbenchmark --layer "$layer" --profile --output "$base/doc/optimized_multichip_decoder/$label.json"
 python "$base/tests/collect_multichip_reader_micro.py" "$base/doc/optimized_multichip_decoder/tracy/$label" "$base/doc/optimized_multichip_decoder/$label.json"
done
