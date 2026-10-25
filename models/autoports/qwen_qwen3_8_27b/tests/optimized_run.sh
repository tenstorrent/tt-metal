#!/usr/bin/env bash
set -euo pipefail
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache
export PYTHONPATH=/workspace TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export TT_MODEL_BRINGUP_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21
export TT_AUTODEBUG_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
eval "$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")"
mode=$1; label=$2; shift 2
base=models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder
if [[ $mode == profile ]]; then
  export TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
  unset TT_METAL_WATCHER
  python -m tracy -r -p -v --no-web-server -o "$base/tracy/$label" -m models.autoports.qwen_qwen3_8_27b.tests.run_optimized_decoder --seq 4096 --profile --output "$base/$label.json" "$@"
  report_args=()
  if [[ " $* " == *" --decode-profile-only "* ]]; then report_args+=(--decode-only); fi
  python -m models.autoports.qwen_qwen3_8_27b.tests.collect_optimized_perf "$base/tracy/$label" "${report_args[@]}"
else
  export TT_METAL_TRACE_ALLOC_TRACKING=1
  python -m models.autoports.qwen_qwen3_8_27b.tests.run_optimized_decoder --output "$base/$label.json" "$@"
fi
