#!/usr/bin/env bash
set -euo pipefail
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache PYTHONPATH=/workspace
export TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
unset TT_METAL_WATCHER
label=$1; shift
base=models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder
python -m tracy -r -p -v --no-web-server -o "$base/tracy/$label" -m models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder --seq 4096 --steps 128 --profile --output "$base/$label.json" "$@"
python -m models.autoports.qwen_qwen3_8_27b.tests.collect_multichip_perf "$base/tracy/$label"
