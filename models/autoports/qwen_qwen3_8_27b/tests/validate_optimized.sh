#!/usr/bin/env bash
# Run serially in qwen38-functional-build; consumes final default policy.
set -euo pipefail
export USER=mvasiljev TT_METAL_HOME=/workspace TT_METAL_CACHE=/workspace/.cache/tt-metal-cache PYTHONPATH=/workspace
export TT_MODEL_BRINGUP_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21
export TT_AUTODEBUG_ROOT=/workspace/.codex/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
eval "$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")"
export QWEN_OPT_POLICY='{}' TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_CPP_POST_PROCESS TT_METAL_PROFILER_TRACE_TRACKING
base=models/autoports/qwen_qwen3_8_27b
export TT_METAL_WATCHER=10
python -m pytest -q "$base/tests/test_optimized_decoder.py"
python -m models.autoports.qwen_qwen3_8_27b.tests.ragged_optimized_decode
for layer in 0 3; do
 bash "$base/tests/optimized_run.sh" check "watcher_l${layer}" --layer "$layer" --seq 4096 --recorded --workload --trace-prefill
done
unset TT_METAL_WATCHER
for batch in 8 16 32; do
 for layer in 0 3; do
  bash "$base/tests/optimized_run.sh" check "workload_b${batch}_l${layer}" --layer "$layer" --batch "$batch" --seq 4096 --recorded --workload --timing
 done
done
for layer in 0 3; do
 bash "$base/tests/optimized_run.sh" check "context262144_l${layer}" --layer "$layer" --seq 262144 --recorded --prefill-only
 bash "$base/tests/optimized_run.sh" check "context262142_l${layer}" --layer "$layer" --seq 262142 --recorded
done
