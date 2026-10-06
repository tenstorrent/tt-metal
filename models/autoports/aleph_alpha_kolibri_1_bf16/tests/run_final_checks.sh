#!/usr/bin/env bash
# Targeted final review follow-ups. Run with exclusive ownership of chip 0.
set -euo pipefail
EVIDENCE=models/autoports/aleph_alpha_kolibri_1_bf16/doc/functional_decoder
MODULE=models.autoports.aleph_alpha_kolibri_1_bf16.tests
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
for layer in 0 4; do
  TT_METAL_TRACE_ALLOC_TRACKING=1 python -m "$MODULE.run_decoder" --layer "$layer" --length 3329 --capacity 4096 --batch 2 --public --output "batch2_${layer}_final.json" > "$EVIDENCE/batch2_${layer}_final.log" 2>&1
done
# Identical inputs and timing instrumentation for the archived and repaired code.
for layer in 0 4; do
  baseline="$EVIDENCE/before_numerical_fix/untracked_perf"
  mkdir -p "$baseline"
  TT_METAL_TRACE_ALLOC_TRACKING=0 python -m tracy -r -p -v --no-web-server -o "$PWD/$baseline/tracy/layer_${layer}/raw" -m "$MODULE.run_profile" --layer "$layer" --baseline > "$baseline/profile_${layer}.log" 2>&1
  TT_METAL_TRACE_ALLOC_TRACKING=0 python -m tracy -r -p -v --no-web-server -o "$PWD/$EVIDENCE/tracy/layer_${layer}/raw" -m "$MODULE.run_profile" --layer "$layer" > "$EVIDENCE/profile_${layer}.log" 2>&1
done
python -m "$MODULE.report_performance" > "$EVIDENCE/performance_report.log" 2>&1
