#!/usr/bin/env bash
set -euo pipefail
unset FUSIONS FUSION_IMPL FUSION_TAG TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
MODULE=models.autoports.aleph_alpha_kolibri_1_bf16.tests
EVIDENCE=models/autoports/aleph_alpha_kolibri_1_bf16/doc/fused_decoder
for layer in 0 4; do
  TT_METAL_TRACE_ALLOC_TRACKING=0 FUSION_TAG=reference python -m "$MODULE.fused_profile" --layer "$layer" --baseline --repetitions 100 > "$EVIDENCE/baseline_final_${layer}.log" 2>&1
  TT_METAL_TRACE_ALLOC_TRACKING=0 FUSION_TAG=final python -m "$MODULE.fused_profile" --layer "$layer" --repetitions 100 > "$EVIDENCE/final_profile_${layer}.log" 2>&1
  TT_METAL_TRACE_ALLOC_TRACKING=0 FUSION_TAG=profiled python -m tracy -r -p -v --no-web-server -o "$PWD/$EVIDENCE/baseline/tracy/layer_${layer}/raw" -m "$MODULE.fused_profile" --layer "$layer" --baseline --repetitions 3 > "$EVIDENCE/baseline_tracy_${layer}.log" 2>&1
  TT_METAL_TRACE_ALLOC_TRACKING=0 FUSION_TAG=profiled python -m tracy -r -p -v --no-web-server -o "$PWD/$EVIDENCE/tracy/layer_${layer}/raw" -m "$MODULE.fused_profile" --layer "$layer" --repetitions 3 > "$EVIDENCE/tracy_${layer}.log" 2>&1
  echo "timing and profiler layer=$layer passed"
done
