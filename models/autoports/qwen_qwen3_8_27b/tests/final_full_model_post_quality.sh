#!/usr/bin/env bash
# Run through optimized_multichip_env.sh after CPU reference work has finished.
# B32 already has clean final measurements; the B1-only geometry change cannot
# enter its prefill branch. Each measured batch starts a fresh process/model.
set -euo pipefail
base=models/autoports/qwen_qwen3_8_27b/doc/full_model
module=models.autoports.qwen_qwen3_8_27b.tests
for batch in 1 8 16; do
    context=32768
    extra=()
    if [[ "$batch" == 1 ]]; then
        context=262144
        extra=(--qualitative-control-tokens 1024 --verify-native-context --teacher-forcing-control)
    fi
    env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_DEBUG_DELAY -u TT_METAL_DEVICE_PROFILER -u TT_METAL_TRACE_ALLOC_TRACEBACKS -u TT_METAL_TRACE_ALLOC_REFERRER_DEPTH TT_METAL_TRACE_ALLOC_TRACKING=0 \
        python -m "$module.benchmark_full_model" --batch "$batch" --context "$context" --lengths 4096 \
        --outputs 128 --repeats 3 --memory-audit "${extra[@]}" --output "$base/benchmark_b${batch}_perf.json" \
        > "$base/benchmark_b${batch}_perf.log" 2>&1
done
