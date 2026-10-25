#!/usr/bin/env bash
# Resume after completed tracked batches and repaired sparse-map probes.
set -euo pipefail
base=models/autoports/qwen_qwen3_8_27b/doc/full_model
module=models.autoports.qwen_qwen3_8_27b.tests
for batch in 1 32 8 16; do
    context=32768
    lengths=(4096)
    extra=()
    if [[ "$batch" == 1 ]]; then
        context=262144
        extra=(--qualitative-control-tokens 512 --verify-native-context)
    elif [[ "$batch" == 32 ]]; then
        context=33024
        lengths=(4096 32768)
    fi
    env -u TT_METAL_WATCHER -u TT_METAL_WATCHER_DEBUG_DELAY -u TT_METAL_DEVICE_PROFILER -u TT_METAL_TRACE_ALLOC_TRACEBACKS -u TT_METAL_TRACE_ALLOC_REFERRER_DEPTH TT_METAL_TRACE_ALLOC_TRACKING=0 \
        python -m "$module.benchmark_full_model" --batch "$batch" --context "$context" --lengths "${lengths[@]}" \
        --outputs 128 --repeats 3 --memory-audit "${extra[@]}" --output "$base/benchmark_b${batch}_perf.json" \
        > "$base/benchmark_b${batch}_perf.log" 2>&1
done
for batch in 1 32; do
    bash models/autoports/qwen_qwen3_8_27b/tests/profile_full_model_reduced.sh "$batch" > "$base/profile_b${batch}.log" 2>&1
done
python -m "$module.probe_external_prepare" --full-stack --batch 32 --context 33024 --output "$base/external_prepare_b32.json" > "$base/external_prepare_b32.log" 2>&1
python -m "$module.run_full_validation" > "$base/full_validation_04.log" 2>&1
