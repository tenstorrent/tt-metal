#!/usr/bin/env bash
# Run inside the existing container through optimized_multichip_env.sh.
# Requires exclusive use of all four devices; stops at the first failed gate.
set -euo pipefail
base=models/autoports/qwen_qwen3_8_27b/doc/full_model
module=models.autoports.qwen_qwen3_8_27b.tests
python -m "$module.probe_mixed_slots" > "$base/probe_mixed_slots_05.log" 2>&1
python -m "$module.probe_trace_contract" > "$base/probe_contract_06.log" 2>&1
env TT_METAL_TRACE_ALLOC_TRACKING=0 python -m "$module.probe_memory_audit" > "$base/memory_probe.log" 2>&1
for batch in 1 32; do
    env TT_METAL_TRACE_ALLOC_TRACKING=0 python -m "$module.probe_sampling" --batch "$batch" > "$base/probe_sampling_b${batch}_03.log" 2>&1
done
env TT_METAL_TRACE_ALLOC_TRACKING=0 python -m "$module.probe_terminal_packing" --batch 32 > "$base/probe_terminal_b32_03.log" 2>&1
python -m "$module.run_full_validation" > "$base/full_validation_03.log" 2>&1
for batch in 8 32 16; do
    context=32768
    lengths=(4096)
    repeats=1
    if [[ "$batch" == 8 ]]; then repeats=3; fi
    if [[ "$batch" == 32 ]]; then context=33024; lengths=(4096 32768); fi
    python -m "$module.benchmark_full_model" --batch "$batch" --context "$context" --lengths "${lengths[@]}" --outputs 128 --repeats "$repeats" --output "$base/benchmark_b${batch}_tracking_final.json" > "$base/benchmark_b${batch}_tracking_final.log" 2>&1
done
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
