#!/usr/bin/env bash
# Run inside the validated build container, after selection and quality gates.
set -euo pipefail
root=models/autoports/qwen_qwen3_8_27b
out=$root/doc/datatype_sweep/selected
test -f "$root/doc/datatype_sweep/selected_precision_config.json"
mkdir -p "$out"
unset QWEN38_PRECISION_CONFIG
for batch in 1 32 8 16; do
    context=32768
    lengths=(4096)
    extra=()
    if [[ $batch == 1 ]]; then
        context=262144
        extra=(--verify-native-context)
    elif [[ $batch == 32 ]]; then
        context=33024
        lengths=(4096 32768)
        extra=(--long-repeats 1)
    fi
    bash "$root/tests/optimized_multichip_env.sh" \
        env TT_METAL_TRACE_ALLOC_TRACKING=0 python -m models.autoports.qwen_qwen3_8_27b.tests.benchmark_full_model \
        --batch "$batch" --context "$context" --lengths "${lengths[@]}" \
        --outputs 128 --repeats 3 --buffered-output --memory-audit --precision-evidence \
        "${extra[@]}" --output "$out/benchmark_b${batch}_perf.json" \
        > "$out/benchmark_b${batch}_perf.log" 2>&1
    echo "SELECTED_BENCHMARK_COMPLETE batch=$batch"
done
