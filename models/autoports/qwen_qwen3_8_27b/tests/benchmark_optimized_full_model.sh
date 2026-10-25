#!/usr/bin/env bash
# Run inside the existing source-built container. Hardware jobs are serialized.
set -euo pipefail
root=models/autoports/qwen_qwen3_8_27b
out=$root/doc/optimized_full_model
for batch in 1 32 8 16; do
    context=32768
    extra=()
    lengths=(4096)
    if [[ $batch == 1 ]]; then
        context=262144
        extra+=(--teacher-forcing-control --eos-control)
    elif [[ $batch == 32 ]]; then
        context=33024
        lengths+=(32768)
        extra+=(--long-repeats 1)
    fi
    bash "$root/tests/optimized_multichip_env.sh" env TT_METAL_TRACE_ALLOC_TRACKING=0 \
        python -m models.autoports.qwen_qwen3_8_27b.tests.benchmark_full_model \
        --batch "$batch" --context "$context" --lengths "${lengths[@]}" --outputs 128 \
        --repeats 3 --buffered-output --memory-audit "${extra[@]}" \
        --output "$out/benchmark_b${batch}_perf.json" > "$out/benchmark_b${batch}_perf.log" 2>&1
    echo "BENCHMARK_COMPLETE batch=$batch"
done
