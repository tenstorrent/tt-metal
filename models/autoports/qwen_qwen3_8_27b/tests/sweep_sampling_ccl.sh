#!/usr/bin/env bash
set -euo pipefail
base=models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model
for batch in 1 32; do
    for mode in baseline async persistent; do
        args=()
        if [[ "$mode" == async ]]; then args+=(--async-ccl); fi
        if [[ "$mode" == persistent ]]; then args+=(--persistent); fi
        name=sampling_${mode}_b${batch}_v2
        python -m models.autoports.qwen_qwen3_8_27b.tests.probe_sampling --batch "$batch" "${args[@]}" --output "$base/$name.json" > "$base/$name.log" 2>&1
        echo "SAMPLER_CANDIDATE_DONE $name"
    done
done
