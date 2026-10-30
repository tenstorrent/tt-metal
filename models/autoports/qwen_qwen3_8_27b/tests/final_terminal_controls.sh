#!/usr/bin/env bash
set -euo pipefail
base=models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model
python -m models.autoports.qwen_qwen3_8_27b.tests.probe_head_geometry --generic --generic-chunks 2 --block 16 --output "$base/head_generic_split2_k16.json" > "$base/head_generic_split2_k16.log" 2>&1
python -m models.autoports.qwen_qwen3_8_27b.tests.probe_head_geometry --generic --generic-chunks 4 --block 20 --subblock 1 --output "$base/head_generic_split4_k20.json" > "$base/head_generic_split4_k20.log" 2>&1
for batch in 8 16; do
    for mode in baseline async; do
        args=()
        if [[ "$mode" == async ]]; then args+=(--async-ccl); fi
        name=sampling_${mode}_b${batch}
        python -m models.autoports.qwen_qwen3_8_27b.tests.probe_sampling --batch "$batch" "${args[@]}" --output "$base/$name.json" > "$base/$name.log" 2>&1
        echo "SAMPLER_CANDIDATE_DONE $name"
    done
done
