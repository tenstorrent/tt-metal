#!/usr/bin/env bash
set -euo pipefail
label=${1:?Provide an old/new result prefix}
out=experiments/sdpa-l2/bf16-denom-pair-v3
for length in 32768 262144; do
    for seed in 1236 1237; do
        for distribution in normal scaled_qk outliers common_q common_k common_v constant_v uniform; do
            name=${label}-${length}-${seed}-${distribution}
            test ! -e "$out/$name.jsonl"
            python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
                --kv-lens "$length" --full --sampled-device --heads 2 \
                --q-len 128 --query-sampling spread --q-chunk 128 --k-chunks 512 \
                --variants hifi2 --seed "$seed" --distribution "$distribution" \
                --label "$name" --output "$out/$name.jsonl" > "$out/$name.log" 2>&1
        done
    done
done
