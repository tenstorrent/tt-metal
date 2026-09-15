#!/usr/bin/env bash
set -euo pipefail
label=${1:?Provide old or new}
out=experiments/sdpa-l2/bf16-denom-pair-v3
for length in 32768 65536; do
    for distribution in normal scaled_qk; do
        name=causal-${label}-${length}-${distribution}
        test ! -e "$out/$name.jsonl"
        python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
            --kv-lens "$length" --causal --heads 2 --q-len 128 --query-sampling spread \
            --q-chunk 128 --k-chunks 512 --variants hifi2 --seed 1236 \
            --distribution "$distribution" --check-full-output \
            --label "$name" --output "$out/$name.jsonl" > "$out/$name.log" 2>&1
    done
done
