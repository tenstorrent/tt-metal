#!/usr/bin/env bash
set -euo pipefail
out=experiments/sdpa-l2/hybrid-mixed-v1
for mode in qk4_pv2 qk4_pv2_fullsub accurate fast; do
    for chunks in 64 512; do
        for seed in 1236 1237; do
            label=distinct-$mode-normal-$chunks-$seed
            python_env/bin/python "$out/run.py" --mode "$mode" --distinct-kv \
                --q-repeats 1 --k-chunks "$chunks" --iters 0 --seed "$seed" \
                --label "$label" > "$out/$label.log" 2>&1
        done
    done
    for distribution in scaled_qk outliers common_q common_k constant_v; do
        label=distinct-$mode-$distribution-512-1236
        python_env/bin/python "$out/run.py" --mode "$mode" --distinct-kv \
            --q-repeats 1 --k-chunks 512 --iters 0 --distribution "$distribution" \
            --label "$label" > "$out/$label.log" 2>&1
    done
done
