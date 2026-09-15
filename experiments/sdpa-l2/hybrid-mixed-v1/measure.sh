#!/usr/bin/env bash
set -euo pipefail
out=experiments/sdpa-l2/hybrid-mixed-v1
for mode in hybrid fast accurate fp32_hifi2_cheap qk4_pv2 qk4_pv2_fullsub; do
    label=final-$mode
    python_env/bin/python "$out/run.py" --mode "$mode" --q-repeats 16 --k-chunks 512 \
        --warmup 20 --iters 10 --label "$label" > "$out/$label.log" 2>&1
done
for chunks in 64 512; do
    for seed in 1236 1237; do
        label=distinct-hybrid-normal-$chunks-$seed
        python_env/bin/python "$out/run.py" --mode hybrid --distinct-kv --q-repeats 1 \
            --k-chunks "$chunks" --seed "$seed" --iters 0 --label "$label" > "$out/$label.log" 2>&1
    done
done
for distribution in scaled_qk outliers common_q common_k constant_v; do
    label=distinct-hybrid-$distribution-512-1236
    python_env/bin/python "$out/run.py" --mode hybrid --distinct-kv --q-repeats 1 \
        --k-chunks 512 --distribution "$distribution" --iters 0 --label "$label" > "$out/$label.log" 2>&1
done
