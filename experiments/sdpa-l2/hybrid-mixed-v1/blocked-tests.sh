#!/usr/bin/env bash
set -euo pipefail
out=experiments/sdpa-l2/hybrid-mixed-v1
python_env/bin/python "$out/run.py" --mode hybrid --hybrid-block-pack --q-repeats 16 \
    --k-chunks 512 --warmup 20 --iters 10 --label blocked-hybrid-reverse-local > "$out/blocked-hybrid-reverse-local.log" 2>&1
for seed in 1236 1237 1238; do
    label=blocked-distinct-normal-512-$seed
    python_env/bin/python "$out/run.py" --mode hybrid --hybrid-block-pack --distinct-kv \
        --q-repeats 1 --k-chunks 512 --seed "$seed" --iters 0 --label "$label" > "$out/$label.log" 2>&1
done
for distribution in scaled_qk outliers common_q common_k constant_v uniform common_v; do
    label=blocked-distinct-$distribution-512-1236
    python_env/bin/python "$out/run.py" --mode hybrid --hybrid-block-pack --distinct-kv \
        --q-repeats 1 --k-chunks 512 --distribution "$distribution" --iters 0 \
        --label "$label" > "$out/$label.log" 2>&1
done
python_env/bin/python -m tracy -r --profiler-capture-perf-counters fpu \
    -o "$out/profile-hybrid-blocked" "$out/run.py" --mode hybrid --hybrid-block-pack \
    --q-repeats 8 --k-chunks 512 --iters 0 --label profile-hybrid-blocked > "$out/profile-hybrid-blocked.log" 2>&1
