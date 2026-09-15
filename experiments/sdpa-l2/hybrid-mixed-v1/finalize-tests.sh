#!/usr/bin/env bash
set -euo pipefail
out=experiments/sdpa-l2/hybrid-mixed-v1
for mode in hybrid qk4_pv2_fullsub qk4_pv2 fp32_hifi2_cheap accurate fast hybrid; do
    suffix=steady
    if [[ -f "$out/qualified-$mode-$suffix.json" ]]; then suffix=reverse; fi
    label=qualified-$mode-$suffix
    python_env/bin/python "$out/run.py" --mode "$mode" --q-repeats 16 --k-chunks 512 \
        --warmup 20 --iters 10 --label "$label" > "$out/$label.log" 2>&1
done
for chunks in 64 512; do
    for seed in 1236 1237; do
        label=qualified-distinct-hybrid-normal-$chunks-$seed
        python_env/bin/python "$out/run.py" --mode hybrid --distinct-kv --q-repeats 1 \
            --k-chunks "$chunks" --seed "$seed" --iters 0 --label "$label" > "$out/$label.log" 2>&1
    done
done
for distribution in scaled_qk outliers common_q common_k constant_v; do
    label=qualified-distinct-hybrid-$distribution-512-1236
    python_env/bin/python "$out/run.py" --mode hybrid --distinct-kv --q-repeats 1 \
        --k-chunks 512 --distribution "$distribution" --iters 0 --label "$label" > "$out/$label.log" 2>&1
done
for mode in hybrid qk4_pv2; do
    python_env/bin/python -m tracy -r --profiler-capture-perf-counters fpu \
        -o "$out/profile-$mode" "$out/run.py" --mode "$mode" \
        --q-repeats 8 --k-chunks 512 --iters 0 --label "profile-$mode" \
        > "$out/profile-$mode.log" 2>&1
done
