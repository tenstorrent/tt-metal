#!/usr/bin/env bash
set -euo pipefail
out=experiments/sdpa-l2/hifi2-fp32-resident-v1
for mode in fp32_hifi2_cheap fp32_hifi2; do
    python_env/bin/python -m tracy -r --profiler-capture-perf-counters fpu \
        -o "$out/profile-$mode" "$out/run.py" --mode "$mode" \
        --q-repeats 8 --k-chunks 512 --iters 0 --label "profile-$mode" \
        > "$out/profile-$mode.log" 2>&1
done
