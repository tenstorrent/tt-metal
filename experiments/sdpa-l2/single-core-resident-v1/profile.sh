#!/usr/bin/env bash
set -euo pipefail
out=experiments/sdpa-l2/single-core-resident-v1
prefix=${1:?Provide a fresh label}
for resident_mode in main fast accurate; do
    label=${prefix}-${resident_mode}
    test ! -e "$out/$label.json"
    python_env/bin/python -m tracy -r --profiler-capture-perf-counters fpu \
        -o "$out/$label" "$out/run.py" --mode "$resident_mode" \
        --q-repeats 8 --k-chunks 512 --iters 0 --label "$label" \
        > "$out/$label.log" 2>&1
done
