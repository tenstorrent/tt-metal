#!/usr/bin/env bash
set -euo pipefail
# Instrumented runs are for stages/activity, never production timing.
source experiments/sdpa-l2/fp32-pipeline-v2/candidate-env.sh
out=experiments/sdpa-l2/fp32-pipeline-v2
prefix=${1:-stage-l1-v2}
for stage in 1 2; do
    export TT_SDPA_UTIL_PROFILE=$stage
    label=${prefix}-s${stage}-n65536
    if [[ -e "$out/$label.jsonl" ]]; then
        echo "Refusing to overwrite $label" >&2
        exit 1
    fi
    python_env/bin/python -m tracy -r --enable-sum-profiling \
        --profiler-capture-perf-counters fpu -o "$out/$label" \
        tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
        --kv-lens 65536 --full --heads 10 --q-chunk 128 --k-chunks 1024 \
        --variants fp32_hifi2 --seed 1236 --q-len 32 --query-sampling spread \
        --benchmark-iters 0 --label "$label" --output "$out/$label.jsonl" \
        > "$out/$label.log" 2>&1
done
