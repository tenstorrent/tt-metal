#!/usr/bin/env bash
set -euo pipefail
# Run only after the isolated timing queue finishes; profiling changes kernels.
export TT_SDPA_BLOCK_SWEEP=1
sha256sum -c experiments/sdpa-l2/fp32-block-perf/SOURCE-SHA256.txt
for entry in 4:65536 4:131072 1:131072; do
    export TT_SDPA_ACCURACY_DIAG=${entry%%:*}
    n=${entry##*:}
    label=profile-mode${TT_SDPA_ACCURACY_DIAG}-n${n}-q128-k1024
    out=experiments/sdpa-l2/fp32-block-perf/${label}
    if [[ -e "${out}.jsonl" ]]; then
        echo "Refusing to overwrite ${out}.jsonl" >&2
        exit 1
    fi
    python_env/bin/python -m tracy -r --profiler-capture-perf-counters fpu -o "$out" \
        tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
        --kv-lens "$n" --full --heads 10 --q-chunk 128 --k-chunks 1024 \
        --variants fp32_hifi2 --seed 1236 --q-len 128 --query-sampling spread \
        --benchmark-iters 0 --per-head-metrics --label "$label" \
        --output "${out}.jsonl" > "${out}.log" 2>&1
done
