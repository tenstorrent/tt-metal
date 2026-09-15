#!/usr/bin/env bash
set -euo pipefail
build=${1:?main or improved}
shift
[[ "$build" == main || "$build" == improved ]]
for seed in "$@"; do
    for mode in bf16 fp32; do
        out="experiments/sdpa-l2/requested-lengths/$build-$mode-$seed"
        [[ ! -e "$out.jsonl" ]] || { echo "Refusing to overwrite $out.jsonl" >&2; exit 1; }
        variant=hifi2
        extra=()
        if [[ "$mode" == fp32 ]]; then
            variant=fp32_hifi2
            extra=(--no-record-fp32-streaming)
            if [[ "$build" == improved ]]; then
                extra+=(--q-round-bits 6 --q-bitceil --q-prescale 1.0027)
            fi
        fi
        python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
            --kv-lens 25920 75600 --full --heads 5 --dim 128 --distribution normal \
            --variants "$variant" --q-chunk 128 --k-chunks 512 --seed "$seed" \
            --benchmark-iters 2 "${extra[@]}" --label "$build-$mode-$seed" \
            --output "$out.jsonl" > "$out.log" 2>&1
    done
done
