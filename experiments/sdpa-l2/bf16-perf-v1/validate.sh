#!/usr/bin/env bash
set -euo pipefail
# Run from the built worktree with PYTHONPATH, ARCH_NAME, and TT_METAL_HOME set.
label=${1:?provide a fresh label}
repro=tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py
out=experiments/sdpa-l2/bf16-perf-v1
case_id=0
run() {
    case_id=$((case_id + 1))
    target="$out/$label-accuracy-$case_id.jsonl"
    if [[ -e "$target" ]]; then
        echo "Refusing to overwrite $target" >&2
        exit 1
    fi
    python_env/bin/python "$repro" "$@" --label "$label" --output "$target"
}
base=(--kv-lens 262144 --full --sampled-device --heads 10 --variants hifi2
      --q-chunk 128 --k-chunks 512 --benchmark-iters 2)
for seed in 1234 1235 1236; do
    run "${base[@]}" --seed "$seed" --q-round-bits 6 --q-bitceil --q-prescale 1.0027 --max-l2-pct 3.3
done
for distribution in normal scaled_qk outliers biased_v uniform constant_v uniform_constant_v; do
    run "${base[@]}" --seed 1236 --query-sampling spread --distribution "$distribution"
done
for distribution in common_q common_k common_v; do
    for offset in 8 32; do
        run "${base[@]}" --seed 1236 --query-sampling spread --distribution "$distribution" --common-mode "$offset"
    done
done
run --kv-lens 32768 65536 131072 --full --sampled-device --heads 10 --variants hifi2 \
    --q-chunk 128 --k-chunks 512 --seed 1236 --benchmark-iters 2
run --kv-lens 32768 --causal --query-sampling spread --heads 4 --variants hifi2 \
    --q-chunk 128 --k-chunks 512 --seed 1236 --benchmark-iters 2
run --kv-lens 4096 --dim 64 --heads 1 --variants hifi2 --seed 1236 --benchmark-iters 2
run --kv-lens 262144 --full --sampled-device --heads 10 --variants fp32_hifi2 \
    --q-chunk 128 --k-chunks 1024 --q-round-bits 6 --q-bitceil --q-prescale 1.0027 \
    --seed 1236 --benchmark-iters 2 --record-fp32-streaming --max-l2-pct 0.5
