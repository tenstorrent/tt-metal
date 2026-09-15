#!/usr/bin/env bash
set -euo pipefail
# Run from the built repository with TT_METAL_HOME, ARCH_NAME, and PYTHONPATH set.
label=${1:?provide a label}
repro=tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py
out=experiments/sdpa-l2/perf-v4/${label}-accuracy
case_id=0
run() {
    case_id=$((case_id + 1))
    python_env/bin/python "$repro" "$@" --output "$out-$case_id.jsonl"
}
base=(--kv-lens 262144 --full --sampled-device --heads 10 --variants fp32_hifi2 --k-chunks 1024 --record-fp32-streaming
      --q-round-bits 6 --q-prescale 1.0027 --q-bitceil --benchmark-iters 2
      --label "$label")
for seed in 1234 1235 1236; do
    run "${base[@]}" --seed "$seed" --max-l2-pct 0.5
done
for distribution in normal scaled_qk outliers biased_v uniform constant_v uniform_constant_v; do
    run "${base[@]}" --seed 1236 --query-sampling spread --distribution "$distribution"
done
for distribution in common_q common_k common_v; do
    for offset in 8 32; do
        run "${base[@]}" --seed 1236 --query-sampling spread --distribution "$distribution" --common-mode "$offset"
    done
done
# Unchanged BF16 streaming path; no preprocessing.
run --kv-lens 262144 --full --sampled-device --query-sampling spread --heads 10 --variants hifi2 --seed 1236 --benchmark-iters 2 --label "$label-bf16"
# Prior accurate FP32 path outside the new noncausal/geometry guard.
run --kv-lens 32768 --causal --query-sampling spread --heads 4 --variants fp32_hifi2 --q-round-bits 6 --q-prescale 1.0027 --q-bitceil --seed 1236 --benchmark-iters 2 --label "$label-causal-fallback"
run --kv-lens 4096 --dim 64 --heads 1 --variants fp32_hifi2 --q-round-bits 6 --q-prescale 1.0027 --q-bitceil --seed 1236 --benchmark-iters 2 --label "$label-d64-fallback"
# Other lengths inside the new path's activation guard.
run --kv-lens 32768 65536 131072 --full --sampled-device --heads 10 --variants fp32_hifi2 --q-round-bits 6 --q-prescale 1.0027 --q-bitceil --seed 1236 --benchmark-iters 2 --max-l2-pct 0.5 --k-chunks 1024 --record-fp32-streaming --label "$label-lengths"
# The other enabled streaming chunk geometry.
for seed in 1234 1235 1236; do
    run --kv-lens 262144 --full --sampled-device --heads 10 --variants fp32_hifi2 --q-round-bits 6 --q-prescale 1.0027 --q-bitceil --seed "$seed" --benchmark-iters 2 --max-l2-pct 0.5 --k-chunks 512 --record-fp32-streaming --label "$label-k512"
done
