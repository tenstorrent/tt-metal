#!/usr/bin/env bash
set -euo pipefail
out=experiments/sdpa-l2/hifi2-fp32-resident-v1
for mode in fp32_hifi2_cheap fp32_hifi2 fast accurate; do
    python_env/bin/python "$out/run.py" --mode "$mode" --q-repeats 16 --k-chunks 512 \
        --warmup 20 --iters 10 --label "$mode-steady" > "$out/$mode-steady.log" 2>&1
done
python_env/bin/python experiments/sdpa-l2/single-core-resident-v1/run.py \
    --mode main --q-repeats 16 --k-chunks 512 --warmup 20 --iters 10 \
    --label hifi2-fp32-main-control > "$out/main-control.log" 2>&1
for mode in fast fp32_hifi2 fp32_hifi2_cheap; do
    python_env/bin/python "$out/run.py" --mode "$mode" --q-repeats 16 --k-chunks 512 \
        --warmup 20 --iters 10 --label "$mode-reverse" > "$out/$mode-reverse.log" 2>&1
    python_env/bin/python "$out/run.py" --mode "$mode" --q-repeats 2 --k-chunks 64 \
        --warmup 2 --iters 2 --seed 1237 --label "$mode-heldout" > "$out/$mode-heldout.log" 2>&1
    python_env/bin/python "$out/run.py" --mode "$mode" --q-repeats 2 --k-chunks 512 \
        --warmup 2 --iters 2 --distribution constant_v --label "$mode-constant" > "$out/$mode-constant.log" 2>&1
done
