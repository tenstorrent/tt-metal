#!/usr/bin/env bash
set -euo pipefail
out=experiments/sdpa-l2/bf16-sfpu-v2
control=experiments/sdpa-l2/single-core-resident-v1
python_env/bin/python "$out/run.py" --mode fast --q-repeats 16 --k-chunks 512 \
    --warmup 20 --iters 10 --label final-steady > "$out/final-steady.log" 2>&1
python_env/bin/python "$control/run.py" --mode fast --q-repeats 16 --k-chunks 512 \
    --warmup 20 --iters 10 --label sfpu-v2-fast-control > "$out/fast-control.log" 2>&1
python_env/bin/python "$control/run.py" --mode main --q-repeats 16 --k-chunks 512 \
    --warmup 20 --iters 10 --label sfpu-v2-main-control > "$out/main-control.log" 2>&1
python_env/bin/python "$out/run.py" --mode fast --q-repeats 16 --k-chunks 512 \
    --warmup 20 --iters 10 --label final-reverse > "$out/final-reverse.log" 2>&1
python_env/bin/python "$out/run.py" --mode accurate --q-repeats 8 --k-chunks 64 \
    --warmup 2 --iters 2 --seed 1237 --label accurate-regression > "$out/accurate-regression.log" 2>&1
python_env/bin/python -m tracy -r --profiler-capture-perf-counters fpu \
    -o "$out/profile-final" "$out/run.py" --mode fast --q-repeats 8 \
    --k-chunks 512 --iters 0 --label profile-final > "$out/profile-final.log" 2>&1
python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
    --kv-lens 262144 --full --heads 10 --q-len 512 --query-sampling spread \
    --q-chunk 128 --k-chunks 512 --variants hifi2 --seed 1236 \
    --benchmark-warmup 40 --benchmark-iters 10 --check-full-output \
    --label new-fast-full --output "$out/new-fast-full.jsonl" > "$out/new-fast-full.log" 2>&1
