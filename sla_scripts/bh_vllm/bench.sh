#!/bin/bash
# SLA traffic tests against the server on :8000. Fresh seed per case so the prefix cache starts cold.
: "${TT_METAL_DIR:?}" "${HF_MODEL:?}" "${OUT:?}"
cd "$TT_METAL_DIR" && source python_env/bin/activate
HERE=$(dirname "$(readlink -f "$0")")
mkdir -p "$OUT"
b() {
  local tag=$1; shift
  vllm bench serve --backend openai --base-url http://localhost:8000 --model llama70b --tokenizer "$HF_MODEL" \
    --dataset-name random --ignore-eos --top-k 32 --top-p 0.9 --temperature 0.6 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" --result-filename $tag.json "$@" > "$OUT/$tag.log" 2>&1
  echo "=== $tag EXIT $?" >> "$OUT/$tag.log"
}
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 30; done
# warmup, not used in results
b warmup --seed 9000 --random-input-len 4096 --random-output-len 32 --num-prompts 8 --max-concurrency 8
b single_4k_500 --seed 9001 --random-input-len 4096 --random-output-len 500 --num-prompts 4 --max-concurrency 1
b cold_single_16k --seed 9002 --random-input-len 16384 --random-output-len 200 --num-prompts 3 --max-concurrency 1
b cold_single_32k --seed 9003 --random-input-len 32768 --random-output-len 200 --num-prompts 3 --max-concurrency 1
# SLA mean: 60 RPM, 2K-6K in, 250-750 out
b rpm60_mix --seed 9004 --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 120 --request-rate 1 --max-concurrency 128
# SLA mean with 40% prefix hit (2K shared + ~3K unique)
b rpm60_prefix40 --seed 9005 --random-prefix-len 2048 --random-input-len 3072 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 120 --request-rate 1 --max-concurrency 128
# SLA peak: 240 RPM
b rpm240_mix --seed 9006 --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 240 --request-rate 4 --max-concurrency 128
# measured prefix hit rate at 60 RPM, 40% designed
python3 "$HERE/prefix_hit_probe.py" rpm60_40pct 120 1.0 2048 3072
python3 "$HERE/summary.py" | tee "$OUT/summary.txt"
echo "BENCH FINISHED"
