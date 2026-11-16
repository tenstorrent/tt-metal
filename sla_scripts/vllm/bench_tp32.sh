#!/bin/bash
# vLLM load tests against the TP32 server on :8000; results in ~/teja/runs/bench/<tag>.json + .log
cd ~/teja/tt-metal && source python_env/bin/activate
OUT=${OUT:-~/teja/runs/bench}; mkdir -p $OUT
TOK=/home/ubuntu/teja/weights/Llama-3.3-70B-Instruct
b() {
  local tag=$1; shift
  vllm bench serve --backend openai --base-url http://localhost:8000 --model ${MODEL:-llama70b} --tokenizer $TOK \
    --dataset-name random --ignore-eos --top-k 32 --top-p 0.9 --temperature 0.6 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir $OUT --result-filename $tag.json "$@" > $OUT/$tag.log 2>&1
  echo "=== $tag EXIT $?" >> $OUT/$tag.log
}
# single request at the SLA mean shape
b single_4k_500 --random-input-len 4096 --random-output-len 500 --num-prompts 4 --max-concurrency 1
# mean load: 32 in flight, 4K in / 500 out
b c32_4k_500 --random-input-len 4096 --random-output-len 500 --num-prompts 96 --max-concurrency 32
# mean arrival rate 60 RPM with mixed lengths (2K-6K in, 250-750 out)
b rpm60_mix --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 120 --request-rate 1 --max-concurrency 32
# peak arrival rate 240 RPM
b rpm240_mix --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 240 --request-rate 4 --max-concurrency 32
# cold long prompts at full concurrency
b c32_16k_200 --random-input-len 16384 --random-output-len 200 --num-prompts 32 --max-concurrency 32
b c8_32k_200 --random-input-len 32768 --random-output-len 200 --num-prompts 8 --max-concurrency 8
# prefix cache: 2K shared prefix + 2K unique, 4K total
b c32_prefix2k --random-prefix-len 2048 --random-input-len 2048 --random-output-len 500 --num-prompts 96 --max-concurrency 32
echo "BENCH FINISHED"
