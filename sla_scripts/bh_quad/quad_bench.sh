#!/bin/bash
# SLA benchmarks for the 4-Galaxy deployment, through the balancer. Required env: HF_MODEL (tokenizer), OUT.
# API (default http://localhost:9000). TESTS (default all): sanity, needle, c128 (128 users x 4K in / 500 out,
# 3 rounds), c128_16k (128 x 16K at once), c32 / c32_16k (the single-Galaxy lines, now spread over 4),
# rpm960 (4 x the 240 RPM line), mmlu (full MMLU, forced DP4 needs the server env), stats.
API=${API:-http://localhost:9000}
: "${HF_MODEL:?}" "${OUT:?}"
mkdir -p "$OUT"
D=$(cd "$(dirname "$0")" && pwd)
b() { tag=$1; shift
  vllm bench serve --backend openai --base-url "$API" --model llama70b --tokenizer "$HF_MODEL" \
    --dataset-name random --ignore-eos --top-k 32 --top-p 0.9 --temperature 0.6 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result --save-detailed \
    --result-dir "$OUT" --result-filename $tag.json "$@" > "$OUT/$tag.log" 2>&1
  echo "$(date +%T) $tag $(grep -E "Successful requests|Mean TTFT|P99 TTFT|Mean TPOT|P99 TPOT" "$OUT/$tag.log" | awk '{print $NF}' | tr '\n' ' ')"; }
for t in $(echo ${TESTS:-sanity,needle,c128,c128_16k,c32,c32_16k,rpm960,stats} | tr , ' '); do case $t in
  sanity) python "$D/../bh_tp32/sanity_check.py" "$API" > "$OUT/sanity.log" 2>&1; echo "$(date +%T) sanity exit=$?";;
  needle) python "${NEEDLE:-/mnt/nvme0/jerrywang/scripts/needle_conc.py}" "$API/v1/completions" > "$OUT/needle_conc.log" 2>&1; echo "$(date +%T) needle_conc exit=$? $(tail -1 "$OUT/needle_conc.log")";;
  warm) b warmup --random-input-len 1024 --random-output-len 32 --num-prompts 32 --max-concurrency 32;;
  c128) b c128_4k_500 --seed 7108 --random-input-len 4096 --random-output-len 500 --num-prompts 384 --max-concurrency 128;;
  c128_16k) b c128_16k_200 --seed 7109 --random-input-len 16384 --random-output-len 200 --num-prompts 128 --max-concurrency 128;;
  c32) b c32_4k_500 --seed 7008 --random-input-len 4096 --random-output-len 500 --num-prompts 96 --max-concurrency 32;;
  c32_16k) b c32_16k_200 --seed 7009 --random-input-len 16384 --random-output-len 200 --num-prompts 32 --max-concurrency 32;;
  rpm960) b rpm960_mix --seed 7107 --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 960 --request-rate 16 --max-concurrency 128;;
  stats) curl -s "$API/lb/stats" > "$OUT/lbstats.json"; python -c "import json,sys; d=json.load(open(sys.argv[1])); print('lb', {k: d[k] for k in ('requests','failed','queued_max')}, [(b['url'], b['served'], b['errors']) for b in d['backends']])" "$OUT/lbstats.json";;
esac; done
