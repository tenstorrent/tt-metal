#!/bin/bash
# Every SLA line against the server on :8000 (start it with run_server.sh). Required env: TT_METAL_DIR, HF_MODEL, OUT.
# Results: $OUT/<test>.json + .log, progress in $OUT/progress.txt; table: python summary.py $OUT
# If SERVER_PID is set, gives up when that process exits before the server comes up.
: "${TT_METAL_DIR:?}" "${HF_MODEL:?}" "${OUT:?}"
D=$(cd "$(dirname "$0")" && pwd)
cd "$TT_METAL_DIR" && source python_env/bin/activate
mkdir -p "$OUT"
until curl -sf localhost:8000/v1/models >/dev/null; do
  if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server exited, see $OUT/server.log"; exit 1; fi
  sleep 30
done
# Garbage from chips left dirty by an earlier failed process looks fine in latency numbers: refuse to bench it.
python "$D/sanity_check.py" | tee -a "$OUT/progress.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 1
b() {
  local tag=$1; shift
  echo "$(date +%T) START $tag" >> "$OUT/progress.txt"
  vllm bench serve --backend openai --base-url http://localhost:8000 --model llama70b --tokenizer "$HF_MODEL" \
    --dataset-name random --ignore-eos --top-k 32 --top-p 0.9 --temperature 0.6 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir "$OUT" --result-filename "$tag.json" "$@" > "$OUT/$tag.log" 2>&1
  echo "$(date +%T) END $tag exit=$?" >> "$OUT/progress.txt"
}
b warmup --random-input-len 1024 --random-output-len 32 --num-prompts 8 --max-concurrency 8   # not scored
b single_4k_500 --seed 7000 --random-input-len 4096 --random-output-len 500 --num-prompts 4 --max-concurrency 1
b cold_single_16k --seed 7001 --random-input-len 16384 --random-output-len 200 --num-prompts 3 --max-concurrency 1
b cold_single_32k --seed 7002 --random-input-len 32768 --random-output-len 200 --num-prompts 3 --max-concurrency 1
# 60 RPM, random prompts 2K-6K in / 250-750 out (no shared prefix: 0% cache hits)
b rpm60_mix --seed 7005 --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 120 --request-rate 1 --max-concurrency 32
# 60 RPM with the SLA's 2K shared prefix (~40% of a 5K prompt)
b rpm60_prefix40 --seed 7006 --random-prefix-len 2048 --random-input-len 3072 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 120 --request-rate 1 --max-concurrency 32
python "$D/prefix_hit_probe.py" run 120 1.0 2048 3072 > "$OUT/prefix_hit_probe.log" 2>&1
b rpm240_mix --seed 7007 --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 240 --request-rate 4 --max-concurrency 32
b c32_4k_500 --seed 7008 --random-input-len 4096 --random-output-len 500 --num-prompts 96 --max-concurrency 32
echo "$(date +%T) BENCH FINISHED" >> "$OUT/progress.txt"
