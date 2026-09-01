#!/usr/bin/env bash
# Batch/concurrency sweep against the running vLLM server.
# Table A: concurrency scaling at fixed ISL.  Table B: max feasible conc per ISL.
set -uo pipefail

PY=/home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv/bin/python
OUT=/home/ttuser/dev/muse-glimmer/logs/batchsweep
MODEL=meta-models/Muse-Glimmer-30B
OSL=512

point() {  # isl conc nprompts tag
  local isl=$1 conc=$2 n=$3 tag=$4
  local f="${tag}_isl${isl}_conc${conc}.json"
  echo ">>> $tag  ISL=$isl  conc=$conc  n=$n  $(date +%H:%M:%S)"
  $PY -m vllm.entrypoints.cli.main bench serve \
    --host 127.0.0.1 --port 8000 \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$isl" \
    --random-output-len "$OSL" \
    --random-range-ratio 0 \
    --num-prompts "$n" \
    --max-concurrency "$conc" \
    --num-warmups 1 \
    --ignore-eos \
    --seed 1234 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,99 \
    --disable-tqdm \
    --save-result --result-dir "$OUT" --result-filename "$f" \
    >> "$OUT/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "    FAILED rc=$rc (see $OUT/${tag}.log)"; else echo "    ok -> $f"; fi
  return 0
}

# ---------- Table A: concurrency scaling at ISL 1024 ----------
for c in 1 2 4 8 16 32; do
  point 1024 "$c" $((c*2)) tableA
done

# ---------- Table B: frontier ----------
# args: pairs of "isl:conc"
for pair in "$@"; do
  isl=${pair%%:*}; conc=${pair##*:}
  if [ "$isl" -le 16384 ]; then n=$((conc*2)); else n=$conc; fi
  point "$isl" "$conc" "$n" tableB
done

echo "=== sweep complete $(date +%H:%M:%S) ==="
