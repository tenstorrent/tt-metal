#!/bin/bash
C=$1; V=/home/ttuser/atupe/python_env_vllm/bin/vllm; D=$(pwd)/$C
run(){ n=$1; isl=$2; osl=$3; conc=$4; np=$5
 timeout 1500 $V bench serve --backend openai --endpoint /v1/completions --model Qwen/Qwen3.6-27B --tokenizer /home/runara/models/Qwen3.6-27B --host 127.0.0.1 --port 8000 --dataset-name random --random-input-len $isl --random-output-len $osl --ignore-eos --temperature 0 --seed 0 --max-concurrency $conc --num-prompts $np --num-warmups 2 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 --save-result --result-dir $D --result-filename $n.json > $D/$n.out 2>&1
 echo "$n rc=$?"; }
run R5 300 128 1 8
