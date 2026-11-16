#!/bin/bash
# Same SLA cases against tt-inference-server; fresh seeds so the prefix cache starts cold
export OUT=${OUT:-~/teja/runs/bench_tis} MODEL=$(curl -s localhost:8000/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
source <(sed -n '1,12p' ~/teja/bench_tp32.sh)
b single_4k_500 --seed 8001 --random-input-len 4096 --random-output-len 500 --num-prompts 4 --max-concurrency 1
b cold_single_16k --seed 8002 --random-input-len 16384 --random-output-len 200 --num-prompts 3 --max-concurrency 1
b cold_single_32k --seed 8003 --random-input-len 32768 --random-output-len 200 --num-prompts 3 --max-concurrency 1
b rpm15_mix --seed 8004 --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 60 --request-rate 0.25 --max-concurrency 32
b rpm60_mix --seed 8005 --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 120 --request-rate 1 --max-concurrency 32
b rpm60_prefix40 --seed 8006 --random-prefix-len 2048 --random-input-len 3072 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 120 --request-rate 1 --max-concurrency 32
b rpm240_mix --seed 8007 --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 240 --request-rate 4 --max-concurrency 32
cd ~/teja/tt-metal && source python_env/bin/activate && python3 ~/teja/prefix_hit_probe.py ${HIT_TAG:-tis_rpm60_40pct} 120 1.0 2048 3072
echo "TIS FINISHED"
