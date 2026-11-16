#!/bin/bash
source <(sed -n '1,12p' ~/teja/bench_tp32.sh)
b single_16k_200 --random-input-len 16384 --random-output-len 200 --num-prompts 3 --max-concurrency 1
b single_32k_200 --random-input-len 32768 --random-output-len 200 --num-prompts 3 --max-concurrency 1
b rpm15_mix --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 60 --request-rate 0.25 --max-concurrency 32
b rpm60_prefix40 --random-prefix-len 2048 --random-input-len 3072 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 120 --request-rate 1 --max-concurrency 32
echo "EXTRA FINISHED"
