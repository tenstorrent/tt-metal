#!/bin/bash
source <(sed -n '1,12p' ~/teja/bench_tp32.sh)
b cold_single_16k --seed 7001 --random-input-len 16384 --random-output-len 200 --num-prompts 3 --max-concurrency 1
b cold_single_32k --seed 7002 --random-input-len 32768 --random-output-len 200 --num-prompts 3 --max-concurrency 1
b cold_rpm15_mix --seed 7003 --random-input-len 4096 --random-output-len 500 --random-range-ratio 0.5 --num-prompts 60 --request-rate 0.25 --max-concurrency 32
b cold_c128_16k --seed 7004 --random-input-len 16384 --random-output-len 200 --num-prompts 32 --max-concurrency 32
echo "COLD FINISHED"
