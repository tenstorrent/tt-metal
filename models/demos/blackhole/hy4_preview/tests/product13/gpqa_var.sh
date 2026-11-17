#!/bin/bash
# gpqa_var.sh <tag> <nrep>: wait for server, run GPQA 8q x nrep -> ev/gpqa_<tag>_r<k>_{a,b}.out
cd /mnt/tt-data/ssinghal/hy4-logs/program/product13; TAG=$1; N=${2:-2}; H=10.82.97.45; . ../serve-long/evals2/env.sh 2>/dev/null
until curl -s -m 5 http://$H:8000/health >/dev/null; do sleep 20; grep -q "RC=" logs/server_$TAG.log && { echo BOOT_FAILED > ev/gpv_$TAG.done; exit 1; }; done
for k in $(seq $N); do python3 val10.py $H gpqa ev/gpqa_${TAG}_r${k}_a.json 9000 0.9 > ev/gpqa_${TAG}_r${k}_a.out 2>&1; GPQA_OFF=4 python3 val10.py $H gpqa ev/gpqa_${TAG}_r${k}_b.json 9000 0.9 > ev/gpqa_${TAG}_r${k}_b.out 2>&1; done
echo DONE > ev/gpv_$TAG.done
