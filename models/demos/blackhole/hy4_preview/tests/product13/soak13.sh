#!/bin/bash
# soak.sh <host> : 2 x repro (4 story prompts) + 3 rounds x (T=0, T=0.7) x accept_probe all (8 chat + 8 gsm + 8 code + 3 story) = 162 requests incl. 18 story prompts of ~4.4k tokens
cd /mnt/tt-data/ssinghal/hy4-logs/program/product13; H=$1
STALL=600 python3 repro.py $H 4 256 > logs/soak_repro.out 2>&1 || { echo STALL_REPRO; exit 3; }
for i in 1 2 3; do
 OUT=logs/soak_r${i}_T0.json timeout -s TERM 2400 python3 accept_probe.py $H all > logs/soak_r${i}_T0.out 2>&1; echo "R$i T0 rc=$? $(date +%T)" >> logs/soak.log
 TEMP=0.7 OUT=logs/soak_r${i}_T07.json timeout -s TERM 2400 python3 accept_probe.py $H all > logs/soak_r${i}_T07.out 2>&1; echo "R$i T07 rc=$? $(date +%T)" >> logs/soak.log
done
echo SOAK_DONE >> logs/soak.log
