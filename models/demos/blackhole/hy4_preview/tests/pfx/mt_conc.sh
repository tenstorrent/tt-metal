#!/bin/bash
# mt_conc.sh <host> <nsessions> <turns> <tag>: concurrent multi-turn sessions (different content each)
H=$1; N=$2; T=$3; TAG=$4; D=/mnt/tt-data/ssinghal/hy4-logs/program/pfx
for s in $(seq 1 $N); do SEED=$s /mnt/tt-data/ssinghal/tests/tt-metal/python_env/bin/python $D/multiturn_s.py $H $T 190 > $D/ev_mt_${TAG}_s$s.out 2>&1 & done; wait
