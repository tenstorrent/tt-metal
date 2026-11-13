#!/bin/bash
# usage: run.sh script.py logname
P=/mnt/tt-data/ssinghal/hy4-logs/program; S=/tmp/claude-1000/-mnt-tt-data-ssinghal-tests-tt-metal/9f313510-26a3-47cd-8736-0cc09623d5a0/scratchpad
eval "$($P/bin/slot_env.sh 0 10.82.97.44 | sed 's/^/export /')"
cd /mnt/tt-data/ssinghal/tests/tt-metal; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD:$PWD/ttnn:$S/idx OMP_NUM_THREADS=16; source python_env/bin/activate
mkdir -p $TT_METAL_CACHE
python $S/idx/$1 > $P/idx-1m/$2 2>&1
